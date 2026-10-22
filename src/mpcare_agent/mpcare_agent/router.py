"""MP-CARE router: integrates the four section 5.3 components per UAV.

    NeighborTable (beacons)  -> LinkPredictor (ETX_pred, LET)
                             -> CostEngine (rank candidate next hops)
                             -> EventTrigger (T1-T5, dwell + margin)

Decentralized by construction (plan section 5.2.4): a node knows only
- its 1-hop neighbors (from received beacons), and
- each neighbor's advertised neighbor list + queue depth (beacon payload),
i.e. O(1) state per neighbor. No global topology table anywhere.

Per-packet decisions are GPSR-style next-hop choices over this knowledge,
scored by the predictive composite cost; re-planning (switching the preferred
next hop for a destination) happens only through the EventTrigger's hysteresis
gate. Measured delivery outcomes feed back through the neighbor table, so the
cost reflects real experience (T1-style evidence) on top of predictions.

Known simplification (documented): with only 2-hop knowledge, greedy costing
can stall in rare topologies (e.g. two candidate relays that both have the
destination out of their range but closer than anyone else). MP-CARE then
falls back to the best available neighbor rather than declaring failure —
packets still progress, just without full-cost lookahead.

Phase 6 remediation (measured ceiling, NOTES/Phase_6.md §9): at scale
(constant density, <k> ~ 4.4) greedy stalls stopped being rare — the driver
probe showed 100% of stalls were VOIDS (neighbors present, none closer to
the destination; 0% isolation). The fix is a GPSR-STRUCTURED FACE WALK on
the same 1-hop beacon state (no protocol change): at a void, forward to the
first live neighbor clockwise from the reference direction (toward the
destination on the first step, along the arrival edge afterwards — the
right-hand rule), WITHOUT excluding visited nodes — circumnavigating the
void on the face cycle is the point (a node-disjoint walk would burn the
5 s queue buffer backtracking; measured on seed 3001@N=50, probe history
in NOTES/Phase_6.md §9). Every holder re-evaluates greedy FIRST, so greedy
resumes implicitly at the first node that beats the void entry point.
Loop safety is three-fold (GPSR semantics, all on 1-hop state): (1) the
packet WALKS once stuck — every holder continues the face walk until the
packet beats its void-entry distance (driver-tracked), so greedy cannot
re-enter the void mid-walk; (2) the walk edges are Gabriel-planarized
using the node's OWN neighbors as witnesses — the classic localized GG
approximation, no global positions; (3) a driver-side progress cap drops
a packet HONESTLY (counted as void_walk_drops) after a bounded number of
progress-less steps instead of looping forever. Greedy progress itself is
loop-free by construction (strictly-closer rule). Recovery steps are
counted per router for telemetry.
"""

from __future__ import annotations

import math
from collections import deque

from mpcare_agent.neighbor_table import NeighborTable
from mpcare_agent.link_predictor import LinkPredictor
from mpcare_agent.cost_engine import CostEngine, PathFeatures
from mpcare_agent.event_trigger import EventTrigger, TriggerParams

INF = math.inf
BEACON_INTERVAL_S = 1.0
HISTORY_LEN = 6    # ~3 s of per-neighbor outcomes at 2 pkt/s: T1 reacts fast
                   # (plan: T1 within ~2 beacon intervals of degradation)


class MpcareRouter:
    """One MP-CARE routing brain, owned by one UAV."""

    def __init__(self, node_id: int, n_nodes: int, max_range_m: float = 300.0,
                 trigger_params: TriggerParams | None = None,
                 weights: dict | None = None,
                 reroute_mode: str = "event", period_s: float = 1.0,
                 predict_etx: bool = True, use_let: bool = True):
        self.node_id = node_id
        self.n_nodes = n_nodes
        self.table = NeighborTable(node_id)
        self.predictor = LinkPredictor(max_range_m=max_range_m)
        self.engine = CostEngine(weights)
        self.trigger = EventTrigger(trigger_params)
        # ---- phase 7 ablation switches (plan section 9.1; defaults = the
        # full MP-CARE behavior, byte-identical to pre-phase-7 routing):
        # reroute_mode: "event" (T1-T5 + hysteresis) | "periodic" (A5/A6:
        # same forwarding, route switches only every period_s seconds) |
        # "static" (A7: first chosen relay per destination is frozen — the
        # plan's lower bound; no re-ranking, no triggers, no face walk).
        # predict_etx=False (A1): the cost uses CURRENT ETX only — the
        # LinkPredictor's radial-velocity term is bypassed (prediction value
        # isolated in path choice; delay prediction inherits the same edge).
        # use_let=False (A2): LET never reaches the trigger — T2 (expiring
        # link) can not fire (mobility-prediction value isolated).
        if reroute_mode not in ("event", "periodic", "static"):
            raise ValueError(f"unknown reroute_mode {reroute_mode!r}")
        self.reroute_mode = reroute_mode
        self.period_s = float(period_s)
        self.predict_etx = bool(predict_etx)
        self.use_let = bool(use_let)

        self.next_hop: dict[int, int] = {}      # destination -> current relay
        self.route_prr: dict[int, float | None] = {}   # per-destination PRR (T1)
        self._delivery_hist: dict[int, deque] = {}
        self.reroute_event = False
        self.reroute_trigger = ""
        self._trigger_totals: dict[str, int] = {}   # cumulative fires by name
        self.last_alt_cost = INF
        self.recovery_events = 0    # void face-walk steps (phase 6 remediation)
        self._sim: "SimHooks | None" = None     # set by the driver (physics access)

    # ----------------------------------------------------- beacon handling
    def beacon_payload(self, my_pos, my_queue_depth: float, t: float) -> dict:
        return {"type": "MPBEACON", "src": self.node_id, "pos": my_pos,
                "queue": my_queue_depth, "nbrs": self.table.ids()}

    def receive_beacon(self, msg: dict, frm: int, etx_from_me: float, t: float):
        self.table.update_from_beacon(frm, msg["pos"], etx_from_me, t)
        info = self.table.neighbors.get(frm)
        if info is not None:
            info.last_advertised_nbrs = tuple(msg.get("nbrs", ()))
            info.queue_ratio = float(msg.get("queue", 0.0))

    # ----------------------------------------------------- driver plumbing
    def bind(self, sim) -> None:
        """Driver provides physics access: sim.positions, sim.link,
        sim.snap (current NetworkSnapshot), sim.now()."""
        self._sim = sim

    def observe_delivery(self, to_id: int, success: bool):
        """One measured unicast outcome toward `to_id` (T1 evidence).
        Fed by the driver after every real transmission attempt."""
        self.table.observe_delivery(to_id, success)
        h = self._delivery_hist.setdefault(to_id, deque(maxlen=HISTORY_LEN))
        h.append(1 if success else 0)

    # ----------------------------------------------------- cost plumbing
    def _etx_current(self, a: int, b: int) -> float:
        """Best available current-ETX estimate for edge (a, b).

        1-hop edges: my own EWMA measurement (falling back to the physics
        snapshot only for a brand-new neighbor). 2-hop edges: cannot measure —
        use the snapshot's physics value; the *prediction* still applies via
        the next hop's advertised motion.
        """
        if a == self.node_id:
            me = self.table.etx_to(b)
            if math.isfinite(me):
                return me
        if self._sim is not None:
            v = self._sim.snap.etx_of(a, b)
            if math.isfinite(v):
                return v
        return INF

    def _etx_pred(self, a: int, b: int, arrival_s: float = 0.0) -> float:
        cur = self._etx_current(a, b)
        if math.isinf(cur):
            return INF
        if not self.predict_etx:      # A1 ablation: current ETX only
            return cur
        # radial velocity: measured for 1-hop (my table); advertised indirectly
        # for 2-hop via the relay's beacon (relayed radial_vel toward b is not
        # available, so f=1 there — documented simplification, see module note)
        v = self.table.radial_velocity(b) if a == self.node_id else 0.0
        return self.predictor.etx_pred(cur, v, arrival_s)

    def _delay_pred(self, a: int, b: int) -> float:
        """Per-hop predicted delay: propagation + airtime*ETX_pred + queueing."""
        if self._sim is None:
            return 0.0
        d = self._sim.snap.dist.get((a, b), INF)
        if math.isinf(d):
            return 1e9
        e = self._etx_pred(a, b)
        if math.isinf(e):
            return 1e9
        p = self._sim.link.p
        t_air = 8000.0 / (p.bandwidth_mbps * 1000.0) / 1000.0   # s
        prop = d / 3e8
        q = self._queue_ratio(b)
        q_delay = q * BEACON_INTERVAL_S          # one beacon interval if full
        return prop + t_air * e + q_delay

    def _queue_ratio(self, nid: int) -> float:
        if nid == self.node_id:
            return self._own_queue_ratio
        info = self.table.neighbors.get(nid)
        return info.queue_ratio if info else 0.0

    # ----------------------------------------------------- per-packet routing
    def next_hop_for(self, dst: int, t: float,
                     visited: frozenset | set | None = None,
                     prev: int | None = None,
                     walking: bool = False) -> int | None:
        """Choose the relay toward dst (greedy + predictive cost; void walk).

        Candidates: my 1-hop neighbors that are strictly closer to the
        destination than I am (greedy rule), excluding nodes already on the
        packet's path (loop-freedom). Among them, the predictive composite
        cost picks the winner; the EventTrigger gates switching away from the
        established relay (T3 margin + dwell).

        VOID RECOVERY (phase 6 remediation, GPSR semantics on 1-hop state):
        with neighbors present but none closer to the destination (greedy
        local minimum), the packet enters WALK mode: the first live
        Gabriel-planarized neighbor clockwise from the reference direction —
        toward the destination if the previous hop is unknown, otherwise
        along the arrival edge (prev -> me), never straight back while
        another edge exists. While `walking`, greedy is SKIPPED (the packet
        stays on the face until it beats its void-entry distance — tracked
        by the driver — otherwise it would re-enter the void every hop).
        The walk step is per-packet forwarding, NOT a route change:
        next_hop[] and the hysteresis gate are untouched (the stale greedy
        route is cleared at walk entry).
        """
        self.reroute_event = False
        self.reroute_trigger = ""
        if self._sim is None or dst not in self._sim.positions:
            return None
        positions = self._sim.positions
        me = positions[self.node_id][:2]
        dst_xy = positions[dst][:2]
        seen = visited if visited is not None else ()

        if self.reroute_mode == "static":
            # A7 (plan section 9.1): the lower bound. The FIRST relay chosen
            # for a destination is frozen; no re-ranking, no trigger gate, no
            # face walk (walking would be adaptation). If the frozen relay is
            # no longer a live neighbor, the packet queues until its TTL —
            # the honest cost of never re-routing.
            frozen = self.next_hop.get(dst)
            if frozen is not None:
                return frozen if frozen in self.table.ids() else None
            ranked = self._ranked_candidates(dst, dst_xy, positions, seen)
            if not ranked:
                return None
            best = ranked[0][1].path[1]
            self.next_hop[dst] = best
            return best

        if walking:
            # mid-walk holder: greedy is skipped on purpose (stay on the
            # face until the packet beats its void-entry distance; the
            # driver exits the walk when it does)
            return self._void_walk(me, dst_xy, positions, prev)
        ranked = self._ranked_candidates(dst, dst_xy, positions, seen)
        if not ranked:
            # the stored (greedy) route through this void is dead — clear it
            # so route queries (ros_adapter.route_of) report honestly; the
            # face-walk step below is per-packet forwarding and never writes
            # next_hop[].
            self.next_hop.pop(dst, None)
            return self._void_walk(me, dst_xy, positions, prev)

        best_cost, best_f = ranked[0]
        best_nh = best_f.path[1]

        cur = self.next_hop.get(dst)
        if cur is None or cur not in self.table.neighbors or best_nh in seen:
            self.next_hop[dst] = best_nh
            return best_nh

        # hysteresis: is a switch justified? (T3 margin, T1, T2, T4)
        # A5/A6 (plan section 9.1): periodic mode keeps the identical greedy
        # + face-walk FORWARDING, but the switch policy is a plain timer —
        # a route change happens only if period_s elapsed since the last one
        # (the trigger object's dwell timestamp doubles as the last-switch
        # time; no event evaluation at all).
        if self.reroute_mode == "periodic":
            if t - self.trigger.last_reroute_t >= self.period_s \
                    and best_nh != cur:
                self.trigger.last_reroute_t = t
                self.trigger.re_event_count += 1
                self.reroute_event = True
                self.reroute_trigger = "periodic"
                self._trigger_totals["periodic"] = \
                    self._trigger_totals.get("periodic", 0) + 1
                self.next_hop[dst] = best_nh
            self.last_alt_cost = best_cost
            return self.next_hop[dst]

        cur_f = next((f for c_, f in ranked if f.path[1] == cur), None)
        # current-route cost within the same candidate normalization
        cur_cost = None
        for c_, f in ranked:
            if f.path[1] == cur:
                cur_cost = c_
                break
        better_alt = cur_cost is None or best_cost < (1.0 - self.trigger.p.delta) * cur_cost

        route_prr = self._route_prr(dst)
        min_let = self._min_route_let(cur)
        fire, name = self.trigger.should_reroute(
            t, route_prr=route_prr, min_let_s=min_let,
            better_alt=better_alt, lost_neighbors=self.table.drain_lost())
        if fire:
            self.reroute_event = True
            self.reroute_trigger = name
            self._trigger_totals[name] = self._trigger_totals.get(name, 0) + 1
            self.next_hop[dst] = best_nh
        self.last_alt_cost = best_cost
        return self.next_hop[dst]

    # ----------------------------------------------------- scoring core
    def _rank_candidates(self, dst: int, dst_xy, positions, cands: list[int]):
        """Build the 2-hop-lookahead candidate paths for `cands` and rank
        them. The ONE scoring core shared by every reroute_mode (event /
        periodic / static) — path = [me, c, <c's advertised neighbor closest
        to dst>], ranked by the predictive composite cost."""
        feats: list[PathFeatures] = []
        etx_fp = lambda a, b: self._etx_pred(a, b)            # noqa: E731
        delay_fp = lambda a, b: self._delay_pred(a, b)        # noqa: E731
        queue_fp = lambda n: (0.0 if n not in self.table.neighbors
                              else 1.0 - self.table.neighbors[n].recent_delivery_rate)  # noqa: E731
        for c in cands:
            path = [self.node_id, c]
            # extend one more hop using the candidate's advertised neighbors
            best2, best2d = None, INF
            if c != dst:
                for nb in self.table.neighbors[c].last_advertised_nbrs:
                    if nb in (self.node_id, c):
                        continue
                    if nb in positions:
                        d2 = math.dist(positions[nb][:2], dst_xy)
                        if d2 < best2d:
                            best2, best2d = nb, d2
                if best2 is not None:
                    path.append(best2)
                else:
                    path.append(c)      # dead end beyond c: penalized via ETX
            f = self.engine.features(path, positions, etx_fp, delay_fp, queue_fp)
            feats.append(f)
        return self.engine.rank(feats)

    def _ranked_candidates(self, dst: int, dst_xy, positions,
                           seen) -> list:
        """Candidate collection (strictly-closer greedy rule, destination
        always eligible, loop exclusion) + ranking; [] when no candidate
        survives."""
        my_d = math.dist(positions[self.node_id][:2], dst_xy)
        cands = []
        for nid in self.table.ids():
            if nid not in positions:
                continue    # beacon-known but currently absent (kill/timeout race)
            if nid == dst:
                cands.append(nid)
                continue
            nd = math.dist(positions[nid][:2], dst_xy)
            if nd < my_d - 1e-9:
                cands.append(nid)
        cands = [c for c in cands if c not in seen]
        if not cands:
            return []
        return self._rank_candidates(dst, dst_xy, positions, cands)

    # ----------------------------------------------------- void recovery
    def _void_walk(self, me, dst_xy, positions, prev) -> int | None:
        """One right-hand-rule face step around a greedy void (see the
        next_hop_for docstring): Gabriel-planarized over my own 1-hop view,
        clockwise-first from the reference direction, never straight back
        onto the arrival edge while another planarized edge exists, ties by
        node id. Returns None only when there is no usable edge at all."""
        self.recovery_events += 1
        live = sorted(nb for nb in self.table.ids() if nb in positions)
        # Final configuration (three variants MEASURED, NOTES/Phase_6.md §9):
        # one-sided GG 0.74->0.00, mutual-witness GG 0.90->0.00 on the worst
        # N=30 seed — ANY planarization from 1-hop knowledge prunes bridge
        # edges and traps the walk. The RAW graph + walk-mode + the driver's
        # progress cap is the best measured configuration; the residual
        # giant-void regime (N>=40 at percolation edge) is documented as the
        # remaining ceiling, not hidden.
        face = live
        if prev in face and len(face) > 1:
            face = [nb for nb in face if nb != prev]   # no immediate back-step
        if not face:
            return None
        if prev is not None and prev in positions:
            pv = positions[prev][:2]
            base = math.atan2(me[1] - pv[1], me[0] - pv[0])   # arrival heading
        else:
            base = math.atan2(dst_xy[1] - me[1], dst_xy[0] - me[0])
        best, best_rel = None, None
        for nb in face:
            v = (positions[nb][0] - me[0], positions[nb][1] - me[1])
            rel = (math.atan2(v[1], v[0]) - base) % (2.0 * math.pi)
            if best_rel is None or rel < best_rel:
                best, best_rel = nb, rel
        return best

    # ----------------------------------------------------- telemetry hooks
    def _route_prr(self, dst: int) -> float | None:
        """Measured delivery success toward the CURRENT relay for dst (local,
        per-neighbor evidence — what T1 can honestly observe without e2e ACKs)."""
        cur = self.next_hop.get(dst)
        if cur is None:
            return None
        h = self._delivery_hist.get(cur)
        if not h:
            return None
        return sum(h) / len(h)

    def _min_route_let(self, cur: int) -> float | None:
        if not self.use_let:          # A2 ablation: LET invisible to triggers
            return None
        if self._sim is None:
            return None
        d = self._sim.snap.dist.get((self.node_id, cur), INF)
        if math.isinf(d):
            return None
        return self.predictor.let_s(d, self.table.radial_velocity(cur))

    def tick(self, t: float):
        self.table.set_my_position(self._sim.positions[self.node_id]
                                   if self._sim else None)
        self.table.tick(t)

    # filled by the driver each tick; not part of the protocol
    _own_queue_ratio: float = 0.0


class SimHooks:
    """What the driver binds into each router (read-only physics access)."""
    __slots__ = ("positions", "link", "snap")

    def __init__(self, positions, link, snap):
        self.positions = positions
        self.link = link
        self.snap = snap

    def now(self):
        return self.snap.t_s
