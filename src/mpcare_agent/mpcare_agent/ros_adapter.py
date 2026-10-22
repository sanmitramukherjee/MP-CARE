"""MP-CARE swarm adapter for ROS 2 (the Phase 4 gate, Task B).

Pure-Python glue that drives the Phase 3 ``MpcareRouter`` cores from
ROS-shaped inputs — the ``/swarm/topology`` JSON blob plus per-UAV odometry —
using EXACTLY the per-tick sequence of ``scripts/run_sim.py``:

    1. one router ``tick(t)`` per alive UAV (beacon clock + expiry),
    2. a beacon exchange over the live graph (one hop per tick, physically
       gated by ``LinkModel.unicast_success``),
    3. my-position refresh on every router's neighbor table,
    4. route decisions per (src, dst) on demand (``next_hop_for``).

No ``rclpy`` in this file: the ROS node (``planner_node.py``) feeds inputs in
and reads decisions out, which keeps the whole logic unit-testable on Windows
without a ROS install — see ``tests/test_phase4_ros.py``.

Honest scope (mirrors the headless lab):
- Positions come from odometry (ground truth in simulation). MP-CARE is
  decentralized over *beacons*; odometry plays the role the physics
  snapshot plays in the headless driver.
- A UAV whose odometry goes silent for ``stale_s`` seconds is pruned (the
  node-failure case): its beacons stop, its routes break, packets in flight
  die with it — the same semantics as ``run_sim.py``'s kill test.
- Delivery evidence for T1/ETX EWMA comes from ``observe_delivery``; on the
  ROS side the caller reports per-transmission outcomes (the beacon exchange
  feeds measured ETX the same way the headless loop does).
"""

from __future__ import annotations

import math

from mpcare_agent.router import MpcareRouter, SimHooks
from network_model.model import LinkModel, NetworkParams, NetworkSnapshot
from network_model.topology import parse_blob, prune_stale


class MpcareSwarmAdapter:
    """Owns one router per UAV and advances them with ROS-shaped inputs."""

    def __init__(self, n_uav: int, max_range_m: float = 300.0,
                 stale_s: float = 3.0, rng_seed: int = 1):
        self.n_uav = n_uav
        self.stale_s = stale_s
        params = NetworkParams(rng_seed=rng_seed, max_range_m=max_range_m)
        self.link = LinkModel(params)
        self.routers: dict[int, MpcareRouter] = {
            i: MpcareRouter(i, n_uav, max_range_m=max_range_m)
            for i in range(n_uav)}
        # per-UAV odometry bookkeeping
        self.odom: dict[int, tuple[float, float, float]] = {}
        self.last_odom_t: dict[int, float] = {}
        self.positions: dict[int, tuple[float, float, float]] = {}   # alive set
        self.t = 0.0
        # route decisions published since the last tick (for /uav_i/route)
        self.route_events: list[dict] = []

    # ------------------------------------------------------------ inputs
    def on_odom(self, uav_id: int, pos: tuple[float, float, float], t_s: float):
        self.odom[uav_id] = (float(pos[0]), float(pos[1]), float(pos[2]))
        self.last_odom_t[uav_id] = float(t_s)

    def on_topology(self, blob_json: str):
        """Advance every router by one tick, exactly like run_sim.py's
        mpcare branch. Call this once per received /swarm/topology message
        (the network-model node publishes at the beacon interval)."""
        t, links, etx = parse_blob(blob_json)
        self.t = t
        alive = prune_stale(self.odom, self.last_odom_t, t, self.stale_s)
        positions = {uid: alive[uid] for uid in sorted(alive)}
        self.positions = positions
        snap = NetworkSnapshot(positions, self.link, t)
        hooks = SimHooks(positions, self.link, snap)

        # 1) router ticks (skip dead nodes, as in the headless loop)
        for uid, r in self.routers.items():
            if uid not in positions:
                continue
            r.bind(hooks)
            r._own_queue_ratio = 0.0
            r.tick(t)

        # 2) beacon exchange over the live graph, gated by the radio
        for uid, r in self.routers.items():
            if uid not in positions:
                continue
            payload = r.beacon_payload(positions[uid], r._own_queue_ratio, t)
            for nb in sorted(links.get(uid, ())):
                if nb not in positions:
                    continue
                d = math.dist(positions[uid][:2], positions[nb][:2])
                if self.link.unicast_success(d, uid, nb):
                    self.routers[nb].receive_beacon(
                        payload, uid, snap.etx_of(uid, nb), t)

        # 3) my-position refresh
        for uid, r in self.routers.items():
            if uid in positions:
                r.table.set_my_position(positions[uid])

        # 4) route decisions for every alive source (greedy toward the far
        #    end); collect triggers so the node can publish route changes
        self.route_events = []
        for src in sorted(positions):
            r = self.routers[src]
            dst = self.n_uav - 1
            if dst == src or dst not in positions:
                continue
            before = r.next_hop.get(dst)
            nh = r.next_hop_for(dst, t)
            if nh is not None and nh != before:
                self.route_events.append({
                    "uav_id": src, "dst": dst, "next_hop": nh,
                    "trigger": r.reroute_trigger or "new-route",
                    "t": round(t, 2)})

    def alive_ids(self) -> list[int]:
        """UAVs alive at the last topology tick (odometry fresh)."""
        return sorted(self.positions)

    # ---------------------------------------------------------- outputs
    def route_of(self, uav_id: int, dst: int | None = None) -> list[int] | None:
        """The greedy route path from uav_id toward dst (default: last UAV),
        walking the routers' current next-hop choices. Returns None when no
        viable relay exists."""
        if dst is None:
            dst = self.n_uav - 1
        if uav_id not in self.positions or dst not in self.positions:
            return None
        path = [uav_id]
        cur, seen = uav_id, {uav_id}
        while cur != dst:
            r = self.routers.get(cur)
            if r is None:
                return None
            nh = r.next_hop.get(dst)
            if nh is None or nh in seen:
                return None          # no route (yet) or loop — report honestly
            path.append(nh)
            seen.add(nh)
            cur = nh
        return path

    def trigger_counters(self) -> dict[str, int]:
        agg: dict[str, int] = {}
        for r in self.routers.values():
            for k, v in (r._trigger_totals or {}).items():
                agg[k] = agg.get(k, 0) + v
        return agg

    # ------------------------------------------------------- telemetry row
    def telemetry_row(self, uav_id: int, **kw) -> dict | None:
        """One §3.4-schema row for uav_id (None when the UAV is not alive).

        Required from the caller: position, velocity, target_waypoint,
        pdr_instantaneous, latency_avg_ms, throughput_kbps, queue_depth.
        Filled here: neighbors, neighbor_etx, neighbor_let, route cost/hops,
        reroute flags, min_separation.
        """
        r = self.routers.get(uav_id)
        pos = self.positions.get(uav_id) or self.odom.get(uav_id)
        if r is None or pos is None:
            return None
        nbs = sorted(r.table.ids())
        etx = [r.table.etx_to(n) for n in nbs]
        let = [r.predictor.let_s(
                   math.dist(pos[:2], r.table.neighbors[n].last_pos[:2]),
                   r.table.radial_velocity(n))
               for n in nbs]
        dst = self.n_uav - 1
        route = self.route_of(uav_id, dst) or []
        cost = 0.0
        for a, b in zip(route, route[1:]):
            e = r._etx_current(a, b)
            cost += e if math.isfinite(e) else 0.0
        sep = min((math.dist(pos[:2], p2[:2]) for o, p2 in self.positions.items()
                   if o != uav_id), default=0.0)
        row = {
            "uav_id": uav_id,
            "neighbors": nbs,
            "neighbor_etx": etx,
            "neighbor_let": let,
            "route_hop_count": max(0, len(route) - 1),
            "current_route_cost": round(cost, 2),
            "reroute_event": bool(r.reroute_event),
            "reroute_trigger": r.reroute_trigger or "",
            "min_separation_m": round(sep, 2),
        }
        # NOTE: `algorithm` / `experiment` are deliberately NOT here — the
        # logger node stamps them (a duplicate kwarg would crash TrialLogger.log).
        row.update(kw)
        return row
