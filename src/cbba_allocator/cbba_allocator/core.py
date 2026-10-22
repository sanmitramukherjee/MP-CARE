"""Communication-cost-aware task allocation — Phase 5 Extension B (plan §7.2).

Base algorithm: CBBA — Consensus-Based Bundle Algorithm, Choi, Brunet &
How, "Consensus-Based Decentralized Auctions for Robust Task Allocation"
(IEEE T-RO 25(4), 2009). Implementation honesty: this is the DECISION half
of CBBA — the greedy bundle construction with marginal-gain bidding and
conflict resolution by (path-cost, id) lexicographic win messages over the
comms graph. The plan benchmark exercises allocation under packet loss on
the control channel, which is exactly this half; the full asynchronous
local-information-revamp phase of the original paper is future work (each
agent here re-bids from scratch each round on fresh knowledge — the same
monotonic-bid semantics, one round = one consensus exchange).

The §7.2 extension: the bid for task T gains a routing-cost term

    b_iT = -w_d * D_iT - w_t * T_iT - w_c * C_route(i, T)

where C_route is MP-CARE's route cost from UAV i to task T (greedy path
over the CURRENT neighbor knowledge, ETX-weighted — the same cost
MpcareRouter optimizes). With w_c = 0 this is plain CBBA (baseline bit-exact
at w_d=w_t defaults), which the A/B benchmark relies on.

Loss model: bid messages cross the SAME §3.3 channel (LinkModel.unicast_success
at the inter-agent distance). A lost bid = the bidder's knowledge is missing
that round — the decentralized robustness CBBA is famous for; the benchmark
measures what that costs in completion rate, with and without the comm-aware
bid. Pure and deterministic (RNG only through the injected LinkModel seed).
"""

from __future__ import annotations

import math


class Task:
    """One task: 2-D position + service duration (plan §7.2 benchmark)."""

    def __init__(self, task_id: int, pos: tuple[float, float], service_s: float = 10.0):
        self.task_id = task_id
        self.pos = pos
        self.service_s = service_s


UNREACHABLE_PENALTY_M = 3000.0   # bid-space penalty when no route exists
                                 # (5x max_range): an agent that cannot reach
                                 # the task over the comms graph does not bid
                                 # meaningfully for it — the honest
                                 # decentralized semantics.


def route_cost_to(uav_pos: tuple[float, float], task_pos: tuple[float, float],
                  links: dict[int, set[int]], positions: dict[int, tuple],
                  src: int, snap_etx=None,
                  unreachable_penalty: float = 0.0) -> float:
    """C_route(i, T): MP-CARE-style greedy path cost from UAV `src` toward the
    task position over the current comms graph. Greedy next-hop = neighbor
    strictly closer to the task (id order, deterministic); ETX-weighted when
    snap_etx given.

    unreachable_penalty: added when the greedy walk STALLS before reaching
    the task (no neighbor strictly closer). With 0.0 the raw geometric cost
    is returned (used by metrics); bids pass UNREACHABLE_PENALTY_M so an
    agent without a connected route effectively withdraws from that task."""
    cost = 0.0
    cur, cur_pos = src, uav_pos[:2]
    seen = {src}
    while math.dist(cur_pos, task_pos[:2]) > 1e-9:
        my_d = math.dist(cur_pos, task_pos[:2])
        best, best_key = None, None
        for nb in sorted(links.get(cur, ())):
            if nb in seen:
                continue
            d = math.dist(positions[nb][:2], task_pos[:2])
            if d >= my_d - 1e-9:
                continue
            e = snap_etx(cur, nb) if snap_etx else 1.0
            key = (e * d, nb)
            if best_key is None or key < best_key:
                best, best_key = nb, key
        if best is None:
            cost += my_d + unreachable_penalty   # greedy stall: cannot serve
            break
        cost += (snap_etx(cur, best) if snap_etx else 1.0) * \
            math.dist(cur_pos, positions[best][:2])
        seen.add(best)
        cur, cur_pos = best, positions[best][:2]
    return cost


class CbbaAgent:
    """One bidder: bundle + win list over its OWN knowledge only."""

    def __init__(self, uav_id: int, w_d: float = 1.0, w_t: float = 0.5,
                 w_c: float = 0.0):
        self.uav_id = uav_id
        self.w_d, self.w_t, self.w_c = w_d, w_t, w_c
        self.bundle: list[int] = []
        self.win: dict[int, tuple[float, int]] = {}   # task -> (score, bidder)

    # ---- knowledge injection (what this agent can see THIS round) --------
    def set_view(self, positions: dict[int, tuple], links: dict[int, set[int]],
                 snap_etx=None):
        self.view_positions = positions
        self.view_links = links
        self.view_etx = snap_etx

    # ---- bid -------------------------------------------------------------
    def bid(self, task: Task) -> float:
        """Score for `task` (higher = better). Plain CBBA marginal gain with
        the §7.2 comm-cost term subtracted. Time-of-arrival term uses the
        agent's max speed proxy: T_iT ~ travel time at 8 m/s (plan default).
        set_view() MUST be called before bidding (uses this round's view).

        Benchmark note (NOTES/Phase_5.md): tasks are anchored at NODE
        positions (other agents' stations), so C_route is a clean multi-hop
        walk that terminates at the host — a relay-intermediate design made
        the nearest agent win under BOTH bid types by construction (its bid
        degenerated to the bare service leg), which is why that variant was
        scrapped."""
        pos = self.view_positions[self.uav_id]
        d = math.dist(pos[:2], task.pos)
        t_arr = d / 8.0
        if self.w_c > 0.0:
            c_route = route_cost_to(pos, task.pos, self.view_links,
                                    self.view_positions, self.uav_id,
                                    self.view_etx,
                                    unreachable_penalty=UNREACHABLE_PENALTY_M)
        else:
            c_route = 0.0
        return -(self.w_d * d + self.w_t * t_arr + self.w_c * c_route)

    # ---- one decision round (bundle build + local win table) -------------
    def decide(self, tasks: list[Task], n_bundle: int = 2) -> dict[int, tuple[float, int]]:
        self.bundle = []
        self.win = {}
        remaining = list(tasks)
        while len(self.bundle) < n_bundle and remaining:
            best_t, best_bid = None, None
            for t in remaining:
                if t.task_id in self.win:
                    continue
                b = self.bid(t)
                if best_bid is None or b > best_bid:
                    best_t, best_bid = t, b
            if best_t is None:
                break
            self.bundle.append(best_t.task_id)
            self.win[best_t.task_id] = (best_bid, self.uav_id)
            remaining.remove(best_t)
        return self.win


def run_consensus(agents: list[CbbaAgent],
                  links: dict[int, set[int]],
                  link_success=None,
                  epochs: int = 12) -> list[dict[int, tuple[float, int]]]:
    """Decentralized consensus over the comms graph, the CBBA conflict rule
    applied iteratively: one exchange per epoch; a task's belief moves toward
    the highest (score, -id) bid SEEN SO FAR through 1-hop exchanges only.

    `link_success(d_m, a, b)` models the §3.3 channel per edge per epoch
    (False = that exchange silently dropped; a/b identify the pair so the
    benchmark can apply per-pair impairments — Phase 4's set_link_loss
    semantics). This is the honesty-critical mechanism: loss degrades
    beliefs by BLOCKING propagation — no global shortcut anywhere. With 0%
    loss and epochs >= graph diameter, beliefs converge to the
    full-information allocation (asserted in tests).

    Returns each agent's final belief table (the benchmark reads the HOST's
    table — only the host's component can execute the task)."""
    belief: list[dict[int, tuple[float, int]]] = [
        {tid: (score, bidder) for tid, (score, bidder) in a.win.items()}
        for a in agents]
    id_of = {a.uav_id: k for k, a in enumerate(agents)}
    for _ in range(epochs):
        new_entries: list[list[tuple[int, int, float, int]]] = [[] for _ in agents]
        for a in agents:
            k = id_of[a.uav_id]
            for nb in sorted(links.get(a.uav_id, ())):
                j = id_of.get(nb)
                if j is None:
                    continue
                if link_success is not None:
                    d = math.dist(a.view_positions[a.uav_id][:2],
                                  a.view_positions[nb][:2])
                    if not link_success(d, a.uav_id, nb):
                        continue            # message lost: nothing crosses
                for tid, (score, bidder) in belief[j].items():
                    new_entries[k].append((tid, j, score, bidder))
        for k, entries in enumerate(new_entries):
            for tid, _j, score, bidder in entries:
                cur = belief[k].get(tid)
                if cur is None or (score, -bidder) > (cur[0], -cur[1]):
                    belief[k][tid] = (score, bidder)
    return belief


class AllocationResult:
    """Metrics per plan §7.2. Completion is measured against the geometry:
    a task counts as completed when the winning bidder is the CLOSEST agent
    to it in the true configuration (the full-information optimum, since the
    bid score is monotone in distance) — no arbitrary service radius."""

    def __init__(self, allocation: dict[int, int], tasks: list[Task],
                 positions: dict[int, tuple]):
        self.allocation = allocation
        self.tasks = tasks
        self.positions = positions

    def completion_rate(self, tol_m: float = 1e-6) -> float:
        """Fraction of tasks won by the nearest agent (true positions)."""
        if not self.tasks:
            return 0.0
        ok = 0
        for t in self.tasks:
            u = self.allocation.get(t.task_id)
            if u is None or u not in self.positions:
                continue
            nearest = min(self.positions,
                          key=lambda j: math.dist(self.positions[j][:2], t.pos))
            if math.dist(self.positions[u][:2], t.pos) <= \
                    math.dist(self.positions[nearest][:2], t.pos) + tol_m:
                ok += 1
        return ok / len(self.tasks)

    def total_route_cost(self, links, snap_etx=None,
                         unreachable_penalty: float = 0.0) -> float:
        """Sum over assignments of C_route (the comm-aware objective).
        Pass unreachable_penalty > 0 to charge stall/broken-route assignments
        honestly (benchmark scoring); 0.0 keeps the raw geometric walk."""
        total = 0.0
        for t in self.tasks:
            u = self.allocation.get(t.task_id)
            if u is None or u not in self.positions:
                continue
            total += route_cost_to(self.positions[u][:2], t.pos, links,
                                   self.positions, u, snap_etx,
                                   unreachable_penalty=unreachable_penalty)
        return total
