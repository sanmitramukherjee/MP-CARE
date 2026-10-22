"""Tests for Phase 5 Extension B: CBBA + comm-cost bid (plan §7.2).

Pure-Python module tests (Windows, no ROS). Exit criteria under test
(plan §7.3): CBBA+comm allocates tasks with completion metrics; the comm
term steers bids by route cost; consensus is exact at 0% loss (sanity) and
loss-sensitive (the honesty-critical mechanism). The full loss-sweep
benchmark lives in scripts/run_phase5.py.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/cbba_allocator", "src/network_model"):
    sys.path.insert(0, str(ROOT / p))

from cbba_allocator.core import (AllocationResult, CbbaAgent, Task,
                                 route_cost_to, run_consensus)  # noqa: E402


def _chain(n=5, spacing=200.0):
    pos = {i: (i * spacing, 0.0, 40.0) for i in range(n)}
    links = {i: ({i - 1} if i > 0 else set()) | ({i + 1} if i < n - 1 else set())
             for i in range(n)}
    return pos, links


def test_route_cost_greedy_walks_the_chain():
    pos, links = _chain()
    c = route_cost_to(pos[0], (800.0, 0.0), links, pos, 0)
    # 4 hops x 200 m x etx 1.0 (no snap_etx) = 800
    assert math.isclose(c, 800.0, rel_tol=1e-9)


def test_route_cost_no_neighbors_falls_back_to_distance():
    pos, links = _chain(3)
    links[1] = set()                       # isolated middle node
    c = route_cost_to(pos[1], (200.0, 100.0), links, pos, 1)
    assert math.isclose(c, 100.0, rel_tol=1e-9)


def test_comm_bid_flips_the_winner_on_expensive_routes():
    """Both agents bid on the same task; only the comm-aware bid must prefer
    the cheap-route agent. Geometry (w_t=0 isolates the comm term):
    agent 0 at 100 m straight shot to the task (route cost 100).
    agent 1 at 450 m from the task with its 300 m link gone (out of range):
    no greedy path, stall -> route cost = 450 straight-line stall penalty.
    Plain bid: 1 wins (450 < 550). Comm bid: 0 wins."""
    pos = {0: (100.0, 0.0, 40.0), 1: (900.0, 0.0, 40.0)}
    task = Task(0, (200.0, 0.0))
    links = {0: {1}, 1: {0}}               # 800 m apart: link 0-1 is OUT of range
    etx = lambda a, b: 1.0                 # noqa: E731
    c0 = CbbaAgent(0, w_d=1.0, w_t=0.0, w_c=1.0)
    c1 = CbbaAgent(1, w_d=1.0, w_t=0.0, w_c=1.0)
    c0.set_view(pos, links, etx)
    c1.set_view(pos, links, etx)
    assert c0.bid(task) > c1.bid(task)     # comm-aware: 0 wins
    # (baseline distance-only semantics are pinned separately below — here
    # plain bids would ALSO prefer 0, since 0 is simply closer; the point of
    # this test is that the comm term PENALIZES 1's broken 800 m route.)


def test_plain_bid_baseline_is_distance_only():
    """Clean discriminating geometry for the BASELINE: both agents at the
    same distance from the task (200 m), so plain bids tie — comm cost is
    the only difference, and the expensive-route agent must score lower."""
    pos = {0: (0.0, 0.0, 40.0), 1: (400.0, 0.0, 40.0)}
    task = Task(0, (200.0, 0.0))
    links = {0: {1}, 1: {0}}               # 400 m: NOT in range (max 300)
    etx = lambda a, b: 1.0                 # noqa: E731
    p0 = CbbaAgent(0, w_d=1.0, w_t=0.0, w_c=0.0)
    p1 = CbbaAgent(1, w_d=1.0, w_t=0.0, w_c=0.0)
    p0.set_view(pos, links)
    p1.set_view(pos, links)
    assert math.isclose(p0.bid(task), p1.bid(task), rel_tol=1e-9)
    c0 = CbbaAgent(0, w_d=1.0, w_t=0.0, w_c=1.0)
    c1 = CbbaAgent(1, w_d=1.0, w_t=0.0, w_c=1.0)
    c0.set_view(pos, links, etx)
    c1.set_view(pos, links, etx)
    # 0 has no path to the task (its only neighbor is 1, farther from it):
    # stall cost 200; 1 likewise stalls at 200 — equal. Rewire so 1 has a
    # cheap path THROUGH extra nodes; 0 does not.
    pos2 = {0: (0.0, 0.0, 40.0), 1: (400.0, 0.0, 40.0),
            2: (190.0, 0.0, 40.0), 3: (210.0, 0.0, 40.0)}
    links2 = {0: {1}, 1: {0, 2, 3}, 2: {1, 3}, 3: {1, 2}}
    etx2 = lambda a, b: 1.0                # noqa: E731
    c0 = CbbaAgent(0, w_d=1.0, w_t=0.0, w_c=1.0)
    c1 = CbbaAgent(1, w_d=1.0, w_t=0.0, w_c=1.0)
    c0.set_view(pos2, links2, etx2)
    c1.set_view(pos2, links2, etx2)
    # agent 1: greedy path 1->2->task? 2 at 190: d(2,task)=10 < 400 => hop,
    # then stall 10. cost ~ 210*1 + 10 = 220. agent 0: neighbor 1 at 400,
    # d(1,task)=200 > 200-eps => no strict improvement => stall 200.
    b0, b1 = c0.bid(task), c1.bid(task)
    assert b1 < b0                         # 1's route is costlier => lower bid


def test_consensus_converges_at_zero_loss():
    pos, links = _chain(5)
    tasks = [Task(0, (0.0, 0.0)), Task(1, (800.0, 0.0))]
    agents = [CbbaAgent(i, w_d=1.0, w_t=0.0, w_c=0.0) for i in range(5)]
    for a in agents:
        a.set_view(pos, links)
        a.decide(tasks, n_bundle=1)
    belief = run_consensus(agents, links, link_success=None, epochs=12)
    alloc = {tid: bidder for tid, (s, bidder) in belief[0].items()}
    # nearest agent wins each task: task0 -> uav0, task1 -> uav4
    assert alloc[0] == 0 and alloc[1] == 4
    res = AllocationResult(alloc, tasks, pos)
    assert res.completion_rate() == 1.0


def test_consensus_loss_blocks_propagation():
    pos, links = _chain(5)
    tasks = [Task(0, (0.0, 0.0)), Task(1, (800.0, 0.0))]
    agents = [CbbaAgent(i, w_d=1.0, w_t=0.0, w_c=0.0) for i in range(5)]
    for a in agents:
        a.set_view(pos, links)
        a.decide(tasks, n_bundle=1)
    belief = run_consensus(agents, links,
                           link_success=lambda d, a=None, b=None: False,
                           epochs=12)
    # with EVERY edge dropping, each agent only ever hears itself: beliefs
    # stay local — agent 0 can at most know its own bids
    assert set(belief[0]) <= {t.task_id for t in tasks}
    assert all(bidder == 0 for (_, bidder) in belief[0].values())


def test_consensus_channel_receives_pair_ids():
    """link_success gets (d, a, b) — per-PAIR impairments (Phase 4's
    set_link_loss semantics) need the node ids, not just the distance."""
    pos, links = _chain(3)
    tasks = [Task(0, (400.0, 0.0))]
    agents = [CbbaAgent(i, w_d=1.0, w_t=0.0, w_c=0.0) for i in range(3)]
    for a in agents:
        a.set_view(pos, links)
        a.decide(tasks, n_bundle=1)
    seen = []

    def chan(d, a=None, b=None):
        seen.append((round(d), a, b))
        return True

    run_consensus(agents, links, link_success=chan, epochs=1)
    pairs = {(a, b) for (_, a, b) in seen}
    assert (0, 1) in pairs and (1, 2) in pairs
    assert all(a is not None and b is not None for (_, a, b) in seen)


def test_badlink_reroute_flips_winner():
    """The §7.2 win condition as a unit pin (asymmetric corridor chain):
    host 3 at 450 m issues a task at its station; its LEFT chain edge (2,3)
    is degraded (ETX x11.1 in the comm bid). The euclid-nearest bidder is 2
    (150 m) — plain CBBA hands the task to the degraded-side agent; the
    comm-aware bid routes the allocation to the CLEAN-side agent 4 (170 m
    direct). Same channel, same impairment — only the bid changes."""
    xs = [0.0, 150.0, 300.0, 450.0, 620.0, 770.0, 920.0]
    pos = {i: (xs[i], 0.0, 40.0) for i in range(7)}
    links = {i: ({i - 1} if i else set()) | ({i + 1} if i < 6 else set())
             for i in range(7)}
    host, task = 3, Task(0, (450.0, 0.0))
    IMPAIR = {frozenset((2, 3))}
    etx = lambda a, b: 11.1111111 \
        if frozenset((a, b)) in IMPAIR else 1.0     # noqa: E731
    bidders = [task]   # host 3 issues the task; it does not bid (excluded
                       # from the agent list below)

    def ref_winner(w_c):
        agents = [CbbaAgent(i, w_d=1.0, w_t=0.5, w_c=w_c) for i in pos]
        for a in agents:
            mine = [task] if a.uav_id != host else []   # host never bids
            a.set_view(pos, links, etx if w_c else None)
            a.decide(mine, n_bundle=1)
        belief = run_consensus(agents, links, link_success=None, epochs=50)
        return belief[host][task.task_id][1]   # host's table = the reader

    assert ref_winner(0.0) == 2          # plain CBBA: euclid-nearest wins
    assert ref_winner(1.0) == 4          # comm-aware: clean-side agent wins


def test_allocation_result_completion_counts_nearest_winner():
    pos = {0: (0.0, 0.0, 40.0), 1: (1000.0, 0.0, 40.0)}
    tasks = [Task(0, (10.0, 0.0)), Task(1, (990.0, 0.0))]
    good = AllocationResult({0: 0, 1: 1}, tasks, pos)
    bad = AllocationResult({0: 1, 1: 0}, tasks, pos)
    assert good.completion_rate() == 1.0
    assert bad.completion_rate() == 0.0


def test_bids_are_deterministic():
    pos, links = _chain(3)
    tasks = [Task(0, (150.0, 0.0))]
    a = CbbaAgent(0, w_c=0.5)
    a.set_view(pos, links)
    b1 = a.decide(tasks, n_bundle=1)
    b2 = a.decide(tasks, n_bundle=1)
    assert b1 == b2
