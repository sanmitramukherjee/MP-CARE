"""Phase 2 tests: baseline sanity checks (implementation_plan.md section 4.4).

Run from the repo root:  py -3.13 -m pytest tests/ -q
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/network_model", "src/mpcare_agent", "src/logger",
          "src/aodv_routing", "src/olsr_routing"):
    sys.path.insert(0, str(ROOT / p))

from network_model.model import NetworkParams, LinkModel, NetworkSnapshot  # noqa: E402
from mpcare_agent.agent import astar_path  # noqa: E402
from mpcare_agent.messaging import MessageBus  # noqa: E402
from olsr_routing.core import OlsrAgent  # noqa: E402
from aodv_routing.core import AodvAgent  # noqa: E402


# ------------------------------------------------------- B5: AT-PP path choice
def test_astar_etx_prefers_better_links_over_shorter_distance():
    """§4.4: AT-PP selects higher-throughput paths (longer but better links).

    Geometry: a 2-hop direct path (2 x 285 m links, ETX ~3.6 each) versus a
    3-hop detour (links ~190-206 m, ETX ~1 each). Distance-A* must pick the
    short path; ETX-A* must pick the detour.
    """
    pos = {0: (0.0, 0.0, 20.0),
           1: (285.0, 0.0, 20.0),      # direct-path relay
           4: (570.0, 0.0, 20.0),      # destination
           2: (190.0, 80.0, 20.0),     # detour nodes (longer total distance)
           3: (380.0, 80.0, 20.0)}
    p = NetworkParams()
    links = {i: set() for i in pos}
    for a in pos:
        for b in pos:
            if a < b and math.dist(pos[a][:2], pos[b][:2]) <= p.max_range_m:
                links[a].add(b)
                links[b].add(a)
    # sanity of the scenario: ETX costs favor the detour, distance does not
    snap = NetworkSnapshot(pos, LinkModel(p), 0.0)
    etx_direct = snap.etx_of(0, 1) + snap.etx_of(1, 4)
    etx_detour = snap.etx_of(0, 2) + snap.etx_of(2, 3) + snap.etx_of(3, 4)
    assert etx_detour < etx_direct
    assert math.dist(pos[0][:2], pos[4][:2]) < \
        math.dist(pos[0][:2], pos[2][:2]) + math.dist(pos[2][:2], pos[3][:2]) \
        + math.dist(pos[3][:2], pos[4][:2])

    short = astar_path(0, 4, links, pos)
    etx = astar_path(0, 4, links, pos,
                     edge_cost=lambda a, b: snap.etx_of(a, b),
                     heuristic=lambda n: math.dist(pos[n][:2], pos[4][:2]) / p.max_range_m)
    assert short == [0, 1, 4]          # distance-only: shortest geometric path
    assert etx == [0, 2, 3, 4]         # AT-PP style: longer but better links


# ------------------------------------------------------- B4: OLSR
def _olsr_net():
    return [OlsrAgent(i, 5) for i in range(5)]


def _feed_olsr(protos, t):
    """One full HELLO/TC exchange round over an ideal channel."""
    bus = MessageBus(LinkModel(NetworkParams(rng_seed=0)))
    positions = {i: (i * 160.0, 0.0, 20.0) for i in range(5)}
    links = {i: {j for j in (i - 1, i + 1) if 0 <= j < 5} for i in range(5)}
    for pr in protos:
        pr.tick(t, 1.0, links)
        pr.flush(bus, positions, links)
    bus.deliver({i: pr for i, pr in enumerate(protos)}, t)
    for pr in protos:
        pr.flush(bus, positions, links)
    bus.deliver({i: pr for i, pr in enumerate(protos)}, t)
    return positions, links


def test_olsr_mpr_covers_all_two_hop_neighbors():
    """§4.4: MPR set is correct — every 2-hop node is covered by the MPR set.

    Three exchange rounds: MPR selection uses LEARNED 2-hop knowledge, so it
    correctly lags neighbor discovery by one HELLO interval (round 1 learns
    neighbors, round 2 learns their neighbor lists, round 3 selects MPRs
    from complete information)."""
    protos = _olsr_net()
    positions, links = _feed_olsr(protos, 0.0)
    _feed_olsr(protos, 1.0)
    _feed_olsr(protos, 2.0)
    me = protos[2]                # middle node: neighbors {1,3}, 2-hop {0,4}
    n2 = set()
    for via in me.neigh:
        n2 |= {x for x in protos[via].neigh if x != me.node_id} - set(me.neigh)
    covered = set()
    for m in me.mpr_set:
        covered |= {x for x in protos[m].neigh if x != me.node_id} - set(me.neigh)
    assert covered >= n2          # MPR coverage property holds


def test_olsr_converges_and_routes_end_to_end():
    """§4.4: topology converges < 10 s; far-end node reachable via correct hop count."""
    protos = _olsr_net()
    t = 0.0
    converged = None
    while t < 20.0:
        positions, links = _feed_olsr(protos, t)
        if converged is None and protos[0].next_hop_for(4) is not None:
            converged = t
            break
        t += 1.0
    assert converged is not None and converged < 10.0
    nh = protos[0].next_hop_for(4)
    assert nh == 1                # chain: first hop toward node 4 is node 1


# ------------------------------------------------------- B3: AODV
def test_aodv_route_discovery_under_2s():
    """§4.4: route discovery completes in < 2 s on the 4-hop chain."""
    protos = [AodvAgent(i) for i in range(5)]
    bus = MessageBus(LinkModel(NetworkParams(rng_seed=0)))
    positions = {i: (i * 160.0, 0.0, 20.0) for i in range(5)}
    links = {i: {j for j in (i - 1, i + 1) if 0 <= j < 5} for i in range(5)}
    dt = 0.1
    t = 0.0
    discovered = None
    protos[0].next_hop_for(4, t)  # triggers RREQ
    while t < 5.0:
        t += dt
        for pr in protos:
            pr.set_context(bus, t, positions, links)
            pr.tick(t, dt, links)
            pr.flush(bus, positions, links)
        bus.deliver({i: pr for i, pr in enumerate(protos)}, t)
        for pr in protos:
            pr.flush(bus, positions, links)
        bus.deliver({i: pr for i, pr in enumerate(protos)}, t)
        if protos[0].routes.get(4) and protos[0].routes[4].is_valid:
            discovered = t
            break
    assert discovered is not None and discovered < 2.0
    assert protos[0].next_hop_for(4, t) == 1


# ------------------------------------------------------- overhead accounting
def test_message_bus_overhead_accounting():
    bus = MessageBus(LinkModel(NetworkParams(rng_seed=0)))
    positions = {0: (0.0, 0.0, 20.0), 1: (160.0, 0.0, 20.0)}
    links = {0: {1}, 1: {0}}
    bus.broadcast({"type": "HELLO", "size": 24}, 0, positions, links)
    bus.deliver({}, t=0.0)        # no handler registered: lost, but TX counted
    assert bus.control_msgs_sent[0] == 1
    assert bus.control_bytes_sent[0] == 24
    assert bus.control_overhead_per_node(2, 10.0) == pytest.approx(0.05)
