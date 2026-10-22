"""Unit tests for the Phase 1 cores (network model, A*, GPSR, logger).

Run from the repo root:  py -3.13 -m pytest tests/ -q
"""

import csv
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))
sys.path.insert(0, str(ROOT / "src" / "logger"))

from network_model.model import (  # noqa: E402
    NetworkParams, LinkModel, NetworkSnapshot, path_loss_db, snr_db, prr, etx,
    DropTailQueue)
from mpcare_agent.agent import (  # noqa: E402
    astar_path, GpsrEngine, Packet, UavAgent, WaypointMission)
from logger_pkg.core import TrialLogger, verify_schema, SCHEMA  # noqa: E402


# ---------------------------------------------------------------- network model

def test_path_loss_matches_plan_defaults():
    p = NetworkParams()
    assert path_loss_db(1.0, p) == 46.4
    assert path_loss_db(100.0, p) == pytest.approx(46.4 + 50.0)   # 10 * 2.5 * log10(100)


def test_snr_and_prr_sigmoid():
    p = NetworkParams()
    # at max range SNR = 20 - 46.4 - 25*log10(300) + 90 = ~1.66 dB -> PRR ~ 0.5
    assert prr(300.0, p) == pytest.approx(0.5, abs=0.05)
    assert prr(100.0, p) > 0.99
    assert prr(301.0, p) == 0.0


def test_etx_semantics():
    p = NetworkParams()
    assert etx(100.0, p) == pytest.approx(1.0, abs=1e-3)   # near-perfect link
    assert math.isinf(etx(400.0, p))                       # out of range -> infinity
    assert etx(250.0, p) > 1.0                             # degraded link costs more


def test_unicast_retries_help():
    p = NetworkParams(rng_seed=0, mac_retries=4)
    p_mac0 = NetworkParams(rng_seed=0, mac_retries=0)
    lm = LinkModel(p)
    lm0 = LinkModel(p_mac0)
    # count successes over a fixed marginal link (PRR ~ 0.5)
    n1 = sum(lm.unicast_success(290.0) for _ in range(200))
    n0 = sum(lm0.unicast_success(290.0) for _ in range(200))
    assert n1 > n0


def test_snapshot_neighbors_and_range():
    p = NetworkParams()
    pos = {0: (0.0, 0.0, 20.0), 1: (160.0, 0.0, 20.0), 2: (640.0, 0.0, 20.0)}
    snap = NetworkSnapshot(pos, LinkModel(p), 0.0)
    assert snap.neighbors(0) == [1]
    assert snap.etx_of(0, 2) == math.inf


def test_queue_tail_drop():
    q = DropTailQueue(3)
    assert all(q.push(i) for i in range(3))
    assert not q.push(99)
    assert q.dropped == 1
    assert q.depth == 3


# ---------------------------------------------------------------- A* / GPSR

def _chain_positions(n=5, spacing=160.0):
    return {i: (i * spacing, 0.0, 20.0) for i in range(n)}


def test_astar_finds_chain_path():
    pos = _chain_positions()
    links = {i: {j for j in (i - 1, i + 1) if 0 <= j < 5} for i in range(5)}
    path = astar_path(0, 4, links, pos)
    assert path == [0, 1, 2, 3, 4]


def test_astar_no_path_when_disconnected():
    pos = _chain_positions()
    links = {0: {1}, 1: {0}, 2: {3}, 3: {2}, 4: set()}
    assert astar_path(0, 4, links, pos) is None


def _linear_topology(n=5, spacing=160.0):
    pos = _chain_positions(n, spacing)
    links = {i: {j for j in (i - 1, i + 1) if 0 <= j < n} for i in range(n)}
    return pos, links


def test_gpsr_greedy_advances_toward_dst():
    pos, links = _linear_topology()
    eng = GpsrEngine(1)
    pkt = Packet(pid=0, src=0, dst=4, dst_pos=(pos[4][0], pos[4][1]), created_t=0.0)
    assert eng.next_hop(pkt, pos, links) == 2


def test_gpsr_delivers_across_chain():
    pos, links = _linear_topology()
    pkt = Packet(pid=0, src=0, dst=4, dst_pos=(pos[4][0], pos[4][1]), created_t=0.0, path=[0])
    cur, guard = 0, 0
    while cur != 4 and guard < 20:
        eng = GpsrEngine(cur)
        nxt = eng.next_hop(pkt, pos, links)
        assert nxt is not None, f"dropped at {cur} mode={pkt.mode}"
        pkt.path.append(nxt)
        pkt.hops += 1   # the driver increments hops on each forward
        cur = nxt
        guard += 1
    assert cur == 4
    assert pkt.hops == 4 and pkt.path == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- agent/logger

def test_agent_reaches_waypoint():
    a = UavAgent(0, (0.0, 0.0, 20.0), WaypointMission([(100.0, 0.0, 20.0)]), speed_mps=10.0)
    steps = 0
    while a.mission.current_wp() is not None and steps < 200:
        a.move_toward(a.mission.current_wp(), 0.1)
        steps += 1
    assert math.dist(a.pos, (100.0, 0.0, 20.0)) == 0.0


def test_logger_schema_exact(tmp_path):
    with TrialLogger(tmp_path, trial_id="t1") as tl:
        tl.log(timestamp_ms=0, uav_id=0, position_x=1.0, algorithm="astar_dist+gpsr",
               experiment="phase1_mvp", neighbors=[1, 2], neighbor_etx=[1.0, math.inf],
               neighbor_let=[-1.0, -1.0])
    v = verify_schema(tl.path, min_rows=1)
    assert v["schema_ok"] and v["rows"] == 1 and v["rows_ok"]
    with open(tl.path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == SCHEMA
    assert rows[1][SCHEMA.index("neighbor_etx")] == "1.000;inf"


def test_run_sim_end_to_end(tmp_path):
    """The full section 3.5 benchmark must pass inside pytest too."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sim
    r = run_sim.run(42, tmp_path, duration=90.0, algorithm="gpsr")
    assert r["stability_ok"] and r["waypoint_ok"] and r["routing_ok"]
    assert r["network_ok"] and r["logger_ok"]
    assert r["chain_pdr"] > 0.9
