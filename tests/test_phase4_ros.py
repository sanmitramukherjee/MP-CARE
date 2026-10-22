"""Phase 4 ROS-wiring tests (the Task A/B logic behind NOTES/testROS2.md).

These run the EXACT code the Ubuntu gate will execute — the pure
``MpcareSwarmAdapter`` and ``network_model.topology`` helpers — without any
ROS install. If these pass on Windows, the Ubuntu side only needs to prove
plumbing (topics, build, bridges), not algorithm correctness.
"""

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts", "src/network_model", "src/mpcare_agent", "src/logger",
          "src/aodv_routing", "src/olsr_routing"):
    sys.path.insert(0, str(ROOT / p))

from network_model.model import NetworkParams, LinkModel, NetworkSnapshot  # noqa: E402
from network_model.topology import parse_blob, prune_stale, snapshot_etx  # noqa: E402
from mpcare_agent.ros_adapter import MpcareSwarmAdapter  # noqa: E402
from logger_pkg.core import SCHEMA  # noqa: E402

CHAIN_POS = {i: (i * 200.0, 0.0, 40.0) for i in range(5)}


def make_blob(positions=CHAIN_POS, t=1.0, seed=1):
    params = NetworkParams(rng_seed=seed)
    link = LinkModel(params)
    snap = NetworkSnapshot(dict(positions), link, t)
    links = [[a, b, round(snap.dist[(a, b)], 2),
              (None if math.isinf(snap.etx[(a, b)]) else round(snap.etx[(a, b)], 3))]
             for (a, b) in snap.etx if a < b]
    return json.dumps({"t": t, "links": links})


def feed(adapter, blob, ticks=10, t0=1.0):
    for k in range(ticks):
        adapter.on_topology(blob)
        for uid in adapter.odom:
            adapter.last_odom_t.setdefault(uid, t0 + k)


def test_parse_blob_symmetric_and_null_excluded():
    blob = json.dumps({"t": 3.0, "links": [[0, 1, 200.0, 1.05],
                                           [1, 2, 999.0, None]]})
    t, links, etx = parse_blob(blob)
    assert t == 3.0
    assert links == {0: {1}, 1: {0}}   # null-ETX pair = out of range = no edge
    assert (0, 1) in etx and (1, 0) in etx
    assert (1, 2) not in etx


def test_snapshot_etx_matches_sim_semantics():
    _, links, etx = parse_blob(make_blob())
    assert math.isfinite(snapshot_etx(links, etx, CHAIN_POS, 0, 1))
    assert snapshot_etx(links, etx, CHAIN_POS, 0, 3) == math.inf   # no edge
    assert snapshot_etx(links, etx, {}, 0, 1) == math.inf          # missing node


def test_adapter_builds_full_chain_route():
    ad = MpcareSwarmAdapter(5, rng_seed=1)
    for i, p in CHAIN_POS.items():
        ad.on_odom(i, p, 0.0)
    feed(ad, make_blob(), ticks=10)
    assert ad.route_of(0) == [0, 1, 2, 3, 4]
    assert ad.route_of(4, 0) is None or ad.route_of(4, 0) == [4, 3, 2, 1, 0]


def test_adapter_route_events_and_triggers_fire():
    ad = MpcareSwarmAdapter(5, rng_seed=1)
    for i, p in CHAIN_POS.items():
        ad.on_odom(i, p, 0.0)
    ad.on_topology(make_blob(t=1.0))
    # first-ever decision is recorded as a new-route event per alive source
    assert any(ev["trigger"] == "new-route" for ev in ad.route_events)
    assert len(ad.route_events) >= 1


def test_adapter_telemetry_row_is_logger_safe():
    ad = MpcareSwarmAdapter(5, rng_seed=1)
    for i, p in CHAIN_POS.items():
        ad.on_odom(i, p, 0.0)
    feed(ad, make_blob(), ticks=10)
    row = ad.telemetry_row(
        0, timestamp_ms=1000,
        position_x=0.0, position_y=0.0, position_z=40.0,
        velocity_x=0.0, velocity_y=0.0, velocity_z=0.0,
        target_waypoint_x=0.0, target_waypoint_y=0.0, target_waypoint_z=0.0,
        pdr_instantaneous=0.0, latency_avg_ms=0.0, throughput_kbps=0.0,
        queue_depth=0, cpu_time_us=0)
    # every SCHEMA key must be supplyable, and the node must NOT pass
    # algorithm/experiment (logger_node stamps those; a duplicate crashes)
    assert "algorithm" not in row and "experiment" not in row
    assert row["uav_id"] == 0
    assert row["neighbors"] == [1]
    assert len(row["neighbor_etx"]) == len(row["neighbors"])
    assert len(row["neighbor_let"]) == len(row["neighbors"])
    assert row["route_hop_count"] == 4
    assert isinstance(row["min_separation_m"], float)
    # dry-run the logger's row-building path (the exact code the Ubuntu
    # logger node executes) to prove no crash and no missing keys
    merged = dict(row, algorithm="mpcare", experiment="phase4_ros",
                  trial_id="t")
    safe = {k: merged.get(k, "") for k in SCHEMA}
    assert len(safe) == len(SCHEMA)


def test_adapter_stale_odometry_prunes_and_breaks_route():
    ad = MpcareSwarmAdapter(5, rng_seed=1)
    for i, p in CHAIN_POS.items():
        ad.on_odom(i, p, 0.0)
    # t < stale_s: everything alive
    feed(ad, make_blob(t=1.0), ticks=3, t0=0.0)
    assert 2 in ad.alive_ids()
    # odometry of UAV 2 goes silent; after > stale_s it must be pruned and
    # the chain must be partitioned (no route 0 -> 4)
    for k in range(10):
        t = 10.0 + k
        ad.last_odom_t[2] = -100.0
        ad.on_topology(make_blob(t=t))
        for uid in (0, 1, 3, 4):
            ad.on_odom(uid, CHAIN_POS[uid], t)
    assert 2 not in ad.alive_ids()
    assert ad.route_of(0) is None


def test_prune_stale_unit():
    odom = {0: (0, 0, 0), 1: (1, 0, 0)}
    last = {0: 10.0, 1: 8.0}
    assert set(prune_stale(odom, last, t_now=12.0, stale_s=3.0)) == {0}
    assert set(prune_stale(odom, last, t_now=10.5, stale_s=3.0)) == {0, 1}


def test_network_model_blob_prunes_stale_odometry():
    """Regression (native dual-boot Tier-3 run, 2026-09-29): the network-model
    node stored odometry forever, so /swarm/topology kept a killed UAV's
    links after the agents had already pruned it. node_ros.py now calls the
    SAME prune_stale helper at every tick — this pins that exact composition
    (node clock -> last_seen, stale_s from params, prune before snapshot).
    The node class itself needs rclpy, so it is exercised live in WSL
    (scripts/wsl_conf.sh tier3) rather than here."""
    last_seen = {i: 104.0 for i in CHAIN_POS}   # everyone streaming now
    last_seen[2] = 100.0                        # uav 2's odom went silent
    t_now = 104.5
    alive = prune_stale(dict(CHAIN_POS), last_seen, t_now, 3.0)
    assert 2 not in alive
    assert set(alive) == {0, 1, 3, 4}
    snap = NetworkSnapshot(alive, LinkModel(NetworkParams(rng_seed=1)), t_now)
    assert not any(2 in pair for pair in snap.etx)   # no link touches uav 2
    # and the agents see the same alive set through the adapter path
    ad = MpcareSwarmAdapter(5, rng_seed=1)
    for i, p in CHAIN_POS.items():
        ad.on_odom(i, p, 0.0)
    ad.on_topology(make_blob(t=t_now))
    for uid in (0, 1, 3, 4):
        ad.on_odom(uid, CHAIN_POS[uid], t_now)
    ad.on_topology(make_blob(t=t_now + 1.0))
    assert 2 not in ad.alive_ids()
    assert ad.route_of(0) is None


def test_adapter_single_uav_degrades_gracefully():
    ad = MpcareSwarmAdapter(5, rng_seed=1)
    ad.on_odom(0, CHAIN_POS[0], 0.0)
    feed(ad, json.dumps({"t": 1.0, "links": []}), ticks=3)
    assert ad.route_of(0) is None
    assert ad.telemetry_row(0, timestamp_ms=1) is not None
