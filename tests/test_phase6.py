"""Phase 6 tests (plan section 8): the scale-point parameterization (8.2),
the section 8.3 metric instrumentation in run_sim.run, and the run_phase6
alarm logic (verdicts from measured numbers only).

Every expected number is derivable from the definitions (no-hallucination
rule); run-specific assertions are pinned to deterministic outputs and
commented with their derivation.
"""

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts", "src/network_model", "src/mpcare_agent", "src/logger",
          "src/aodv_routing", "src/olsr_routing"):
    sys.path.insert(0, str(ROOT / p))

import run_sim  # noqa: E402
import run_phase6  # noqa: E402

OUT = ROOT / "data" / "raw"

# implementation_plan.md section 8.2, exact table
PLAN_SCALE_POINTS = {5: (500.0, 500.0, 10), 10: (700.0, 700.0, 20),
                     20: (1000.0, 1000.0, 40), 30: (1200.0, 1200.0, 60),
                     40: (1500.0, 1500.0, 80), 50: (1700.0, 1700.0, 100)}


# ------------------------------------------- section 8.2 scale parameterization

def test_scale_points_match_plan_table():
    assert run_sim.SCALE_POINTS == PLAN_SCALE_POINTS


def test_scale_mission_legs_are_bounded_and_in_box():
    # bounded-leg walk: every leg <= 0.5 * (box - 2*inset), targets inside the
    # inset box (mirror keeps them in-box and only shortens a leg)
    for seed in (0, 1, 7, 123):
        for n in (5, 50):
            w, h, n_wps = PLAN_SCALE_POINTS[n]
            m0 = random.Random(seed)
            starts = [(m0.uniform(40, w - 40), m0.uniform(40, h - 40))
                      for _ in range(n)]
            span = min(w, h) - 2 * run_sim.SCALE_INSET
            for i in range(n):
                wps = run_sim._scale_waypoints(seed, i, n_wps, (w, h), starts[i])
                assert len(wps) == n_wps
                prev = starts[i]
                for wp in wps:
                    assert run_sim.SCALE_INSET <= wp[0] <= w - run_sim.SCALE_INSET
                    assert run_sim.SCALE_INSET <= wp[1] <= h - run_sim.SCALE_INSET
                    assert math.dist(prev, wp[:2]) <= 0.5 * span + 1e-9
                    prev = wp[:2]


def test_scale_scenario_builds_mission_and_box():
    params, link, agents, start, goal = run_sim.build_swarm(11, "scale", None, 30)
    w, h, n_wps = PLAN_SCALE_POINTS[30]
    assert len(agents) == 30
    assert len(agents[0].mission.waypoints) == n_wps
    for a in agents:
        assert 40 <= a.pos[0] <= w - 40 and 40 <= a.pos[1] <= h - 40
    assert agents[0].speed == run_sim.SCALE_SPEED
    # the release goal is the first mission leg
    assert tuple(goal[0]) == tuple(agents[0].mission.waypoints[0])


def test_scale_rejects_non_plan_n():
    try:
        run_sim.build_swarm(1, "scale", None, 7)
        assert False, "expected ValueError for N not in the plan table"
    except ValueError:
        pass


# ------------------------------------- section 8.3 metric instrumentation smoke

def test_scale_run_mpcare_smoke_n10():
    # worst-case first leg at N=10: 0.5 * (700 - 2*40) m / 8 m/s = 38.75 s
    # after the t=30 release -> duration 70 s leaves headroom
    r = run_sim.run(11, OUT, duration=70.0, write_csv=False,
                    algorithm="mpcare", scenario="scale", n_uav=10)
    assert r["packets_sent"] == 50                       # 25 s window / 0.5 s
    assert r["packets_delivered"] == r["packets_sent"]   # queue-until-delivered
    assert r["pdr"] == r["chain_pdr"] == 1.0             # nominal, connected seed
    assert r["mission_time_s"] is not None and 0.0 < r["mission_time_s"] <= 38.8
    assert r["stability_ok"] and r["waypoint_ok"] and r["network_ok"]
    assert r["beacons_per_node_s"] == 10.0               # 1/0.1 s beacon design
    assert r["data_tx_per_node_s"] < r["beacons_per_node_s"]
    assert r["convergence_def"] == run_sim.CONVERGENCE_DEFS["mpcare"]
    assert r["full_conv_s"] is not None and 0.0 <= r["full_conv_s"] <= 2.0
    assert r["cpu_ms_per_node_tick"] > 0.0
    # distribution calibration: probe measured ~0.016 ms/node/tick at N=50;
    # at N=10 with loop overhead it is well under 0.5 ms (alarm = 10 ms)
    assert r["cpu_ms_per_node_tick"] < 0.5


def test_scale_run_aodv_olsr_smoke_n10():
    ra = run_sim.run(11, OUT, duration=70.0, write_csv=False,
                     algorithm="aodv", scenario="scale", n_uav=10)
    ro = run_sim.run(11, OUT, duration=70.0, write_csv=False,
                     algorithm="olsr", scenario="scale", n_uav=10)
    assert ra["convergence_def"] == run_sim.CONVERGENCE_DEFS["aodv"]
    # full_conv_s is the STATIC-WINDOW (t < 30 s) variant of the discovery time
    if ra["first_route_s"] is not None and ra["first_route_s"] < run_sim.HOVER_UNTIL:
        assert ra["full_conv_s"] == round(ra["first_route_s"], 2)
    else:
        assert ra["full_conv_s"] is None
    assert ro["convergence_def"] == run_sim.CONVERGENCE_DEFS["olsr"]
    # route-to-destination can exist at t=0 (converged_s == 0.0) while the
    # FULL component view needs one TC period (TC_INTERVAL = 5 s): different
    # metrics by definition, so assert each against its own bound
    assert ro["full_conv_s"] is not None and 0.0 < ro["full_conv_s"] <= 5.0
    assert ra["control_pkts_per_node_s"] > 0.0
    assert ro["control_pkts_per_node_s"] > 0.0
    assert ra["avg_degree"] == ro["avg_degree"]          # same physics, same seed


def test_scale_determinism_mpcare_n5():
    # perf_counter-derived metrics (cpu_ms_per_node_tick, run walls) are
    # deliberately EXCLUDED: timing is not expected to be bit-reproducible
    a = run_sim.run(5, OUT, duration=40.0, write_csv=False,
                    algorithm="mpcare", scenario="scale", n_uav=5)
    b = run_sim.run(5, OUT, duration=40.0, write_csv=False,
                    algorithm="mpcare", scenario="scale", n_uav=5)
    for k in ("final_positions", "packets_delivered", "pdr", "chain_pdr",
              "avg_degree", "full_conv_s", "beacons_per_node_s",
              "data_tx_per_node_s"):
        assert a[k] == b[k], k


# ------------------------------------------------------- run_phase6 bookkeeping

def _mkrow(algo: str, n: int, **kw) -> dict:
    base = dict(seed=1, trial=0, algorithm=algo, N=n, condition=f"N{n}",
                chain_pdr=1.0, pdr=1.0, packets_sent=50, packets_delivered=50,
                avg_latency_ms=1.0, mission_time_s=10.0,
                cpu_ms_per_node_tick=0.02, control_pkts_per_node_s=10.0,
                beacons_per_node_s=10.0, data_tx_per_node_s=0.2,
                full_conv_s=0.0, convergence_def="x", avg_degree=4.5,
                reroutes_total=0, reroutes_per_uav_min=0.0, trigger_events="",
                run_wall_s=1.0, sanity_ok=True)
    base.update(kw)
    return base


def test_flatten_normalizes_reroutes_per_uav_minute():
    r = {"seed": 3000, "algorithm": "mpcare",
         "trigger_events": {"T1": 120, "T4": 60},
         "pdr": 1.0, "chain_pdr": 1.0, "packets_sent": 50,
         "packets_delivered": 50, "stability_ok": True, "waypoint_ok": True,
         "network_ok": True, "_duration": 150.0}
    row = run_phase6._flatten(r, 30, 0, 12.5)
    assert row["reroutes_total"] == 180
    # 180 events / 30 UAVs / (150 s / 60 s) = 2.4 per UAV per minute
    assert row["reroutes_per_uav_min"] == 2.4
    assert row["sanity_ok"] is True and row["run_wall_s"] == 12.5
    assert row["condition"] == "N30"


def test_alarm_verdicts_pass_when_within_thresholds():
    algos = ("mpcare", "gpsr", "olsr", "aodv")
    rows = [_mkrow(a, 50) for a in algos]
    rep = run_phase6.evaluate_alarms(rows, list(algos))
    assert rep["evaluated_at_N"] == 50
    assert rep["all_pass"] is True
    assert len(rep["checks"]) == 5 * len(algos)
    assert all(c["verdict"] == "PASS" for c in rep["checks"])


def test_alarm_fires_on_measured_breach_only():
    algos = ("mpcare", "gpsr", "olsr", "aodv")
    rows = [_mkrow("mpcare", 50, cpu_ms_per_node_tick=10.5)]
    rows += [_mkrow(a, 50) for a in algos[1:]]
    rep = run_phase6.evaluate_alarms(rows, list(algos))
    bad = [c for c in rep["checks"] if c["verdict"] == "ALARM"]
    assert len(bad) == 1
    assert bad[0]["algorithm"] == "mpcare"
    assert bad[0]["metric"] == "cpu_ms_per_node_tick"
    assert rep["all_pass"] is False


def test_alarm_thresholds_are_inclusive():
    rows = [_mkrow("mpcare", 50, cpu_ms_per_node_tick=10.0,
                   control_pkts_per_node_s=50.0, full_conv_s=30.0, pdr=0.70,
                   reroutes_per_uav_min=10.0)]
    rep = run_phase6.evaluate_alarms(rows, ["mpcare"])
    assert all(c["verdict"] == "PASS" for c in rep["checks"])
    assert rep["all_pass"] is True


def test_alarm_pdr_breach_and_na_metric():
    rows = [_mkrow("gpsr", 50, pdr=0.65, full_conv_s=None)]
    rep = run_phase6.evaluate_alarms(rows, ["gpsr"])
    alarm = [c for c in rep["checks"] if c["verdict"] == "ALARM"]
    na = [c for c in rep["checks"] if c["verdict"] == "n/a"]
    assert len(alarm) == 1 and alarm[0]["metric"] == "pdr"
    assert len(na) == 1 and na[0]["metric"] == "full_conv_s"
    assert rep["all_pass"] is False


# --------------------------------------- phase 6 remediation: void face walk

def _void_router_with_hood(me_xy, dst_xy, hood, dst_id=99,
                           prev_id=None, prev_xy=None):
    """A router at (0, 0) bound to hooks that expose ONLY the given 1-hop
    neighborhood (the beacon state a real node would have); the destination
    `dst_id` exists in the position set (geocast guard) but is NOT a
    neighbor; optional `prev_id`/`prev_xy` place the arrival-edge predecessor
    (so the walk's right-hand rule uses the arrival heading)."""
    from mpcare_agent.router import MpcareRouter, SimHooks
    from network_model.model import LinkModel, NetworkParams, NetworkSnapshot
    r = MpcareRouter(0, 1 + len(hood) + 1, max_range_m=300.0)
    pos = {0: (me_xy[0], me_xy[1], 40.0), dst_id: (dst_xy[0], dst_xy[1], 40.0)}
    t = 1.0
    if prev_id is not None:
        pos[prev_id] = (prev_xy[0], prev_xy[1], 40.0)
    for nid, (x, y) in hood.items():
        pos[nid] = (x, y, 40.0)
    link = LinkModel(NetworkParams(rng_seed=1))
    snap = NetworkSnapshot(pos, link, t)
    hooks = SimHooks(pos, link, snap)
    r.bind(hooks)
    for nid, (x, y) in hood.items():
        r.receive_beacon({"type": "MPBEACON", "src": nid, "pos": (x, y, 40.0),
                          "queue": 0.0, "nbrs": (0,)}, nid,
                         snap.etx_of(0, nid), t)
    r.table.set_my_position((me_xy[0], me_xy[1], 40.0))
    return r, hooks


def test_void_walk_escapes_local_minimum():
    # classic concave void: dst beyond a keep-out region; both neighbors are
    # FARTHER from dst than we are -> greedy has no candidate. The walk must
    # still return a neighbor (clockwise-first), not None.
    r, _ = _void_router_with_hood((0.0, 0.0), (300.0, 0.0),
                                  {1: (-100.0, 200.0), 2: (-100.0, -200.0)})
    nh = r.next_hop_for(99, 5.0)   # greedy cands empty: both nbrs farther
                                   # (dist 447 > 300) -> true local minimum
    assert nh is not None
    assert nh in (1, 2)
    assert r.recovery_events == 1


def test_void_walk_right_hand_rule_prefers_first_clockwise():
    # arrival from the west (prev node at (-100, 0)); face neighbors: 1 north,
    # 2 south. Reference = arrival heading (pointing east), so the first
    # neighbor clockwise (smallest positive angle from east) is NORTH (1).
    r, _ = _void_router_with_hood((0.0, 0.0), (500.0, 0.0),
                                  {1: (0.0, 200.0), 2: (0.0, -200.0)}, dst_id=99,
                                  prev_id=1, prev_xy=(-100.0, 0.0))
    nh = r.next_hop_for(99, 5.0, visited=frozenset({1, 99}), prev=1)
    assert nh == 2, "south (2) must be first clockwise when 1 is the arrival edge"
    assert r.recovery_events == 1


def test_void_walk_never_back_steps_while_another_edge_exists():
    # same geometry as the clockwise test; the walk must NOT return to the
    # arrival edge (node 1) while node 2 exists — and without prev in
    # positions the reference falls back to the destination direction.
    r, _ = _void_router_with_hood((0.0, 0.0), (500.0, 0.0),
                                  {1: (-100.0, 0.0), 2: (0.0, 200.0)}, dst_id=99)
    r.next_hop[99] = 1
    nh = r.next_hop_for(99, 5.0, visited=frozenset({99}), prev=1)
    assert nh == 2
    assert 99 not in r.next_hop          # stale greedy route cleared at entry


def test_greedy_unchanged_when_candidates_exist():
    # a strictly-closer neighbor must be chosen greedily with NO recovery
    r, _ = _void_router_with_hood((0.0, 0.0), (400.0, 0.0),
                                  {1: (150.0, 0.0), 2: (50.0, 250.0)})
    assert r.next_hop_for(99, 5.0) == 1
    assert r.recovery_events == 0


def test_scale_recovery_smoke_known_bad_seed():
    # phase6v1: seed 3002 @ N=30 scored 0.0 (probe: 2400/2400 void stalls);
    # with the face walk the SAME seed must clear the void and deliver
    r = run_sim.run(3002, OUT, duration=150.0, write_csv=False,
                    algorithm="mpcare", scenario="scale", n_uav=30)
    assert r["pdr"] >= 0.7, r["pdr"]
    assert r["void_recovery_events"] > 0
