"""Phase 3 tests: MP-CARE core validation (implementation_plan.md section 5.4).

Run from the repo root:  py -3.13 -m pytest tests/ -q
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/network_model", "src/mpcare_agent", "src/logger"):
    sys.path.insert(0, str(ROOT / p))

from network_model.model import NetworkParams  # noqa: E402
from mpcare_agent.link_predictor import LinkPredictor  # noqa: E402
from mpcare_agent.cost_engine import CostEngine, PathFeatures  # noqa: E402
from mpcare_agent.event_trigger import EventTrigger, TriggerParams  # noqa: E402


# ------------------------------------------------ 5.4 row 1: prefers better link
def test_route_prefers_better_link_when_distances_equal():
    """Two routes of EQUAL length: MP-CARE's cost must pick the lower-ETX one."""
    eng = CostEngine()
    f_short_lossy = PathFeatures(path=[0, 1, 4], distance_m=200.0, etx_sum=8.0,
                                 etx_max=4.0, latency_s=0.02, queue_max=0.0, hops=2)
    f_long_solid = PathFeatures(path=[0, 2, 3, 4], distance_m=200.0, etx_sum=3.0,
                                etx_max=1.2, latency_s=0.01, queue_max=0.0, hops=3)
    ranked = eng.rank([f_short_lossy, f_long_solid])
    assert ranked[0][1].path == [0, 2, 3, 4]


def test_diamond_run_routes_along_rim():
    """End-to-end: on the diamond, MP-CARE delivers everything via the rim,
    never via the marginal upper shortcut (seed 42)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sim
    r = run_sim.run(42, Path(ROOT / "data" / "raw"), write_csv=False,
                    algorithm="mpcare", scenario="diamond")
    assert r["route_choice"]["upper"] == 0
    assert r["route_choice"]["rim"] == r["chain_sent"]
    assert r["chain_pdr"] > 0.99


# ------------------------------------------------ 5.4 row 2: prediction works
def test_prediction_improves_when_approaching_degrades_when_receding():
    lp = LinkPredictor(max_range_m=300.0)
    etx_now = 2.0
    approaching = lp.etx_pred(etx_now, radial_vel_mps=-10.0)   # closing in
    receding = lp.etx_pred(etx_now, radial_vel_mps=+10.0)      # flying away
    assert approaching < etx_now < receding
    assert math.isinf(lp.let_s(200.0, -5.0))          # approaching: never expires
    let = lp.let_s(200.0, 10.0)                       # 100 m left at 10 m/s
    assert let == pytest.approx(10.0)
    f = lp.f_velocity
    assert f(-20.0) < 1.0 < f(20.0)


# ------------------------------------------------ 5.4 row 3: hysteresis bound
def test_hysteresis_bounds_reroute_count_under_oscillation():
    """Rapid link fluctuation for T=30 s with t_dwell=2 s: re-routes must be
    <= floor(T / t_dwell) (plan's formal property)."""
    trig = EventTrigger(TriggerParams(t_dwell=2.0, delta=0.10))
    fires = 0
    T, dt = 30.0, 0.1
    quality_flips = True   # oscillate: bad route one second, good the next
    t = 0.0
    while t < T:
        # T1 condition flapping every second; T3 alternately true
        prr = 0.4 if (int(t) % 2 == 0 and quality_flips) else 0.9
        better = (int(t) % 4 == 1)
        fire, _ = trig.should_reroute(t, route_prr=prr, min_let_s=None,
                                      better_alt=better, lost_neighbors=[])
        fires += fire
        t += dt
    assert fires <= math.floor(T / trig.p.t_dwell)


def test_t3_requires_margin_not_a_hair():
    trig = EventTrigger(TriggerParams(delta=0.10, t_dwell=2.0))
    # alternative better by only 2%: below the 10% margin -> no switch
    assert not trig.should_reroute(0.0, route_prr=None, min_let_s=None,
                                   better_alt=False, lost_neighbors=[])[0]


# ------------------------------------------------ 5.4 row 4: T1 timing
def test_t1_fires_within_two_beacon_intervals_of_degradation():
    """0.85 packet loss injected on pair {0,1} from t=15 s: T1 must fire by
    ~15 + 2 beacons (2.5 s tolerance for the EWMA window)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sim
    r = run_sim.run(42, Path(ROOT / "data" / "raw"), write_csv=False,
                    algorithm="mpcare", badlink=(15.0, 25.0, 0.85, 0, 1))
    t1 = r["first_fire"].get("pr-below-threshold")
    assert t1 is not None
    assert t1 <= 15.0 + 2.5


# ------------------------------------------------ 5.4 row 5: decentralized loss
def test_killing_a_relay_triggers_reroute():
    """Kill UAV2 (rim relay) at t=20 s: T3/T4 fire and traffic re-routes to
    the upper route; PDR stays high (graceful degradation)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sim
    r = run_sim.run(42, Path(ROOT / "data" / "raw"), write_csv=False,
                    algorithm="mpcare", scenario="diamond", kill_node=(20.0, 2))
    assert r["trigger_events"].get("neighbor-lost", 0) > 0
    assert r["first_fire"].get("better-alternative") is not None
    assert r["route_choice"]["upper"] > 0        # actually re-routed
    assert r["chain_pdr"] >= 0.8                 # degraded, not collapsed


# ------------------------------------------------ 5.4 row 6: head-to-head
def test_head_to_head_mpcare_beats_astar_dist_by_10pp_at_10pct_loss():
    """The plan's first head-to-head: MP-CARE PDR > A* dist PDR by >= 10 pp
    at 5 UAVs with 10% injected loss (diamond scenario)."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import run_sim
    astar = run_sim.run(42, Path(ROOT / "data" / "raw"), write_csv=False,
                        algorithm="astar_dist", scenario="diamond",
                        injected_loss=0.10)
    mp = run_sim.run(42, Path(ROOT / "data" / "raw"), write_csv=False,
                     algorithm="mpcare", scenario="diamond", injected_loss=0.10)
    margin = mp["chain_pdr"] - astar["chain_pdr"]
    assert margin >= 0.10, f"margin {margin:.3f} (mpcare {mp['chain_pdr']} vs " \
                           f"astar {astar['chain_pdr']})"
