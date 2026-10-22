"""Phase 4 tests: stats protocol utilities (section 6.5), the Phase 4
scenarios/knobs in run_sim.py, and the RQ3 LET audit.

Every expected number below is hand-computable from the definitions (no
external reference tables), per the project's no-hallucination rule.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("scripts", "src/network_model", "src/mpcare_agent", "src/logger",
          "src/aodv_routing", "src/olsr_routing"):
    sys.path.insert(0, str(ROOT / p))

import math  # noqa: E402

import run_sim  # noqa: E402
from stats_utils import (mannwhitneyu, kruskal_wallis, holm_bonferroni,  # noqa: E402
                         cohens_d, rank_biserial, mean_ci95, spearman_rho)

OUT = ROOT / "data" / "raw"


# ------------------------------------------------------------ stats utilities

def test_mannwhitney_complete_separation():
    # x strictly below y -> all (x,y) pairs favor y: U = 0
    r = mannwhitneyu([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    assert r["U"] == 0.0
    assert r["n_x"] == 3 and r["n_y"] == 3
    assert 0.0 < r["p"] < 0.15


def test_mannwhitney_interleaved():
    # x = [1,3,5] vs y = [2,4,6]: pairs with x>y: only (3>2) and (5>2),(5>4)
    # -> u1 = 3; U = min(3, 6) = 3
    r = mannwhitneyu([1.0, 3.0, 5.0], [2.0, 4.0, 6.0])
    assert r["U"] == 3.0


def test_mannwhitney_complete_tie():
    r = mannwhitneyu([1.0, 1.0], [1.0, 1.0])
    assert r["U"] == 2.0            # mu = n1*n2/2 = 2 -> no deviation
    assert r["p"] == 1.0


def test_kruskal_wallis_hand_case():
    # groups [1,2] vs [3,4]: ranks 1,2,3,4; H = 12/20 * (9/2 + 49/2) - 15 = 2.4
    r = kruskal_wallis([[1.0, 2.0], [3.0, 4.0]])
    assert math.isclose(r["H"], 2.4, rel_tol=1e-12)
    assert r["df"] == 1


def test_kruskal_wallis_identical_groups():
    r = kruskal_wallis([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    assert r["H"] == 0.0 and r["p"] == 1.0


def test_holm_bonferroni_order_preserved():
    # m=3: sorted 0.01,0.03,0.04 -> adj 0.03, 0.06, then max(1x0.04, 0.06)=0.06
    # (step-down with enforced monotonicity)
    assert holm_bonferroni([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    assert holm_bonferroni([None, 0.5]) == [None, 0.5]


def test_cohens_d_and_rank_biserial():
    # x=[1,2,3], y=[4,5,6]: pooled sd = 1, means 2 vs 5 -> d = -3
    assert math.isclose(cohens_d([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), -3.0)
    # complete separation in favor of y -> rank-biserial = -1
    assert math.isclose(rank_biserial([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), -1.0)


def test_mean_ci95_hand_case():
    # [1..10]: mean 5.5, s = sqrt(82.5/9) = 3.02765, se = 0.95743, t(9) = 2.262
    # -> half-width = 2.1659
    m, half = mean_ci95([float(i) for i in range(1, 11)])
    assert math.isclose(m, 5.5)
    assert abs(half - 2.1659) < 5e-4


def test_spearman_perfect_monotone():
    assert math.isclose(spearman_rho([1, 2, 3, 4], [2, 4, 6, 8]), 1.0)
    assert math.isclose(spearman_rho([1, 2, 3, 4], [8, 6, 4, 2]), -1.0)


# ------------------------------------------------------------ sim scenarios

def test_mobility_deterministic_and_speed_audit():
    a = run_sim.run(11, OUT, duration=40.0, write_csv=False, algorithm="gpsr",
                    scenario="mobility", speed=10.0)
    b = run_sim.run(11, OUT, duration=40.0, write_csv=False, algorithm="gpsr",
                    scenario="mobility", speed=10.0)
    assert a["final_positions"] == b["final_positions"]
    assert a["chain_pdr"] == b["chain_pdr"]
    assert a["speed_ok"] is True
    assert 9.0 <= a["speed_mps"] <= 11.0


def test_mobility_mpcare_let_audit_present():
    r = run_sim.run(11, OUT, duration=45.0, write_csv=False, algorithm="mpcare",
                    scenario="mobility", speed=20.0)
    assert r.get("let_audit_n", 0) > 0
    assert "let_median_rel_err" in r and "let_spearman" in r


def test_grid_kill_applied_and_recovery_recorded():
    r = run_sim.run(3, OUT, duration=60.0, write_csv=False, algorithm="mpcare",
                    scenario="grid", n_uav=10, kill_node=[(20.0, 3)])
    assert r["network_ok"] is True          # kill applied (grid check 4)
    assert r.get("recovery_s") is not None  # recovery metric recorded
    assert r["min_chain_hops"] is not None


def test_injected_latency_reaches_delivery_path():
    base = run_sim.run(7, OUT, duration=60.0, write_csv=False, algorithm="aodv",
                       scenario="chain", injected_latency_ms=0.0)
    slow = run_sim.run(7, OUT, duration=60.0, write_csv=False, algorithm="aodv",
                       scenario="chain", injected_latency_ms=100.0)
    assert base["min_chain_hops"] is not None and base["min_chain_hops"] >= 4
    # 4+ hops x 100 ms + airtime/queue: the sweep showed ~403 ms for this seed
    assert slow["avg_latency_ms"] > 300.0
    assert slow["avg_latency_ms"] > base["avg_latency_ms"] + 300.0


def test_ten_uav_chain():
    r = run_sim.run(7, OUT, duration=60.0, write_csv=False, algorithm="aodv",
                    scenario="chain", n_uav=10)
    assert r["min_chain_hops"] is not None and r["min_chain_hops"] >= 8
    assert r["chain_pdr"] > 0.9


def test_chain_regression_unchanged():
    r = run_sim.run(42, OUT, duration=90.0, write_csv=False, algorithm="gpsr",
                    scenario="chain")
    assert r["chain_pdr"] == 1.0 and r["min_chain_hops"] >= 4


def test_diamond_regression_unchanged():
    r = run_sim.run(1000, OUT, duration=90.0, write_csv=False,
                    algorithm="mpcare", scenario="diamond")
    # Phase 3 head-to-head: all 50 benchmark deliveries take the rim route
    assert r["chain_pdr"] == 1.0 and r["route_choice"] == {"upper": 0, "rim": 50}


# ------------------------------------------------------------- sweep tooling

def test_kills_schedule_deterministic_and_valid():
    from run_sweeps import _kills_for, KILL_POOL
    a = _kills_for(1234, 3)
    b = _kills_for(1234, 3)
    assert a == b
    assert len(a) == 3
    assert all(nid in KILL_POOL for _, nid in a)
    assert [t for t, _ in a] == sorted(t for t, _ in a)
    assert all(15.0 <= t <= 30.0 for t, _ in a)
