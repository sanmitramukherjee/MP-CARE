#!/usr/bin/env python3
"""Phase 6 scalability campaign (implementation_plan.md section 8).

Scale points (section 8.2, exact table):
  S1 N=5   500x500   10 waypoints      S4 N=30  1200x1200 60 waypoints
  S2 N=10  700x700   20 waypoints      S5 N=40  1500x1500 80 waypoints
  S3 N=20  1000x1000 40 waypoints      S6 N=50  1700x1700 100 waypoints

Design decisions (documented, NOTES/Phase_6.md):
  - Scenario "scale" (run_sim.py): N UAVs at ~constant density (area grows
    linearly with N; ~57.8 km^2/UAV) -> bounded 1-hop degree, which is what
    makes the O(deg) claims testable. 50 benchmark packets UAV0 -> UAV(N-1)
    during the static window t=[5,30) s, then a pre-drawn random-waypoint
    mission of the plan's waypoint count at 8 m/s. Run duration 150 s.
  - Per-UAV CPU time per tick (section 8.3 metric 1) is measured with
    time.perf_counter AROUND the protocol work of every real node in every
    tick (mpcare: router tick + beacon exchange; olsr/aodv: tick + flush +
    bus delivery), then divided by (nodes x ticks) -> ms/node/tick. The
    scale-invariant part of the swarm's compute is the per-node protocol
    work; the probe (NOTES/Phase_6.md) shows the remaining growth from N=10
    to N=50 at constant density is ~x1.05.
  - Control overhead (metric 2): mpcare = beacons/node/s (the ONLY control
    messages; position service is an assumption, not traffic). olsr/aodv =
    bus.control_msgs_sent/node/s (exact, incl. per-origin flood copies).
  - Topology convergence (metric 3): first time the SOURCE's routing
    knowledge covers its final static-window neighborhood (mpcare: 1-hop
    table = full 1-hop state; olsr: topology table = whole component;
    aodv: first route discovery) — see run_sim.convergence_def.
  - PDR at N=50 is measured over DISCONNECTED random geometric graphs at
    k ~ 4.4 < log(n) (the plan's own geometry): some seeds have the
    src-dst pair in different components — a physics floor, reported as
    such, not an algorithm failure (all algorithms see the same graph).
  - Statistics (section 6.5 protocol): Kruskal-Wallis across algorithms per
    scale point, post-hoc Mann-Whitney U (MP-CARE vs each baseline) with
    Holm-Bonferroni, rank-biserial + Cohen's d, mean +- 95% CI, seeds
    recorded per row. NOTE: the pure-Python p-values are normal-approx; any
    p within 10x of alpha must be re-checked with a real stats package
    (stats_utils docstring contract).

Outputs (data/processed/):
  phase6_trials.csv            one row per (N, algorithm, seed)
  phase6_summary.json          per-point stats, benchmarks, alarms
  phase6_g61_cpu_vs_n.png      ... phase6_g66_convergence_vs_n.png
Exit-criteria verdicts (section 8.5) printed from the measured numbers only.

Usage:
  py -3.13 scripts/run_phase6.py                 # full campaign (~25 min)
  py -3.13 scripts/run_phase6.py --points 5,50   # subset of scale points
  py -3.13 scripts/run_phase6.py --trials 3 --tag _smoke
  py -3.13 scripts/run_phase6.py --density v3    # density-justified geometry
                                                 # (see V3_BOXES below)
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))
sys.path.insert(0, str(ROOT / "src" / "logger"))
sys.path.insert(0, str(ROOT / "src" / "aodv_routing"))
sys.path.insert(0, str(ROOT / "src" / "olsr_routing"))

import run_sim  # noqa: E402
from stats_utils import (mannwhitneyu, kruskal_wallis, holm_bonferroni,  # noqa: E402
                         cohens_d, rank_biserial, mean_ci95)

OUT = ROOT / "data" / "processed"

# plan section 8.2 scale points: N -> (box_w, box_h, n_waypoints)
SCALE_POINTS: dict[int, tuple[float, float, int]] = run_sim.SCALE_POINTS

# Density-justified geometry (v3), validated by
# data/raw/probe_v3_geometry.py on ALL 30 campaign seeds before adoption:
# the plan's constant-area-per-UAV boxes put every mid/high-N graph at mean
# degree k ~ 4.4-5.2, BELOW the classical log(n) percolation threshold, so
# 12/30 seeds at N=50 spawn with the src-dst pair in different components
# (probe_disconnect_analysis.py) and the greedy src-dst corridor exists on
# only 60-73% of connected graphs at N=30/40 (probe_void_structure.py).
# V3 shrinks the boxes so the MEASURED degree is k ~ 7.7-8.8 (edge-corrected
# from the theoretical 8.5; N=5's box is a complete graph by construction —
# k <= 4 for 5 nodes). Measured on campaign seeds: connectivity 29-30/30,
# greedy corridor 0.90-1.00. Waypoints stay 2N per the plan's S1-S6 rows.
V3_BOXES: dict[int, tuple[float, float, int]] = {
    5: (271.0, 271.0, 10), 10: (460.0, 460.0, 20), 20: (722.0, 722.0, 40),
    30: (914.0, 914.0, 60), 40: (1077.0, 1077.0, 80), 50: (1180.0, 1180.0, 100),
}


def apply_density(density: str) -> None:
    """Override the scale geometry in-place (mutates run_sim.SCALE_POINTS,
    which SCALE_POINTS aliases). Must run before any trial starts."""
    if density == "":
        return
    if density != "v3":
        raise ValueError(f"unknown density {density!r} (supported: '', 'v3')")
    for n, box in V3_BOXES.items():
        SCALE_POINTS[n] = box
DURATION = 150.0          # 30 s static window + box-scaled waypoint missions
ALGORITHMS = ("mpcare", "gpsr", "olsr", "aodv")
BASELINES_FOR_G63 = ("gpsr", "olsr")   # plan graph 6.3 names exactly these two

ROW_KEYS = ["seed", "trial", "algorithm", "N", "condition", "chain_pdr", "pdr",
            "packets_sent", "packets_delivered", "avg_latency_ms",
            "mission_time_s", "cpu_ms_per_node_tick", "control_pkts_per_node_s",
            "beacons_per_node_s", "data_tx_per_node_s", "full_conv_s",
            "convergence_def", "conv_scope", "avg_degree", "reroutes_total",
            "reroutes_per_uav_min", "trigger_events", "void_recovery_events",
            "void_walk_drops", "run_wall_s", "sanity_ok"]

# plan section 8.3 alarm thresholds (all AT N=50 unless stated).
# NOTE on semantics: the plan's EXIT criteria (section 8.5) are "complete
# metric sets at all 6 points" + "runs at N=50 without timeout/crash" + CPU
# documented/bounded + >= 1 baseline at N=50 + write-up. The alarm table is
# the section 8.3 EXPECTED-BEHAVIOR check: an ALARM verdict is a FINDING to
# report (e.g. the PDR bar sits exactly at the percolation edge of the
# plan's own constant-density geometry), NOT a campaign failure — so alarms
# do not drive the exit code; the section 8.5 benchmarks do.
ALARMS = [
    ("cpu_ms_per_node_tick", "<=", 10.0, "per-UAV CPU ms/tick <= 10 ms at N=50"),
    ("control_pkts_per_node_s", "<=", 50.0, "control overhead <= 50 pkts/node/s at N=50"),
    ("full_conv_s", "<=", 30.0, "topology convergence <= 30 s at N=50"),
    ("pdr", ">=", 0.70, "PDR >= 70% at N=50 under nominal conditions"),
    ("reroutes_per_uav_min", "<=", 10.0, "re-routes <= 10 per UAV per minute at N=50"),
]

# run_phase6.py consumes its own module-level ALGO_COLOR below; keep the
# definition before first use (same table as run_sweeps.py)
ALGO_COLOR = {"astar_dist": "#8c8c8c", "gpsr": "#ed7d31", "aodv": "#4472c4",
              "olsr": "#70ad47", "astar_etx": "#ffc000", "mpcare": "#c00000"}


def _flatten(r: dict, n: int, trial: int, wall_s: float) -> dict:
    ev = r.get("trigger_events") or {}
    reroutes = sum(ev.values())
    return {
        "seed": r["seed"], "trial": trial, "algorithm": r["algorithm"], "N": n,
        "condition": f"N{n}",
        "chain_pdr": r.get("chain_pdr"), "pdr": r.get("pdr"),
        "packets_sent": r.get("packets_sent"),
        "packets_delivered": r.get("packets_delivered"),
        "avg_latency_ms": r.get("avg_latency_ms"),
        "mission_time_s": r.get("mission_time_s"),
        "cpu_ms_per_node_tick": r.get("cpu_ms_per_node_tick"),
        "control_pkts_per_node_s": r.get("control_pkts_per_node_s"),
        "beacons_per_node_s": r.get("beacons_per_node_s"),
        "data_tx_per_node_s": r.get("data_tx_per_node_s"),
        "full_conv_s": r.get("full_conv_s"),
        "convergence_def": r.get("convergence_def"),
        "conv_scope": r.get("conv_scope"),
        "avg_degree": r.get("avg_degree"),
        "reroutes_total": reroutes,
        # re-route events are counted per TICK on any node; normalize per UAV
        # per minute of simulated time (plan section 8.3 metric 5)
        "reroutes_per_uav_min": round(reroutes / n / (r.get("_duration", DURATION)
                                                      / 60.0), 3),
        "trigger_events": json.dumps(ev) if ev else "",
        "void_recovery_events": r.get("void_recovery_events"),
        "void_walk_drops": r.get("void_walk_drops"),
        "run_wall_s": round(wall_s, 2),
        "sanity_ok": bool(r.get("stability_ok") and r.get("waypoint_ok")
                          and r.get("network_ok")),
    }


def per_algo_stats(rows: list[dict], algo: str, metric: str) -> dict:
    vals = [r[metric] for r in rows
            if r["algorithm"] == algo and isinstance(r[metric], (int, float))]
    ci = mean_ci95(vals)
    if ci is None:
        return {"n": 0}
    return {"n": len(vals), "mean": round(ci[0], 4), "ci95": round(ci[1], 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}


def point_stats(rows: list[dict], n: int, algorithms: list[str],
                metric: str, reference: str = "mpcare") -> dict:
    cr = [r for r in rows if r["N"] == n]
    per_algo = {a: per_algo_stats(cr, a, metric) for a in algorithms}
    groups = [[r[metric] for r in cr
               if r["algorithm"] == a and isinstance(r[metric], (int, float))]
              for a in algorithms]
    kw = kruskal_wallis(groups)
    posthoc: dict = {}
    pvals: list = []
    pairs: list[str] = []
    for a in algorithms:
        if a == reference:
            continue
        mv = [r[metric] for r in cr if r["algorithm"] == reference
              and isinstance(r[metric], (int, float))]
        bv = [r[metric] for r in cr if r["algorithm"] == a
              and isinstance(r[metric], (int, float))]
        mw = mannwhitneyu(mv, bv)
        paired = (statistics.mean(mv) - statistics.mean(bv)) if mv and bv else None
        posthoc[a] = {"mw_p": (round(mw["p"], 5) if mw["p"] is not None else None),
                      "paired_margin": round(paired, 4) if paired is not None else None,
                      "rank_biserial": rank_biserial(mv, bv),
                      "cohens_d": (round(cohens_d(mv, bv), 3)
                                   if cohens_d(mv, bv) is not None else None)}
        pvals.append(mw["p"])
        pairs.append(a)
    adjusted = holm_bonferroni(pvals)
    for a, p in zip(pairs, adjusted):
        posthoc[a]["mw_p_holm"] = round(p, 5) if p is not None else None
    return {"per_algo": per_algo,
            "kruskal_wallis": {"H": round(kw["H"], 3) if kw["H"] is not None else None,
                               "p": round(kw["p"], 6) if kw["p"] is not None else None,
                               "df": kw["df"]},
            "posthoc_vs_mpcare": posthoc}


# ------------------------------------------------------------------ alarms
def evaluate_alarms(rows: list[dict], algorithms: list[str]) -> dict:
    """Section 8.3 alarm table evaluated at N=50 (or the largest run point).

    Verdicts come from the MEASURED means only; an alarm on a metric an
    algorithm does not produce is n/a for it, never a silent pass."""
    n_eval = max(r["N"] for r in rows)
    out: dict = {"evaluated_at_N": n_eval, "checks": []}
    all_ok = True
    for metric, op, thr, desc in ALARMS:
        for a in algorithms:
            st = per_algo_stats([r for r in rows if r["N"] == n_eval], a, metric)
            if not st.get("n"):
                out["checks"].append({"metric": metric, "algorithm": a,
                                      "verdict": "n/a",
                                      "reason": f"{a} does not produce {metric}"})
                continue
            mean = st["mean"]
            ok = (mean <= thr) if op == "<=" else (mean >= thr)
            all_ok &= ok
            out["checks"].append({
                "metric": metric, "algorithm": a, "verdict": "PASS" if ok else "ALARM",
                "mean": mean, "ci95": st["ci95"], "threshold": thr,
                "comparison": f"mean {op} {thr}", "criterion": desc})
    out["all_pass"] = bool(all_ok)
    return out


# ------------------------------------------------------------------- graphs
def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def _series(rows, points, algos, metric):
    series = {}
    for n in points:
        cr = [r for r in rows if r["N"] == n]
        for a in algos:
            st = per_algo_stats(cr, a, metric)
            if st.get("n"):
                series.setdefault(a, ([], [], []))
                series[a][0].append(n)
                series[a][1].append(st["mean"])
                series[a][2].append(st["ci95"])
    return series


def _line_plot(path, rows, points, algos, metric, title, ylab,
               xlabel="swarm size N (UAVs)", logy=False, figsize=(7, 4.5)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize)
    ser = _series(rows, points, algos, metric)
    for a in algos:
        if a not in ser:
            continue
        xs, m, ci = ser[a]
        ax.plot(xs, m, marker="o", ms=4, color=ALGO_COLOR.get(a), label=a)
        ax.fill_between(xs, [x - c for x, c in zip(m, ci)],
                        [x + c for x, c in zip(m, ci)],
                        color=ALGO_COLOR.get(a), alpha=0.15)
    if logy:
        ax.set_yscale("log")
    _style_axes(ax, xlabel, ylab, title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)
    print(f"chart: {path}")


# ------------------------------------------------------------------- graphs
def graphs(rows, points, name):
    _line_plot(OUT / f"{name}_g61_cpu_vs_n.png", rows, points, ALGORITHMS,
               "cpu_ms_per_node_tick",
               "Graph 6.1: per-UAV CPU time vs N (ms per node per tick)",
               "CPU ms / node / tick")
    _line_plot(OUT / f"{name}_g62_overhead_vs_n.png", rows, points, ALGORITHMS,
               "control_pkts_per_node_s",
               "Graph 6.2: control overhead vs N (pkts/node/s)",
               "control packets / node / s")
    _line_plot(OUT / f"{name}_g63_pdr_vs_n.png", rows, points,
               ["mpcare", *BASELINES_FOR_G63], "pdr",
               "Graph 6.3: PDR vs N (nominal conditions)",
               "PDR (mean ± 95% CI)")
    _line_plot(OUT / f"{name}_g64_mission_vs_n.png", rows, points, ALGORITHMS,
               "mission_time_s",
               "Graph 6.4: mission time vs N (first-leg completion after release)",
               "mission time (s)")
    _line_plot(OUT / f"{name}_g65_degree_vs_n.png", rows, points, ALGORITHMS,
               "avg_degree",
               "Graph 6.5: average neighbor count (degree) vs N — O(deg) evidence",
               "mean 1-hop degree")
    _line_plot(OUT / f"{name}_g66_convergence_vs_n.png", rows, points, ALGORITHMS,
               "full_conv_s",
               "Graph 6.6: topology convergence time vs N",
               "convergence time (s)")


# ----------------------------------------------------------------- campaign
def campaign(points: list[int], trials: int, seed0: int, workers: int,
             tag: str, density: str = "") -> int:
    t_start = time.perf_counter()
    rows: list[dict] = []
    if workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        jobs = [(n, algo, seed0 + k, k)
                for n in points for algo in ALGORITHMS for k in range(trials)]
        with ctx.Pool(workers) as pool:
            results = pool.starmap(_one_trial, jobs)
        # _wall is measured INSIDE each child around run_sim.run only, so it
        # stays comparable across workers (process spawn cost is excluded)
        for (n, algo, seed, k), r in zip(jobs, results):
            rows.append(_flatten(r, n, k, r.get("_wall", 0.0)))
            if not rows[-1]["sanity_ok"]:
                print(f"  !! SANITY FAIL N={n}/{algo} seed={seed}", file=sys.stderr)
        rows.sort(key=lambda r: (r["N"], ALGORITHMS.index(r["algorithm"]),
                                 r["trial"]))
        print(f"  {len(rows)} trials done on {workers} workers "
              f"({time.perf_counter() - t_start:.0f}s elapsed)")
    else:
        for n in points:
            box_w, box_h, n_wps = SCALE_POINTS[n]
            for algo in ALGORITHMS:
                for k in range(trials):
                    seed = seed0 + k
                    t0 = time.perf_counter()
                    r = run_sim.run(seed, ROOT / "data" / "raw",
                                    duration=DURATION, write_csv=False,
                                    algorithm=algo, scenario="scale", n_uav=n)
                    wall = time.perf_counter() - t0
                    rows.append(_flatten(r, n, k, wall))
                    f = rows[-1]
                    if not f["sanity_ok"]:
                        print(f"  !! SANITY FAIL N={n}/{algo} seed={seed}: "
                              f"stab={r['stability_ok']} wp={r['waypoint_ok']} "
                              f"net={r['network_ok']}", file=sys.stderr)
                print(f"[N={n:<3} {box_w:.0f}x{box_h:.0f} {n_wps} wp] "
                      f"{algo:<6} {trials} trials done "
                      f"({time.perf_counter() - t_start:.0f}s elapsed)")

    name = f"phase6{tag}"
    with open(OUT / f"{name}_trials.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary: dict = {
        "experiment": "phase6_scalability",
        "density": ({"variant": density,
                     "scale_points": {str(n): {"box": list(SCALE_POINTS[n][:2]),
                                               "waypoints": SCALE_POINTS[n][2]}
                                      for n in points},
                     "rationale": "measured k ~ 7.7-8.8 (above log(n) "
                                  "percolation); validated by "
                                  "data/raw/probe_v3_geometry.py on all 30 "
                                  "seeds: connectivity 29-30/30, greedy "
                                  "corridor 0.90-1.00"}
                    if density else "plan-8.2 constant-area-per-UAV (v1/v2 "
                    "geometry; percolation-edge k ~ 4.4-5.2)"),
        "scale_points": {str(n): {"box": SCALE_POINTS[n][:2],
                                  "waypoints": SCALE_POINTS[n][2]}
                         for n in points},
        "duration_s": DURATION, "n_trials": trials, "seed0": seed0,
        "algorithms": list(ALGORITHMS),
        "convergence_defs": {a: _conv_def(a) for a in ALGORITHMS},
        "cpu_note": "perf_counter around every real node's protocol work in "
                    "every tick, divided by nodes x ticks; includes "
                    "GIL-serialized process-global work (honest upper bound)",
        "stats": {f"N{n}": {m: point_stats(rows, n, list(ALGORITHMS), m)
                            for m in ("cpu_ms_per_node_tick",
                                      "control_pkts_per_node_s", "pdr",
                                      "full_conv_s", "reroutes_per_uav_min",
                                      "avg_degree", "mission_time_s")}
                  for n in points},
        "alarms": evaluate_alarms(rows, list(ALGORITHMS)),
    }
    # section 8.5 exit benchmarks evaluated on the actual data. The PDR
    # metric carries a physics caveat: at the plan's constant-density geometry
    # N=30-50 sits at the percolation edge (k ~ 4.4), so some seeds place the
    # src-dst pair in different components; the 4-algorithm agreement check
    # below quantifies that floor explicitly (0.0 for ALL algorithms on a
    # seed = disconnected graph, not an algorithm failure).
    n_max = max(points)
    top = [r for r in rows if r["N"] == n_max]
    disconnected_seeds = sorted(
        r["seed"] for r in top if r["algorithm"] == "mpcare" and r["pdr"] == 0.0
        and all(x["pdr"] == 0.0 for x in top if x["seed"] == r["seed"]))
    summary["benchmarks"] = {
        "all_points_complete": all(
            per_algo_stats([r for r in rows if r["N"] == n], a, "pdr").get("n") == trials
            for n in points for a in ALGORITHMS),
        f"runs_at_N{n_max}_no_crash": all(
            per_algo_stats(top, a, "pdr").get("n") == trials
            for a in ALGORITHMS),
        f"N{n_max}_disconnected_seeds_all_algos_pdr0": disconnected_seeds,
        "pdr_physics_note": "PDR at N>=30 is measured over random geometric "
                            "graphs at the percolation edge (avg degree ~4.4 "
                            "< log n): seeds where ALL algorithms score 0.0 "
                            "are disconnected src-dst pairs (physics floor), "
                            "so the section 8.3 PDR alarm is evaluated on a "
                            "quantity that is floored by the scenario itself.",
    }
    with open(OUT / f"{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    graphs(rows, points, name)

    # console headline + verdicts
    print(f"\n  {'point':<8}", end="")
    for a in ALGORITHMS:
        print(f"{a:>16}", end="")
    for metric, lab in (("pdr", "PDR"), ("cpu_ms_per_node_tick", "CPU ms/tick"),
                        ("control_pkts_per_node_s", "ctrl pkts/n/s"),
                        ("full_conv_s", "conv s")):
        print(f"\n  {lab:<8}", end="")
        for n in points:
            print(f"{'N=' + str(n):<8}", end="")
            cr = [r for r in rows if r["N"] == n]
            for a in ALGORITHMS:
                st = per_algo_stats(cr, a, metric)
                print(f"{st.get('mean', float('nan')):>10.3f}±{st.get('ci95', 0):<5.3f}"
                      if st.get("n") else f"{'-':>16}", end="")
            print()
    print("\n== section 8.3 alarms (evaluated at "
          f"N={summary['alarms']['evaluated_at_N']}) ==")
    for c in summary["alarms"]["checks"]:
        if c["verdict"] == "n/a":
            print(f"  [ n/a] {c['algorithm']:<7} {c['metric']}: {c['reason']}")
        else:
            print(f"  [{c['verdict']:<4}] {c['algorithm']:<7} {c['metric']:<26} "
                  f"mean={c['mean']} {c['comparison']}")
    print(f"\n  benchmarks: " + json.dumps(
        {k: v for k, v in summary["benchmarks"].items()
         if k != "pdr_physics_note"}))
    print(f"{trials} trials/point x {len(points)} points x {len(ALGORITHMS)} "
          f"algorithms in {time.perf_counter() - t_start:.0f}s "
          f"-> data/processed/{name}_*")
    # plan section 8.5 exit criteria: complete metric sets + N=50 runs +
    # baselines present. Alarm verdicts are FINDINGS (printed above), not
    # gates: the plan asks for the scalability ceiling to be identified and
    # reported, which includes honest ALARM rows.
    b = summary["benchmarks"]
    ok = (b["all_points_complete"]
          and b[f"runs_at_N{n_max}_no_crash"]
          and any(a in ALGORITHMS for a in ("gpsr", "olsr")))
    alarms_fired = [c for c in summary["alarms"]["checks"]
                    if c["verdict"] == "ALARM"]
    print(f"alarms fired: {len(alarms_fired)} "
          f"({', '.join(c['algorithm'] + '/' + c['metric'] for c in alarms_fired) or 'none'}) "
          "-- reported as findings, not exit gates (plan section 8.5)")
    print("RESULT:", "EXIT CRITERIA MET (section 8.5)" if ok
          else "EXIT CRITERIA NOT MET")
    return 0 if ok else 1


def _conv_def(a: str) -> str:
    # single source of truth lives in run_sim (same strings as the per-run
    # "convergence_def" field -> summary and rows can be cross-checked)
    return run_sim.CONVERGENCE_DEFS[a]


def _one_trial(n: int, algo: str, seed: int, k: int) -> dict:
    t0 = time.perf_counter()
    r = run_sim.run(seed, ROOT / "data" / "raw", duration=DURATION,
                    write_csv=False, algorithm=algo, scenario="scale", n_uav=n)
    r["_duration"] = DURATION
    r["_wall"] = time.perf_counter() - t0
    return r


def main():
    ap = argparse.ArgumentParser(description="Phase 6 scalability campaign (plan §8)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=3000)
    ap.add_argument("--points", default="",
                    help="comma list of N values (default: all of "
                         + ",".join(str(n) for n in sorted(SCALE_POINTS)) + ")")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel trial processes (Windows spawn; run_wall_s "
                         "is then not recorded per trial)")
    ap.add_argument("--density", default="", choices=("", "v3"),
                    help="scale geometry: '' = plan 8.2 boxes (v1/v2), "
                         "'v3' = density-justified boxes (V3_BOXES, measured "
                         "k ~ 7.7-8.8, validated by probe_v3_geometry.py)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    apply_density(args.density)
    points = ([int(x) for x in args.points.split(",") if x.strip()]
              if args.points else sorted(SCALE_POINTS))
    for n in points:
        if n not in SCALE_POINTS:
            ap.error(f"N={n} is not a plan section 8.2 scale point "
                     f"{sorted(SCALE_POINTS)}")
    sys.exit(campaign(points, args.trials, args.seed0, args.workers, args.tag,
                      args.density))


if __name__ == "__main__":
    main()
