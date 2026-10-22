#!/usr/bin/env python3
"""Experiment runner (implementation_plan.md section 12 / Phase 2 exit criteria).

Runs N trials of every baseline algorithm at 5 UAVs under nominal conditions,
aggregates the metrics, and produces:

  data/processed/baseline_comparison.csv   per-trial raw results
  data/processed/baseline_summary.csv      per-algorithm mean / std
  data/processed/baseline_summary.json     same, machine-readable
  data/processed/baseline_comparison.png   PDR bar chart with std error bars
                                           + control-overhead comparison

Usage:
  py -3.13 scripts/run_experiment.py                     # 30 trials x 5 algos
  py -3.13 scripts/run_experiment.py --trials 5          # quick pass
  py -3.13 scripts/run_experiment.py --algorithms gpsr aodv
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

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

METRIC_KEYS = [
    ("chain_pdr", "chain PDR"),
    ("pdr", "total PDR"),
    ("avg_latency_ms", "avg e2e latency (ms)"),
    ("min_chain_hops", "min chain hops"),
    ("control_pkts_per_node_s", "control pkts/node/s"),
    ("first_route_s", "first route (s)"),
]


def main():
    ap = argparse.ArgumentParser(description="Phase 2 baseline campaign")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000,
                    help="first seed; trial i uses seed0+i")
    ap.add_argument("--algorithms", nargs="+", default=list(run_sim.ALGORITHMS),
                    choices=run_sim.ALGORITHMS)
    ap.add_argument("--dur", type=float, default=90.0)
    ap.add_argument("--scenario", default="chain", choices=("chain", "diamond"))
    ap.add_argument("--loss", type=float, default=None,
                    help="global injected per-attempt packet loss 0..1")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    t_start = time.perf_counter()
    for algo in args.algorithms:
        for trial in range(args.trials):
            seed = args.seed0 + trial
            r = run_sim.run(seed, RAW_DIR, duration=args.dur, write_csv=False,
                            algorithm=algo, scenario=args.scenario,
                            injected_loss=args.loss)
            rows.append({
                "algorithm": algo, "trial": trial, "seed": seed,
                "chain_pdr": r["chain_pdr"], "pdr": r["pdr"],
                "chain_sent": r["chain_sent"], "chain_delivered": r["chain_delivered"],
                "min_chain_hops": r["min_chain_hops"],
                "avg_latency_ms": r["avg_latency_ms"],
                "max_arrival_s": r["max_arrival_s"],
                "control_pkts_per_node_s": r.get("control_pkts_per_node_s", ""),
                "first_route_s": r.get("first_route_s", r.get("converged_s", "")),
                "max_discovery_s": r.get("max_discovery_s", ""),
                "waypoint_ok": r["waypoint_ok"], "stability_ok": r["stability_ok"],
                "network_ok": r["network_ok"],
            })
            print(f"[{algo}] trial {trial + 1}/{args.trials} seed={seed} "
                  f"chain_pdr={r['chain_pdr']} hops>={r['min_chain_hops']} "
                  f"lat={r['avg_latency_ms']}ms")
    wall = time.perf_counter() - t_start

    # ---------------- per-trial CSV ----------------
    tag = f"_{args.tag}" if args.tag else ""
    per_trial = OUT_DIR / f"baseline_comparison{tag}.csv"
    with open(per_trial, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---------------- per-algorithm summary ----------------
    summary = {}
    for algo in args.algorithms:
        rs = [r for r in rows if r["algorithm"] == algo]
        s = {"n_trials": len(rs)}
        for key, label in METRIC_KEYS:
            vals = [r[key] for r in rs if isinstance(r[key], (int, float))]
            if vals:
                s[key] = {"mean": round(statistics.mean(vals), 4),
                          "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                          "min": min(vals), "max": max(vals)}
        s["sanity_ok"] = all(r["waypoint_ok"] and r["stability_ok"] and r["network_ok"]
                             for r in rs)
        summary[algo] = s

    with open(OUT_DIR / f"baseline_summary{tag}.json", "w", encoding="utf-8") as f:
        json.dump({"n_trials": args.trials, "duration_s": args.dur,
                   "seed0": args.seed0, "scenario": args.scenario,
                   "injected_loss": args.loss,
                   "algorithms": summary}, f, indent=2)
    with open(OUT_DIR / f"baseline_summary{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["algorithm"] + [label for _, label in METRIC_KEYS] + ["sanity_ok"])
        for algo, s in summary.items():
            w.writerow([algo] + [
                f"{s[key]['mean']} ± {s[key]['std']}" if key in s else "-"
                for key, _ in METRIC_KEYS] + [s["sanity_ok"]])

    # ---------------- chart ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        algos = args.algorithms
        means = [summary[a].get("chain_pdr", {}).get("mean", 0.0) for a in algos]
        stds = [summary[a].get("chain_pdr", {}).get("std", 0.0) for a in algos]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.bar(algos, means, yerr=stds, capsize=4,
                color=["#4472c4", "#ed7d31", "#a5a5a5", "#ffc000", "#5b9bd5"])
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("chain PDR (mean ± std)")
        ax1.set_title(f"PDR vs algorithm — {args.trials} trials, 5 UAVs, no injected loss")
        ov = [summary[a].get("control_pkts_per_node_s", {}).get("mean", 0.0) for a in algos]
        ax2.bar(algos, ov, color=["#4472c4", "#ed7d31", "#a5a5a5", "#ffc000", "#5b9bd5"])
        ax2.set_ylabel("control packets / node / s")
        ax2.set_title("Control-plane overhead")
        fig.tight_layout()
        chart = OUT_DIR / f"baseline_comparison{tag}.png"
        fig.savefig(chart, dpi=600)
        print(f"chart: {chart}")
    except ImportError:
        print("matplotlib not available — skipped the chart")

    # ---------------- console summary ----------------
    print(f"\n{'algorithm':<12} {'chain PDR':<14} {'latency ms':<14} "
          f"{'ctrl pkts/n/s':<14} {'first route s':<13} sanity")
    for algo, s in summary.items():
        def fmt(key):
            if key not in s:
                return "-"
            m, sd = s[key]["mean"], s[key]["std"]
            return f"{m:.3f} ± {sd:.3f}" if isinstance(m, float) and m < 10 else f"{m:.1f}"
        print(f"{algo:<12} {fmt('chain_pdr'):<14} {fmt('avg_latency_ms'):<14} "
              f"{fmt('control_pkts_per_node_s'):<14} {fmt('first_route_s'):<13} "
              f"{'OK' if s['sanity_ok'] else 'FAIL'}")
    print(f"\n{len(rows)} trials in {wall:.0f}s -> {per_trial}, "
          f"baseline_summary{tag}.csv/.json")
    if not all(s["sanity_ok"] for s in summary.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
