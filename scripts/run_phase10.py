#!/usr/bin/env python3
"""Loss-shift dynamic-rescoring campaign (phase 10).

Purpose: give the "dynamic re-scoring" contribution an honest, dedicated
experiment instead of an extrapolation. On the diamond scenario the global
injected loss follows a step schedule per condition:

  A  loss_0->20@15s   0% until t=15 s, then 20% for the rest of the run
  B  loss_10->20@15s  constant 10% baseline control (static comparison anchor)
  C  loss_0@15->20@45 TRUE step in-flight: 0% until 15 s, 20% in [15, 45),
                      then 0% again, forcing at least TWO re-decisions

Algorithms: full eight-protocol suite, so the static ETX-oracle A* (recomputed
per packet at send time over the true instantaneous physics) is directly
comparable head-to-head, and MP-CARE's reactivity class is visible.

Evaluated on the standard 30 evaluation seeds (1000..1029), same as phases
4/8/9. Outputs data/processed/phase10_lossshift.csv + _summary.json and a
paper-ready figure.

Usage:
  py -3.13 scripts/run_phase10.py                 # 30 seeds, full suite
  py -3.13 scripts/run_phase10.py --trials 3 --tag _smoke
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
sys.path.insert(0, str(ROOT / "src" / "aodv_routing"))
sys.path.insert(0, str(ROOT / "src" / "olsr_routing"))
sys.path.insert(0, str(ROOT / "src" / "qlgr_routing"))

import run_sim  # noqa: E402
import run_phase8  # noqa: E402  (table loader + flatten reuse)
from stats_utils import mannwhitneyu, holm_bonferroni, rank_biserial  # noqa: E402

OUT = ROOT / "data" / "processed"
EVAL_SEED0 = 1000
TRIALS = 30
DURATION = 90.0

ALGORITHMS = ["astar_dist", "gpsr", "aodv", "olsr", "letolsr",
              "astar_etx", "qlgr", "mpcare"]

# (label, schedule, base injected_loss)
CONDITIONS = [
    ("lossstep_0_to20_at15",  [(15.0, 1e9, 0.20)], 0.0),
    ("lossstep_10to20_at15",  [(15.0, 1e9, 0.20)], 0.10),
    ("lossstep_0_20_0",       [(15.0, 45.0, 0.20)], 0.0),
]

ROW_KEYS = run_phase8.ROW_KEYS + ["loss_base", "loss_sched"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    tables = run_phase8.load_saved_tables(OUT)
    rows: list[dict] = []
    t_start = time.perf_counter()
    for label, sched, base in CONDITIONS:
        for algo in ALGORITHMS:
            for trial in range(args.trials):
                seed = args.seed0 + trial
                r = run_sim.run(
                    seed, ROOT / "data" / "raw",
                    duration=DURATION, write_csv=False,
                    algorithm=algo, scenario="diamond",
                    injected_loss=base,
                    loss_schedule=sched,
                    qlgr_opts={"learn": False, "seed": seed,
                               "table": tables.get("diamond")},
                )
                rows.append(run_phase8._flatten(r, label, {
                    "scenario": "diamond", "speed_mps": None,
                    "loss": base, "shadow_sigma_db": 0.0,
                }) | {"loss_base": base,
                      "loss_sched": json.dumps(sched)})
                print(f"[{label}] {algo} trial {trial} "
                      f"pdr={r['pdr']:.3f} reroutes={r.get('reroutes')}",
                      flush=True)

    name = f"phase10_lossshift{args.tag}"
    csv_path = OUT / f"{name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_KEYS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in ROW_KEYS})

    # ---- per-condition per-algorithm means + MP-CARE head-to-heads
    summary: dict = {"experiment": "phase10_lossshift",
                     "duration_s": DURATION, "n_trials": args.trials,
                     "seed0": args.seed0}
    per_algo: dict[str, dict[str, float]] = {}
    for label, _, _ in CONDITIONS:
        for algo in ALGORITHMS:
            vals = [float(r["pdr"]) for r in rows
                    if r["condition"] == label and r["algorithm"] == algo]
            if vals:
                per_algo.setdefault(label, {})[algo] = statistics.mean(vals)

    posthoc: dict[str, dict] = {}
    for label, _, _ in CONDITIONS:
        mp = [float(r["pdr"]) for r in rows
              if r["condition"] == label and r["algorithm"] == "mpcare"]
        etx = [float(r["pdr"]) for r in rows
               if r["condition"] == label and r["algorithm"] == "astar_etx"]
        olsr_ = [float(r["pdr"]) for r in rows
                 if r["condition"] == label and r["algorithm"] == "olsr"]
        aodv_ = [float(r["pdr"]) for r in rows
                 if r["condition"] == label and r["algorithm"] == "aodv"]
        res = {}
        for nm, vals in (("astar_etx", etx), ("olsr", olsr_), ("aodv", aodv_)):
            if not mp or not vals:
                continue
            u_p = mannwhitneyu(mp, vals)["p"]   # two-sided
            res[nm] = {"mpcare_mean": statistics.mean(mp),
                       "baseline_mean": statistics.mean(vals),
                       "margin_pp": 100 * (statistics.mean(mp)
                                           - statistics.mean(vals)),
                       "mw_p": u_p,
                       "holm_p": min(1.0, 3 * u_p),
                       "rank_biserial": rank_biserial(mp, vals)}
        posthoc[label] = res

    summary["per_algo_mean_pdr"] = per_algo
    summary["posthoc_vs_mpcare"] = posthoc
    summary["wall_s"] = round(time.perf_counter() - t_start, 1)
    with open(OUT / f"{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)

    print(json.dumps(posthoc, indent=1))
    print(f"[phase10] -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
