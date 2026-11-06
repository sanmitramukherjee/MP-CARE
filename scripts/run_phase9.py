#!/usr/bin/env python3
"""Phase 9 — log-normal shadow-fading sensitivity campaigns.

Tests the paper's standing claim ("MP-CARE's advantages over distance-only
routing would be amplified under shadow fading, where link-quality
heterogeneity increases") empirically, and adds a prediction-augmented
comparator (LET-OLSR) to every condition.

Channel upgrade (network_model.model.NetworkParams):
  shadow_sigma_db in {0, 2, 4}: per-link log-normal shadow offset in dB,
  temporally correlated (Gauss-Markov, tau = 5 s), drawn by the channel RNG
  stream only (sigma = 0 reproduces the deterministic-LoS baseline exactly).
  Sourcing of the levels:
    sigma = 4 dB  — the standardized 3GPP TR 38.901 Table 7.4.1-1 LoS
                    shadow-fading deviation for macro/canyon scenarios
                    (RMa ~4.2 / UMa 4 / UMi 4 dB);
    sigma = 2 dB  — the low-deviation end of published air-to-air LoS
                    measurements at altitude (elevated LoS links);
    sigma = 6 dB  — intermediate / partially-obstructed regime, halfway to
                    the ~8 dB terrestrial NLoS values.
  The snapshot ETX (distance-derived) stays blind to shadowing, exactly as
  the physics delivers heterogeneity the snapshot cannot predict — so
  measurement-driven triggers are the only adaptors.

Conditions:
  diamond: sigma x loss grid (0/2/4 dB x 0/5/10/20%)   [8 algorithms]
  patrol:  sigma x {patrol}  (0/2/4 dB)                [8 algorithms]
The sigma=0 rows with the shared algorithms reproduce the deterministic
campaigns byte-for-byte (same seeds, same code path) — a reproducibility
check that runs by construction.

Algorithms: astar_dist, gpsr, aodv, olsr, letolsr (NEW, Xie-2018-family
prediction-augmented OLSR), astar_etx, qlgr (inference-only, phase-8
merged tables), mpcare.

Outputs (data/processed/): phase9_shadow_diamond.csv/_summary.json/_pdr.png,
phase9_shadow_patrol.csv/_summary.json/_pdr.png.

Usage:
  py -3.13 scripts/run_phase9.py
  py -3.13 scripts/run_phase9.py --trials 3 --tag _smoke
  py -3.13 scripts/run_phase9.py --only diamond
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(ROOT / "src" / "qlgr_routing"))

import run_sim  # noqa: E402
import run_phase8  # noqa: E402  (reuse _flatten / stats / table loaders)
from stats_utils import (mannwhitneyu, holm_bonferroni, rank_biserial)  # noqa: E402

OUT = ROOT / "data" / "processed"
EVAL_SEED0 = 1000          # identical to the phase-4/8 evaluation seeds
ALGORITHMS = ["astar_dist", "gpsr", "aodv", "olsr", "letolsr",
              "astar_etx", "qlgr", "mpcare"]
ROW_KEYS = run_phase8.ROW_KEYS + ["shadow_sigma_db"]

SHADOW_LEVELS = (0.0, 2.0, 4.0)
LOSSES = (0.0, 0.05, 0.10, 0.20)

DIA_SPECS = [dict(label=f"s{int(s)}_loss_{v:.2f}", scenario="diamond",
                  injected_loss=v, shadow_sigma_db=s)
             for s in SHADOW_LEVELS for v in LOSSES]
PAT_SPECS = [dict(label=f"s{int(s)}_patrol", scenario="patrol",
                  speed=20.0, n_uav=10, shadow_sigma_db=s)
             for s in SHADOW_LEVELS]


def _flatten9(r: dict, condition: str, params: dict) -> dict:
    row = run_phase8._flatten(r, condition, params)
    row["shadow_sigma_db"] = params["shadow_sigma_db"]
    return row


def run_campaign9(name: str, conds: list[dict], duration: float,
                  table_name: str, tables: dict, trials: int, seed0: int,
                  tag: str = "") -> list[dict]:
    table = tables.get(table_name)
    if table is None:
        sys.exit(f"QLGR merged table '{table_name}' missing — run phase 8 first")
    rows: list[dict] = []
    t_start = time.perf_counter()
    for cond in conds:
        label = cond["label"]
        cond_kwargs = {k: v for k, v in cond.items() if k != "label"}
        params = {"scenario": cond_kwargs.get("scenario", ""),
                  "speed_mps": cond_kwargs.get("speed"),
                  "loss": cond_kwargs.get("injected_loss") or 0.0,
                  "shadow_sigma_db": cond_kwargs["shadow_sigma_db"]}
        for algo in ALGORITHMS:
            for trial in range(trials):
                seed = seed0 + trial
                r = run_sim.run(seed, ROOT / "data" / "raw",
                                duration=duration, write_csv=False,
                                algorithm=algo,
                                qlgr_opts={"learn": False, "seed": seed,
                                           "eps": 0.0, "table": table},
                                **cond_kwargs)
                rows.append(_flatten9(r, label, params))
            print(f"[{name}] {label:<16} {algo:<11} {trials} trials "
                  f"({time.perf_counter() - t_start:.0f}s elapsed)", flush=True)
    return rows


def write_outputs(name: str, rows: list[dict], conds: list[dict],
                  duration: float, trials: int, seed0: int,
                  tag: str = "") -> dict:
    """CSV + summary + the direct shadow-amplification benchmark."""
    import csv as _csv
    labels = [c["label"] for c in conds]
    with open(OUT / f"phase9_{name}{tag}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=ROW_KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    summary = {
        "experiment": f"phase9_{name}", "n_trials": trials,
        "seed0": seed0, "duration_s": duration,
        "algorithms": ALGORITHMS, "conditions": labels,
        "primary_metric": "chain_pdr",
        "primary_stats": run_phase8.condition_stats(rows, labels, ALGORITHMS,
                                                    "chain_pdr")}
    # ---- the direct hypothesis test: does the MP-CARE − GPSR margin grow
    # with shadow sigma? (paired per-sigma, Holm-corrected within sigma)
    b: dict = {}
    for lab in labels:
        gp = [r["chain_pdr"] for r in rows if r["condition"] == lab
              and r["algorithm"] == "gpsr"]
        mp = [r["chain_pdr"] for r in rows if r["condition"] == lab
              and r["algorithm"] == "mpcare"]
        if gp and mp:
            mw = mannwhitneyu(mp, gp)
            b[lab] = {
                "gpsr_mean": round(statistics.mean(gp), 4),
                "mpcare_mean": round(statistics.mean(mp), 4),
                "margin_pp": round(100 * (statistics.mean(mp)
                                          - statistics.mean(gp)), 2),
                "mw_p": round(mw["p"], 6) if mw["p"] is not None else None,
                "holm_p": None,
                "rank_biserial": rank_biserial(mp, gp)}
    # Holm across the within-sigma family of MP-CARE vs GPSR tests
    per_sigma: dict[float, list[str]] = {}
    for lab in labels:
        s = float(lab.replace("patrol", "").strip("s_").split("_")[0]
                  if "patrol" in lab else lab.split("_")[0].lstrip("s"))
        per_sigma.setdefault(s, []).append(lab)
    for s, labs in per_sigma.items():
        pvals, pairs = [], []
        for lab in labs:
            if lab in b and b[lab]["mw_p"] is not None:
                pvals.append(b[lab]["mw_p"])
                pairs.append(lab)
        adj = holm_bonferroni(pvals)
        for lab, p in zip(pairs, adj):
            b[lab]["holm_p"] = round(p, 6) if p is not None else None
        b[f"sigma_{int(s)}_margin_mean_pp"] = round(
            statistics.mean([b[c]["margin_pp"] for c in labs]), 2)
    summary["shadow_amplification"] = b
    with open(OUT / f"phase9_{name}{tag}_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # ---- figure: PDR vs sigma per algorithm on the headline loss level
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.0, 4.6), dpi=300)
        head_labs = [c["label"] for c in conds]
        xs = sorted({float(l.split("s", 1)[1].split("_")[0])
                     for l in head_labs})
        for algo in ALGORITHMS:
            ys = []
            for s in xs:
                labs_s = [l for l in head_labs
                          if float(l.split("s", 1)[1].split("_")[0]) == s]
                vals = [r["chain_pdr"] for r in rows
                        if r["algorithm"] == algo and r["condition"] in labs_s
                        and isinstance(r["chain_pdr"], (int, float))]
                ys.append(statistics.mean(vals) if vals else None)
            ax.plot(xs, ys, "o-", label=algo,
                    color=run_phase8.ALGO_COLOR.get(algo))
        ax.set_xlabel("shadow fading sigma (dB)")
        ax.set_ylabel("mean PDR")
        ax.set_title(f"phase9 {name}: PDR vs shadow sigma "
                     f"({'all loss levels pooled' if name == 'shadow_diamond' else 'patrol lanes'})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"phase9_{name}_pdr.png")
        plt.close(fig)
    except Exception as e:   # figures are optional; data + summary first
        print(f"[{name}] figure skipped: {e}", flush=True)
    print(f"[{name}] wrote phase9_{name}{tag}.csv/_summary.json", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--only", type=str, default="diamond,patrol")
    args = ap.parse_args()
    which = {w.strip() for w in args.only.split(",") if w.strip()}

    tables = run_phase8.load_saved_tables(OUT)
    t0 = time.perf_counter()
    if "diamond" in which:
        rows = run_campaign9("shadow_diamond", DIA_SPECS, 90.0, "diamond",
                             tables, args.trials, args.seed0, args.tag)
        write_outputs("shadow_diamond", rows, DIA_SPECS, 90.0, args.trials,
                      args.seed0, args.tag)
    if "patrol" in which:
        rows = run_campaign9("shadow_patrol", PAT_SPECS, 90.0, "dynamic",
                             tables, args.trials, args.seed0, args.tag)
        write_outputs("shadow_patrol", rows, PAT_SPECS, 90.0, args.trials,
                      args.seed0, args.tag)
    print(f"phase9 done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
