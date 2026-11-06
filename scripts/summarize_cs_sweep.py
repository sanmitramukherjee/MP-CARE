#!/usr/bin/env python3
"""Consolidate the carrier-sense sweep into one artifact.

Reads the per-trial CSVs and per-condition summaries written by
`scripts/run_dcf.py` at each carrier-sense range and reports, per range:

  * collision rate (collisions / scheduled frames) overall and per scene
  * the MP-CARE--GPSR margin range across the 14 conditions, split by scene
  * MP-CARE's rank positions and whether any protocol beats MP-CARE
    significantly (Holm p < 0.05)
  * MP-CARE's PDR change against the same condition at 600 m
  * per-condition detail for the break-point scan

The 600 m row uses the original campaign files (dcf_campaign.*); the other
ranges use the `_cs<range>` tag.  Ranges below the 300 m decode range are a
stress test of the DCF abstraction, not a physically valid radio setting;
they are kept in the same artifact but flagged by the paper.

Usage:
  py -3.13 scripts/summarize_cs_sweep.py
Outputs:
  data/processed/dcf_cs_sweep.csv
  data/processed/dcf_cs_sweep_summary.json
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RANGES = (600, 450, 400, 375, 350, 325, 300, 250, 200, 150)
ALPHA = 0.05


def _paths(cs: int) -> tuple[Path, Path]:
    tag = "" if cs == 600 else f"_cs{cs}"
    return OUT / f"dcf_campaign{tag}.csv", OUT / f"dcf_campaign{tag}_summary.json"


def _collision_rate(rows: list[dict], scene: str | None = None) -> float:
    sel = rows if scene is None else [r for r in rows if r["scenario"] == scene]
    frames = sum(int(float(r["frames_scheduled"])) for r in sel)
    hits = sum(int(float(r["collisions"])) for r in sel)
    return 100.0 * hits / frames if frames else 0.0


def _mean_per_trial(rows: list[dict], algo: str, field: str) -> float:
    vals = [float(r[field]) for r in rows
            if r["algorithm"] == algo and r[field] not in (None, "")]
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    detail: list[dict] = []
    sweep: list[dict] = []
    base_pdr: dict[str, float] = {}

    for cs in RANGES:
        csv_path, json_path = _paths(cs)
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        summ = json.loads(json_path.read_text(encoding="utf-8"))
        conds = summ["conditions"]
        if cs == 600:
            base_pdr = {lab: c["per_algorithm"]["mpcare"]["mean"]
                        for lab, c in conds.items()}

        margins, ranks, above = [], [], []
        diamond_margins, patrol_margins, deltas = [], [], []
        weakest_p = 0.0
        for label, cond in conds.items():
            per = cond["per_algorithm"]
            mp = per["mpcare"]["mean"]
            rank = cond["ranking"].index("mpcare") + 1
            margin = cond["mpcare_minus_gpsr_pp"]
            margins.append(margin)
            ranks.append(rank)
            if "patrol" in label:
                patrol_margins.append(margin)
            else:
                diamond_margins.append(margin)
            deltas.append(100.0 * (mp - base_pdr.get(label, mp)))
            gp = per["gpsr"].get("holm_p_vs_mpcare")
            weakest_p = max(weakest_p, gp if gp is not None else 0.0)
            winners = []
            for a in per:
                if a == "mpcare":
                    continue
                p = per[a].get("holm_p_vs_mpcare")
                if p is not None and p < ALPHA and per[a]["mean"] > mp:
                    winners.append(a)
            above.extend(winners)
            detail.append({
                "cs_range_m": cs, "condition": label,
                "mpcare_pdr": round(mp, 4),
                "gpsr_pdr": round(per["gpsr"]["mean"], 4),
                "margin_pp": margin,
                "mpcare_delta_vs_600_pp": round(100.0 * (mp - base_pdr.get(label, mp)), 2),
                "gpsr_holm_p": per["gpsr"].get("holm_p_vs_mpcare"),
                "mpcare_rank": rank,
                "ranked_first": cond["ranking"][0],
                "significantly_above_mpcare": "|".join(winners),
                "collisions": sum(int(float(r["collisions"]))
                                  for r in rows if r["condition"] == label),
                "frames_scheduled": sum(int(float(r["frames_scheduled"]))
                                        for r in rows if r["condition"] == label),
            })

        rank_counts = Counter(ranks)
        sweep.append({
            "cs_range_m": cs,
            "collision_rate_pct": round(_collision_rate(rows), 2),
            "collision_rate_diamond_pct": round(_collision_rate(rows, "diamond"), 2),
            "collision_rate_patrol_pct": round(_collision_rate(rows, "patrol"), 2),
            "margin_min_pp": min(margins),
            "margin_max_pp": max(margins),
            "margin_diamond_min_pp": min(diamond_margins),
            "margin_diamond_max_pp": max(diamond_margins),
            "margin_patrol_min_pp": min(patrol_margins),
            "margin_patrol_max_pp": max(patrol_margins),
            "mpcare_delta_vs_600_min_pp": round(min(deltas), 2),
            "mpcare_delta_vs_600_max_pp": round(max(deltas), 2),
            "mpcare_ranks": sorted(rank_counts),
            "rank_counts": {str(k): rank_counts[k] for k in sorted(rank_counts)},
            "negative_margin_cells": sum(1 for m in margins if m <= 0),
            "gpsr_holm_p_worst": weakest_p,
            "conditions_with_significant_winner": len(set(
                d["condition"] for d in detail
                if d["cs_range_m"] == cs and d["significantly_above_mpcare"])),
            "significant_winners": sorted(set(above)),
            "mpcare_frames_per_trial": round(
                _mean_per_trial(rows, "mpcare", "frames_scheduled"), 1),
            "gpsr_frames_per_trial": round(
                _mean_per_trial(rows, "gpsr", "frames_scheduled"), 1),
            "mpcare_reroutes_mean": round(
                _mean_per_trial(rows, "mpcare", "reroutes"), 1),
        })

    with (OUT / "dcf_cs_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)
    (OUT / "dcf_cs_sweep_summary.json").write_text(
        json.dumps({"ranges": sweep}, indent=2), encoding="utf-8")

    print(f"{'cs':>4} {'coll%':>7} {'dia%':>7} {'pat%':>7} "
          f"{'margin pp':>16} {'ranks':>18}  winners")
    for s in sweep:
        ranks = ",".join(f"{k}x{v}" for k, v in s["rank_counts"].items())
        print(f"{s['cs_range_m']:>4} {s['collision_rate_pct']:>7.2f} "
              f"{s['collision_rate_diamond_pct']:>7.2f} "
              f"{s['collision_rate_patrol_pct']:>7.2f} "
              f"{s['margin_min_pp']:>+7.1f}..{s['margin_max_pp']:<+6.1f} "
              f"{ranks:>18}  {'|'.join(s['significant_winners']) or '-'}")


if __name__ == "__main__":
    main()
