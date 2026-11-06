#!/usr/bin/env python3
"""Summarize the instrumented broadcast-loss check at 250 m and 200 m.

Reads the diamond-only reruns written by
``run_dcf.py --cs-range <cs> --tag _bc<cs>_d`` and reports, per condition and
algorithm, the PDR, broadcast frames per trial, the share of receiver-link
broadcast receptions blocked by a concurrent transmitter, the share of
broadcast frames blocked at one or more receivers, re-routes and unicast
collisions.

Usage:
  py -3.13 scripts/analyze_broadcast_loss.py
Outputs:
  data/processed/dcf_broadcast_check.csv
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
RANGES = (250, 200)


def main() -> None:
    detail: list[dict] = []
    for cs in RANGES:
        path = OUT / f"dcf_campaign_bc{cs}_d.csv"
        rows = list(csv.DictReader(path.open(encoding="utf-8")))
        for lab in sorted({r["condition"] for r in rows}):
            for algo in sorted({r["algorithm"] for r in rows
                                if r["condition"] == lab}):
                sub = [r for r in rows
                       if r["condition"] == lab and r["algorithm"] == algo]

                def mean(key: str) -> float:
                    return statistics.mean(float(r[key] or 0) for r in sub)

                att = mean("bc_rx_attempts")
                bcf = mean("bc_frames")
                detail.append({
                    "cs_range_m": cs,
                    "condition": lab,
                    "algorithm": algo,
                    "pdr": round(mean("chain_pdr"), 4),
                    "bc_frames": round(bcf, 1),
                    "bc_rx_attempts": round(att, 1),
                    "bc_rx_blocked": round(mean("bc_rx_blocked"), 1),
                    "block_rate_pct": round(100 * mean("bc_rx_blocked") / att, 2)
                    if att else 0.0,
                    "frames_blocked_pct": round(
                        100 * mean("bc_collided_frames") / bcf, 2) if bcf else 0.0,
                    "reroutes": round(mean("reroutes"), 1),
                    "collisions": round(mean("collisions"), 1),
                })

    with (OUT / "dcf_broadcast_check.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)

    print(f"{'cs':>4} {'condition':<24} {'algo':<9} {'PDR':>6} "
          f"{'bc/trial':>9} {'block%':>7} {'frames%':>8}")
    for d in detail:
        if d["condition"] != "diamond_dcf_bg0":
            continue
        print(f"{d['cs_range_m']:>4} {d['condition']:<24} {d['algorithm']:<9} "
              f"{d['pdr']:>6.3f} {d['bc_frames']:>9.0f} "
              f"{d['block_rate_pct']:>7.2f} {d['frames_blocked_pct']:>8.1f}")


if __name__ == "__main__":
    main()
