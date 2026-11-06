#!/usr/bin/env python3
"""Merge chunked DCF carrier-sense campaigns into one range artifact.

The patrol conditions are expensive, so they are run in chunks with tags such
as ``_cs200_p0``.  This script concatenates every
``dcf_campaign_cs<range>_*.csv`` chunk (deduplicating seed/algorithm/condition),
recomputes the condition summaries with the same procedure as
``run_dcf.py``, and writes the canonical range files:

  data/processed/dcf_campaign_cs<range>.csv
  data/processed/dcf_campaign_cs<range>_summary.json

Usage:
  py -3.13 scripts/merge_dcf_parts.py --cs-range 200
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))
sys.path.insert(0, str(ROOT / "src" / "logger"))
sys.path.insert(0, str(ROOT / "src" / "aodv_routing"))
sys.path.insert(0, str(ROOT / "src" / "olsr_routing"))
sys.path.insert(0, str(ROOT / "src" / "qlgr_routing"))

import run_dcf  # noqa: E402

OUT = ROOT / "data" / "processed"
INT_FIELDS = {"seed", "reroutes", "collisions", "frames_scheduled", "deferred",
              "bg_frames", "bc_frames", "bc_rx_attempts", "bc_rx_blocked",
              "bc_rx_ok", "bc_collided_frames"}
FLOAT_FIELDS = {"chain_pdr", "avg_latency_ms", "airtime_s", "utilization",
                "loss", "sigma_db", "bg_fps"}


def _cast(row: dict) -> dict:
    out = dict(row)
    for k in INT_FIELDS:
        if out.get(k) not in (None, ""):
            out[k] = int(float(out[k]))
    for k in FLOAT_FIELDS:
        if out.get(k) not in (None, ""):
            out[k] = float(out[k])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cs-range", type=float, required=True)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    args = ap.parse_args()

    cs = int(args.cs_range)
    parts = sorted(OUT.glob(f"dcf_campaign_cs{cs}_*.csv"))
    if not parts:
        raise SystemExit(f"no chunk CSVs found for cs={cs}")
    rows: list[dict] = []
    seen: set[tuple] = set()
    for p in parts:
        for r in csv.DictReader(p.open(encoding="utf-8")):
            key = (r["seed"], r["algorithm"], r["condition"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(_cast(r))
        print(f"  {p.name}: {len(seen)} unique rows so far", flush=True)

    tag = f"_cs{cs}"
    summary = run_dcf._summarise(rows, args.trials, args.seed0, tag,
                                 args.cs_range)
    print(f"merged {len(rows)} rows from {len(parts)} chunks -> "
          f"dcf_campaign_cs{cs}.csv")
    for lab, s in summary["conditions"].items():
        print(f"{lab:<28} margin_vs_gpsr={s['mpcare_minus_gpsr_pp']:+.1f}pp "
              f"rank={s['ranking'][:3]}")


if __name__ == "__main__":
    main()
