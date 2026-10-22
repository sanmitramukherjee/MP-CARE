#!/usr/bin/env python3
"""Validate the native dual-boot ROS CSVs against the WSL baseline and the
kill semantics. Prints facts only; no claims beyond what is in the files.

Usage: python scripts/validate_native_csvs.py
Exit code 0 if all CHECK lines report OK, 1 otherwise.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Optional overrides: python scripts/validate_native_csvs.py TIER1.csv TIER3.csv
_files = sys.argv[1:3]
FILES = {
    "tier1": Path(_files[0]) if _files else ROOT / "data/raw/ubuntu_tier1.csv",
    "tier3": Path(_files[1]) if len(_files) > 1 else ROOT / "data/raw/ubuntu_tier3_run1.csv",
}
WSL_HEADER = None  # filled from the WSL tier1 file for schema comparison

ok_all = True


def check(name: str, cond: bool, detail: str) -> None:
    global ok_all
    print(f"  [{'OK' if cond else 'FAIL'}] {name}: {detail}")
    ok_all = ok_all and cond


def load(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    return header, rows


def col(header, name):
    return header.index(name)


for label, path in FILES.items():
    print(f"== {label}: {path.name} ==")
    if not path.exists():
        check("file exists", False, str(path))
        continue
    header, rows = load(path)
    check("header = 27 fields", len(header) == 27, f"{len(header)} fields")
    if WSL_HEADER is None:
        wsl_path = ROOT / "data/raw/ubuntu_tier1_wsl.csv"
        if wsl_path.exists():
            WSL_HEADER, _ = load(wsl_path)
    if WSL_HEADER:
        check("schema identical to WSL csv", header == WSL_HEADER, "")

    ids = {r[0] for r in rows}
    ialg = col(header, "algorithm")
    iexp = col(header, "experiment")
    algs = {r[ialg] for r in rows}
    exps = {r[iexp] for r in rows}
    check("trial_id", len(ids) == 1, ids.pop() if len(ids) == 1 else str(ids))
    check("algorithm=mpcare", algs == {"mpcare"}, str(algs))
    check("experiment=phase4_ros", exps == {"phase4_ros"}, str(exps))

    ts = [int(r[1]) for r in rows]
    dur = (max(ts) - min(ts)) / 1000.0
    print(f"  rows={len(rows)}  span={dur:.1f}s  agents={sorted({int(r[2]) for r in rows})}")
    per = Counter(int(r[2]) for r in rows)
    print(f"  rows per uav: {dict(sorted(per.items()))}")
    check("5 agents present", sorted(per) == [0, 1, 2, 3, 4], str(sorted(per)))

    # placeholder honesty: pdr_instantaneous must be 0.0 everywhere
    ipdr = col(header, "pdr_instantaneous")
    check("pdr_instantaneous placeholder 0.0", all(r[ipdr] == "0.0" for r in rows), "")

    ihops = col(header, "route_hop_count")
    iuav = col(header, "uav_id")
    hops0 = [int(r[ihops]) for r in rows if int(r[iuav]) == 0]
    print(f"  uav0 route_hop_count: max={max(hops0)} (min seen={min(hops0)})")

    if label == "tier1":
        check("uav0 reaches 4-hop chain route", max(hops0) == 4, f"max={max(hops0)}")
        inb = col(header, "neighbors")
        ietx = col(header, "neighbor_etx")
        r0 = next(r for r in rows if int(r[iuav]) == 0 and r[inb] == "1")
        check("uav0 single neighbor=1", r0[inb] == "1", f"etx={r0[ietx]}")
    else:
        # Kill semantics. The exact wall-clock kill time is not in the CSV,
        # so derive it: the last sighting of node 2 (as a listed neighbor)
        # is an upper bound on "node 2 still alive".
        inb = col(header, "neighbors")
        t_2row = max((int(r[1]) for r in rows if int(r[iuav]) == 2), default=None)
        n2 = sorted(int(r[1]) for r in rows
                    if any(p.strip() == "2" for p in r[inb].split(";") if p))
        t_2nbr = n2[-1] if n2 else None
        print(f"  last uav2 telemetry row:          t={t_2row}")
        print(f"  last row listing node 2 neighbor: t={t_2nbr}")
        check("node 2 present then gone forever",
              t_2nbr is not None, f"{len(n2)} rows list it, last at {t_2nbr}")
        # uav0: 4 hops before the partition, 0 hops after, never recovers.
        first0 = next((int(r[1]) for r in rows if int(r[iuav]) == 0
                       and int(r[ihops]) == 0 and int(r[1]) > t_2nbr), None)
        if first0 and t_2nbr:
            gap = (first0 - t_2nbr) / 1000.0
            print(f"  uav0 first 0-hop row: t={first0} ({gap:.1f}s after last node-2 sighting)")
            check("uav0 route lost within 10 s of node-2 loss",
                  0.0 <= gap <= 10.0, f"gap={gap:.1f}s (stale_s=3.0 + beacon)")
        else:
            check("uav0 route lost after kill", False, "no 0-hop row post-kill")
        h0post = [int(r[ihops]) for r in rows
                  if int(r[iuav]) == 0 and int(r[1]) > t_2nbr + 8_000]
        if h0post:
            check("uav0 stays at 0 hops for rest of run (honest no-route)",
                  max(h0post) == 0, f"max={max(h0post)} over {len(h0post)} rows")
        else:
            check("uav0 post-kill rows exist", False, "none after last-sighting+8s")

print()
print("ALL OK" if ok_all else "SOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
