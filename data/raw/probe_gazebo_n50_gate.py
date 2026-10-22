"""Gazebo N=50 gate validation (machine checks, evidence only).

Validates data/raw/phase6_dev/gazebo_n50.csv:
  1. header = the 27-field logger schema
  2. trial_id = gazebo_n50, algorithm = mpcare
  3. all 50 UAVs log rows; 10 rows/s per node (logger contract)
  4. span of the run
  5. LIVE ROUTING DEGREE from the neighbor-list columns (the ROS-side
     adapter's 1-hop view) vs the campaign reference (seed 3001 t=0: 8.00)
  6. every listed neighbor carries a finite ETX estimate
"""
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "raw" / "phase6_dev" / "gazebo_n50.csv"

CAMP_REF_DEG = 8.00   # gen_gazebo_positions.py pre-flight print (seed 3001)


def main() -> None:
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    ok = True

    def check(name: str, cond: bool, detail: str) -> None:
        nonlocal ok
        print(f"  [{'OK' if cond else 'FAIL'}] {name}: {detail}")
        ok &= cond

    header = list(rows[0].keys())
    check("header fields", len(header) == 27, f"{len(header)} fields")
    check("trial_id", rows[0].get("trial_id") == "gazebo_n50",
          rows[0].get("trial_id", "?"))
    algos = {r.get("algorithm") for r in rows}
    check("algorithm", algos == {"mpcare"}, str(algos))

    per_node = defaultdict(list)
    for r in rows:
        per_node[int(r["uav_id"])].append(r)
    check("all 50 UAVs log", sorted(per_node) == list(range(50)),
          f"{len(per_node)} nodes")

    t0 = min(float(r["timestamp_ms"]) for r in rows) / 1000.0
    t1 = max(float(r["timestamp_ms"]) for r in rows) / 1000.0
    span = t1 - t0
    n = len(per_node)
    rate = len(rows) / n / span
    check("logger rate ~10 Hz/node", 8.0 <= rate <= 12.0,
          f"{rate:.1f} rows/node/s over {span:.1f} s")

    # live degree from the neighbor-list column ("nbrs" = "1;5;7;..." style)
    degs = []
    etx_bad = 0
    nbr_field = "neighbors"
    etx_field = "neighbor_etx"
    for r in rows:
        ids = r.get(nbr_field, "")
        if not ids:
            continue
        nbrs = [int(x) for x in ids.split(";") if x != ""]
        degs.append(len(nbrs))
        if etx_field:
            vals = [v for v in r.get(etx_field, "").split(";") if v != ""]
            if len(vals) == len(nbrs):
                for v in vals:
                    try:
                        if math.isfinite(float(v)) and float(v) > 0:
                            continue
                    except ValueError:
                        pass
                    etx_bad += 1
    mean_deg = statistics.mean(degs) if degs else float("nan")
    check("live mean degree vs campaign ref",
          abs(mean_deg - CAMP_REF_DEG) < 1.5,
          f"{mean_deg:.2f} (campaign static-window ref {CAMP_REF_DEG}; "
          f"spread {min(degs)}-{max(degs)})" if degs else "no data")
    check("neighbor ETX finite", etx_bad == 0, f"{etx_bad} bad entries")

    print("\nGATE:", "PASS" if ok else "FAIL",
          f"({len(rows)} rows, {n} UAVs, {span:.0f} s)")


if __name__ == "__main__":
    sys.exit(main())
