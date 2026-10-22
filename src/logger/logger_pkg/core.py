"""Swarm metrics logger (implementation_plan.md section 3.4).

Pure-Python core + a thin ROS 2 wrapper (logger_node.py). Writes one CSV per
trial with EXACTLY the section 3.4 schema, in column order. Rows are appended
per tick per UAV (>= 10 Hz per UAV required by section 3.5).
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

SCHEMA = [
    "trial_id", "timestamp_ms", "uav_id", "position_x", "position_y", "position_z",
    "velocity_x", "velocity_y", "velocity_z", "target_waypoint_x", "target_waypoint_y",
    "target_waypoint_z", "current_route_cost", "route_hop_count", "neighbors",
    "neighbor_etx", "neighbor_let", "pdr_instantaneous", "latency_avg_ms",
    "throughput_kbps", "queue_depth", "reroute_event", "reroute_trigger",
    "min_separation_m", "cpu_time_us", "algorithm", "experiment",
]


def default_trial_id() -> str:
    return datetime.now().strftime("trial_%Y%m%d_%H%M%S")


class TrialLogger:
    """CSV writer for one trial (section 3.4 schema, append-only)."""

    def __init__(self, out_dir: Path | str, trial_id: str | None = None):
        self.trial_id = trial_id or default_trial_id()
        self.path = Path(out_dir) / f"{self.trial_id}.csv"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="", encoding="utf-8")
        self._w = csv.writer(self._fh)
        self._w.writerow(SCHEMA)
        self.rows = 0

    def log(self, **kw):
        row = {k: kw.get(k, "") for k in SCHEMA}
        row["trial_id"] = self.trial_id
        row["neighbors"] = ";".join(str(n) for n in kw.get("neighbors", []))
        row["neighbor_etx"] = ";".join(
            "inf" if e == float("inf") else f"{e:.3f}" for e in kw.get("neighbor_etx", []))
        row["neighbor_let"] = ";".join(
            "inf" if e == float("inf") else f"{e:.1f}" for e in kw.get("neighbor_let", []))
        row["reroute_event"] = int(bool(kw.get("reroute_event", False)))
        self._w.writerow([row[k] for k in SCHEMA])
        self.rows += 1

    def close(self):
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def verify_schema(csv_path: Path | str, min_rows: int = 0) -> dict:
    """Section 3.5 check: header matches the schema exactly; row rate reported."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        n = sum(1 for _ in r)
    return {
        "schema_ok": header == SCHEMA,
        "rows": n,
        "rows_ok": n >= min_rows,
    }
