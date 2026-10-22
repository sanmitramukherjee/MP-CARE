#!/usr/bin/env python3
"""Re-render the paper's figure PNGs at 600 DPI from the CACHED campaign
outputs in data/processed/ (CSVs + summary JSONs), without rerunning any
simulation trial. Idempotent: the graph functions are the exact ones the
campaign scripts use, so re-renders are pixel-equivalent to a full rerun
apart from resolution.

Usage:
  python scripts/rerender_paper_figs.py

Then copy the regenerated PNGs listed in COPY_TO_PAPER into paper/figures/.
"""
from __future__ import annotations

import csv
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_paper_arch_fig  # noqa: E402
import run_phase5  # noqa: E402
import run_phase6  # noqa: E402
import run_phase7  # noqa: E402
import run_sweeps  # noqa: E402

P = ROOT / "data" / "processed"
FIGDIR = ROOT / "paper" / "figures"

# paper/figures/*.png that come from data/processed/
COPY_TO_PAPER = [
    "envelope_g47_benefit_envelope.png",
    "exp2b_loss_diamond_g41_pdr_vs_loss.png",
    "exp5_mobility_g45_pdr_vs_speed.png",
    "exp6_failure_g46_recovery_box.png",
    "phase5_cbba.png",
    "phase5_orca.png",
    "phase6v3_g61_cpu_vs_n.png",
    "phase6v3_g63_pdr_vs_n.png",
    "phase7_ablation_g71_ablation.png",
    "phase7_weights_g72_pareto.png",
]


def coerce(v):
    """CSV cell -> typed value: '' -> None, numeric strings -> float/int,
    everything else stays a string (matches the in-memory campaign rows)."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return v
    try:
        f = float(v)
    except ValueError:
        return v
    return int(f) if f.is_integer() and re.fullmatch(r"-?\d+", v.strip()) else f


def load_rows(name: str) -> list[dict]:
    with open(P / f"{name}.csv", newline="", encoding="utf-8") as f:
        return [{k: coerce(v) for k, v in row.items()} for row in csv.DictReader(f)]


def load_summary(name: str) -> dict:
    with open(P / f"{name}_summary.json", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # ---- phase 4 sweeps (g41/g45/g46) -------------------------------------
    for name, fn in [
        ("exp2b_loss_diamond", run_sweeps.graph_41_44),
        ("exp5_mobility", run_sweeps.graph_45),
        ("exp6_failure", run_sweeps.graph_46),
    ]:
        summ = load_summary(name)
        fn(load_rows(name), name, summ["conditions"], list(summ["algorithms"]))
        print(f"[ok] {name} graphs")

    # ---- envelope (g47): loss/speed axes parsed from condition labels -----
    env_rows = load_rows("envelope")
    env_conds = load_summary("envelope")["conditions"]
    parsed = [re.match(r"L([\d.]+)_S(\d+)", c) for c in env_conds]
    losses = sorted({float(m.group(1)) for m in parsed if m})
    speeds = sorted({float(m.group(2)) for m in parsed if m})
    run_sweeps.graph_47(env_rows, "envelope", losses, speeds)
    print("[ok] envelope g47")

    # ---- phase 6 scale sweep (g61..g66) -----------------------------------
    p6_rows = load_rows("phase6v3_trials")
    points = sorted({int(r["N"]) for r in p6_rows if r["N"] is not None})
    run_phase6.graphs(p6_rows, points, "phase6v3")
    print("[ok] phase6v3 graphs")

    # ---- phase 7 ablation + pareto ----------------------------------------
    run_phase7.ablation_graph(load_rows("phase7_ablation_trials"),
                              "phase7_ablation")
    run_phase7.pareto_graph(load_summary("phase7_weights")["per_config"],
                            "phase7_weights")
    print("[ok] phase7 ablation/pareto graphs")

    # ---- phase 5 extensions (from summary JSONs only) ---------------------
    run_phase5.plot_orca(load_summary("phase5_orca"))
    run_phase5.plot_cbba(load_summary("phase5_cbba"))
    print("[ok] phase5 orca/cbba charts")

    # ---- architecture figure (writes straight into paper/figures) ---------
    gen_paper_arch_fig.main()
    print("[ok] architecture figure")

    # ---- copy campaign PNGs into paper/figures/ ---------------------------
    for fname in COPY_TO_PAPER:
        src = P / fname
        if not src.exists():
            print(f"[MISSING] {src}")
            continue
        shutil.copy2(src, FIGDIR / fname)
        print(f"[copy] {fname} -> paper/figures/")

    print("\nDONE. Verify DPI with: python scripts/elsevier_audit.py")


if __name__ == "__main__":
    main()
