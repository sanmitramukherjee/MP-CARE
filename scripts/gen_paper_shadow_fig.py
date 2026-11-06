#!/usr/bin/env python3
"""Paper figure G9: shadow-fading sensitivity (phase 9).

Two panels from the cached phase9 summaries:
  (a) patrol lanes: mean PDR vs shadow sigma per algorithm;
  (b) diamond: MP-CARE - GPSR margin (pp) vs sigma at 0/5/10/20% injected loss.
Output: paper/figures/phase9_shadow_g91.png at 600 dpi.

Usage: py -3.13 scripts/gen_paper_shadow_fig.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_phase8  # noqa: E402  (ALGO_COLOR palette)

OUT = ROOT / "data" / "processed"
FIG = ROOT / "paper" / "figures" / "phase9_shadow_g91.png"

SMOOTH = {"astar_dist", "gpsr", "aodv", "olsr", "letolsr", "qlgr",
          "astar_etx", "mpcare"}
LABELS = {"astar_dist": "A* dist", "gpsr": "GPSR", "aodv": "AODV",
          "olsr": "OLSR", "letolsr": "LET-OLSR", "qlgr": "QLGR",
          "astar_etx": "ETX oracle", "mpcare": "MP-CARE"}


def patrol_panel(ax):
    d = json.loads((OUT / "phase9_shadow_patrol_summary.json")
                   .read_text(encoding="utf-8"))
    conds = sorted(d["primary_stats"], key=lambda c: float(c[1:].split("_")[0]))
    sigmas = [float(c[1:].split("_")[0]) for c in conds]
    for algo in ("mpcare", "astar_etx", "letolsr", "olsr", "gpsr", "qlgr",
                 "astar_dist", "aodv"):
        ys = [d["primary_stats"][c]["per_algo"][algo]["mean"] for c in conds]
        ax.plot(sigmas, ys, "o-", label=LABELS[algo], linewidth=1.8,
                markersize=4, color=run_phase8.ALGO_COLOR.get(algo))
    ax.set_xlabel("shadow fading $\\sigma$ (dB)")
    ax.set_ylabel("mean PDR")
    ax.set_title("(a) Patrol lanes (10 UAV, 20 m/s)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="lower left")


def diamond_panel(ax):
    d = json.loads((OUT / "phase9_shadow_diamond_summary.json")
                   .read_text(encoding="utf-8"))
    res: dict[float, dict[float, float]] = {}
    for cond, st in d["primary_stats"].items():
        sigma = float(cond.split("_")[0][1:])
        loss = float(cond.split("loss_")[1])
        gp = st["per_algo"]["gpsr"]["mean"]
        mp = st["per_algo"]["mpcare"]["mean"]
        res.setdefault(sigma, {})[loss] = 100.0 * (mp - gp)
    sigmas = sorted(res)
    for loss in (0.0, 0.05, 0.10, 0.20):
        ys = [res[s].get(loss) for s in sigmas]
        ax.plot(sigmas, ys, "o-", linewidth=1.8, markersize=4,
                label=f"{int(loss*100)}% injected loss")
    ax.axhline(0.0, color="k", linewidth=0.8, linestyle=":")
    ax.set_xlabel("shadow fading $\\sigma$ (dB)")
    ax.set_ylabel("MP-CARE $-$ GPSR margin (pp)")
    ax.set_title("(b) Diamond loss sweep, margin vs $\\sigma$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=600)
    patrol_panel(axes[0])
    diamond_panel(axes[1])
    fig.tight_layout()
    fig.savefig(FIG)
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
