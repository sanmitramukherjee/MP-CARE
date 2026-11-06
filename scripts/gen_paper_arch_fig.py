#!/usr/bin/env python3
"""Generate an IEEE publication-grade monochrome architecture diagram for MP-CARE
(paper/figures/method_components.png).

Clean black-and-white vector aesthetics, high contrast, orthogonal feedback routing,
zero text-line collisions.
"""

from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "method_components.png"

# Grayscale / Monochromatic palette for IEEE publication
BOX_FC = "#ffffff"
BOX_EC = "#000000"
INPUT_FC = "#f0f0f0"
INPUT_EC = "#333333"
TEXT_COLOR = "#000000"
MUTED_TEXT = "#222222"


def draw_box(ax, xy, w, h, title, lines, fc=BOX_FC, ec=BOX_EC, ls="-", lw=1.2, is_title_box=True):
    x, y = xy
    # Main box
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.0,rounding_size=0.05",
                                fc=fc, ec=ec, lw=lw, linestyle=ls))
    if is_title_box:
        # Subtle header separator line inside box
        header_h = 0.35
        ax.plot([x, x + w], [y + h - header_h, y + h - header_h], color=ec, lw=0.8)
        # Header text
        ax.text(x + w / 2, y + h - header_h / 2, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=TEXT_COLOR)
        # Body text
        ax.text(x + w / 2, y + (h - header_h) / 2, "\n".join(lines), ha="center", va="center",
                fontsize=8.0, color=MUTED_TEXT, linespacing=1.25)
    else:
        # Body-only or small box
        ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
                fontsize=9.0, fontweight="bold", color=TEXT_COLOR)
        ax.text(x + w / 2, y + h * 0.32, "\n".join(lines), ha="center", va="center",
                fontsize=7.8, color=MUTED_TEXT, linespacing=1.2)


def draw_arrow(ax, p0, p1, color="#000000", style="-|>", lw=1.2, ls="solid", shrinkA=2, shrinkB=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=10,
                                 color=color, lw=lw, linestyle=ls,
                                 shrinkA=shrinkA, shrinkB=shrinkB))


def main() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(figsize=(8.5, 3.4), dpi=600)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    # Dimensions
    w_main, h_main = 2.15, 1.45
    w_in, h_in = 2.15, 0.95
    y_top = 1.95
    y_bot = 0.55

    x1, x2, x3, x4 = 0.35, 2.75, 5.15, 7.55

    # Top Row: Protocol Modules
    draw_box(ax, (x1, y_top), w_main, h_main, "Neighbor Table",
             ["10 Hz beacons $B_i$ (T$_b$=0.1 s)", "EWMA ETX ($\\alpha=0.3$)", "radial vel. $\\dot{d}$", "loss timeout 2.5 s"])

    draw_box(ax, (x2, y_top), w_main, h_main, "Link Predictor",
             ["$\\mathrm{ETX}_{pred}=\\mathrm{ETX}\\cdot f(\\dot{d})$",
              "$f(v)=1+\\beta\\tanh(\\gamma v)$", "$\\beta=1.0,\\ \\gamma=0.15$",
              "LET $=(R_{max}-d)/\\dot{d}$"])

    draw_box(ax, (x3, y_top), w_main, h_main, "Cost Engine",
             ["$C(p)=\\sum_k w_k\\hat{X}_k(p)$", "min-max over candidates",
              "2-hop expansion", "D, E, L, Q, H terms"])

    draw_box(ax, (x4, y_top), w_main, h_main, "Event Trigger",
             ["T1 PRR $<\\theta_{pr}$", "T2 LET $<\\tau_{min}$",
              "T3 margin $\\Delta$", "T4 relay loss (bypass)",
              "dwell $t_{dwell}=2$ s"])

    # Bottom Row: External Inputs
    draw_box(ax, (x1, y_bot), w_in, h_in, "Physical Channel",
             ["802.11b MAC (4 retries)", "log-distance PRR/ETX"], fc=INPUT_FC, ec=INPUT_EC, ls="--")

    draw_box(ax, (x2, y_bot), w_in, h_in, "Beacons In",
             ["1-hop payloads", "(can be lost)"], fc=INPUT_FC, ec=INPUT_EC, ls="--")

    draw_box(ax, (x3, y_bot), w_in, h_in, "Flight / Mission",
             ["waypoints, destinations", "(T5 trigger source)"], fc=INPUT_FC, ec=INPUT_EC, ls="--")

    # Inter-module data flow (Top row horizontal arrows)
    draw_arrow(ax, (x1 + w_main, y_top + h_main / 2), (x2, y_top + h_main / 2))
    draw_arrow(ax, (x2 + w_main, y_top + h_main / 2), (x3, y_top + h_main / 2))
    draw_arrow(ax, (x3 + w_main, y_top + h_main / 2), (x4, y_top + h_main / 2))

    # Input arrows (Bottom -> Top)
    draw_arrow(ax, (x1 + w_in / 2, y_bot + h_in), (x1 + w_main / 2, y_top), color="#444444")
    draw_arrow(ax, (x2 + w_in / 2, y_bot + h_in), (x2 + w_main / 2, y_top), color="#444444")
    draw_arrow(ax, (x3 + w_in / 2, y_bot + h_in), (x4 + w_main / 2, y_top), color="#444444")

    # Feedback Loop: Clean orthogonal routing around the bottom
    y_fb = 0.18
    draw_arrow(ax, (x4 + w_main / 2, y_top), (x4 + w_main / 2, y_fb), color="#000000", ls=":", shrinkA=0, shrinkB=0)
    draw_arrow(ax, (x4 + w_main / 2, y_fb), (x1 + w_main / 2, y_fb), color="#000000", ls=":", shrinkA=0, shrinkB=0)
    draw_arrow(ax, (x1 + w_main / 2, y_fb), (x1 + w_main / 2, y_bot), color="#000000", ls=":", shrinkA=0, shrinkB=2)

    # Feedback Loop Label (neat centered box at bottom)
    ax.text(x2 + w_main / 2 + 0.1, y_fb + 0.08,
            "Re-route feedback (delivery outcomes $\\to$ T1 evidence)",
            ha="center", va="bottom", fontsize=7.5, fontweight="bold", style="italic",
            color="#111111",
            bbox=dict(boxstyle="square,pad=0.25", fc="#ffffff", ec="#666666", lw=0.6))

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote clean black-and-white architecture figure to {OUT}")


if __name__ == "__main__":
    main()
