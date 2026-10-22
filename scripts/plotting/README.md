# scripts/plotting/

Matplotlib scripts that produce every figure in the paper from the aggregated
data in `data/processed/`.

---

## Role in the Pipeline

```
data/processed/   →   scripts/plotting/   →   paper/figures/
(summary JSONs,        (figure scripts)        (PNG files used
 benchmark tables)                              in the paper)
```

## Figure Map

| Script | Output figure | Paper section |
|---|---|---|
| `plot_phase4.py` | PDR vs. link-loss benchmark curves | §VII |
| `plot_phase5_orca.py` | ORCA extension comparison | §VII |
| `plot_phase5_cbba.py` | CBBA extension sweep | §VII |
| `plot_phase6.py` | Scalability curves (N = 5→50) | §VIII |
| `plot_phase7_ablation.py` | Ablation bar chart (A0–A7) | §VIII |
| `plot_phase7_weights.py` | Weight-sensitivity Pareto plot | §VIII |

## Style

All figures follow the IEEE Transactions on Vehicular Technology style guide:
- Consistent line styles and markers per algorithm across all plots.
- Error bars show 95% confidence intervals.
- Raster output at 300 DPI (PNG) for LaTeX inclusion.
