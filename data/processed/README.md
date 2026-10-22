# data/processed/

Aggregated experiment outputs — per-condition means, confidence intervals, and
effect sizes — produced by `scripts/run_experiment.py` and the analysis scripts
in `scripts/analysis/`. These files are the direct inputs to the paper figures
and tables in `paper/figures/`.

---

## File Registry

### Phase 2–4 — Baseline Benchmarks

| File(s) | Description |
|---|---|
| `baseline_*`, `exp2*`–`exp6*`, `envelope_*` | Raw aggregated results from early campaign phases. |
| `phase4_benchmarks.json` | Canonical Phase 4 benchmark set: 30 seeds × 4 algorithms × 5 link-loss levels (0–40%). Source for the +14.6 pp PDR gain at 10% loss reported in §VII of the paper. |

### Phase 5 — Extension A & B

| File(s) | Description |
|---|---|
| `phase5_orca_trials.csv`, `_summary.json`, `.png` | Phase 5 Extension A (ORCA + connectivity penalty): 2 scenarios × 2 variants × 10 seeds (2000–2009). |
| `phase5_cbba_trials.csv`, `_summary.json`, `.png` | Phase 5 Extension B (CBBA + comm-cost bid): 7 loss points (0–60%) × 2 variants × 10 seeds. |

### Phase 6 — Scalability Campaign

Three successive runs at the same 30 seeds (3000–3029). Only v3 is cited in the
paper; v1 and v2 are retained as a percolation-edge provenance study.

| File(s) | Description |
|---|---|
| `phase6_trials.csv`, `_summary.json`, `_g61…g66.png` | v1 geometry: constant-area boxes. N = 5→50 × 4 algorithms × 30 seeds (720 rows). Retained as the percolation-edge study (§IX). |
| `phase6v2_trials.csv`, `_summary.json`, `_g61…g66.png` | v2 geometry: frozen void-recovery router applied. PDR +6…+11 pp over v1. Superseded by v3. |
| `phase6v3_trials.csv`, `_summary.json`, `_g61…g66.png` | **Canonical Phase 6 set** (`--density v3` boxes; measured ⟨k⟩ ≈ 7.7–8.8, probe-validated on all 30 seeds). Zero connectivity alarms. MP-CARE PDR ranked first at every N ≥ 10 (N = 50: PDR = 0.939, Holm-corrected p ≤ 0.0002). Source for paper §VIII. |

### Phase 7 — Ablation Study

| File(s) | Description |
|---|---|
| `phase7_ablation_trials.csv`, `_summary.json`, `_g71_ablation.png` | 8 ablation variants (A0 full + A1–A7) × 30 seeds (5000–5029) × 2 conditions. Mann-Whitney + Holm vs. full MP-CARE. Source for the A7 (static routing) −15.5 pp PDR result (Holm p = 0.0015) in §VIII. |
| `phase7_weights_trials.csv`, `_summary.json`, `_g72_pareto.png` | Weight sensitivity sweep: 24 configs (w_e, w_t, w_q, w_h each ∈ {0, 0.25, 0.5, 1, 2, 4}; anchor w_d = 1.0) × 30 seeds × 2 conditions. Pareto plot in paper §VIII. |

---

## Reproducibility Notes

- Per-trial rows carry their random seed, enabling exact reconstruction of any
  reported statistic.
- Phase 5 trial CSVs are bit-identical across Windows and WSL runs.
- Phase 6 v3 parity: 12 rows × 17 behavioral columns are bit-identical
  Windows ↔ WSL. Only the machine-dependent `cpu_ms_per_node_tick` and
  `run_wall_s` columns differ by nature (wall-clock performance counter).
- Mann-Whitney U p-values in summary JSONs may differ in the last ULP between
  platforms due to floating-point rounding in the normal approximation; any
  p-value within 10× of the significance threshold should be re-verified with a
  standard statistics package before citing.
