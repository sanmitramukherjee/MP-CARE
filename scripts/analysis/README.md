# scripts/analysis/

Statistical analysis scripts that reduce per-trial simulation output to the
summary statistics reported in the paper.

---

## Role in the Pipeline

```
data/raw/   →   scripts/analysis/   →   data/processed/
(per-trial CSVs)   (aggregation,       (summary JSONs,
                    significance        benchmark tables)
                    tests, effect sizes)
```

Scripts in this directory implement:

- **Aggregation** — computes per-condition means, standard deviations, and
  confidence intervals across the 30 trial seeds.
- **Significance testing** — two-sided Mann-Whitney U test with
  Holm-Bonferroni correction for pairwise algorithm comparisons;
  Kruskal-Wallis H for multi-algorithm families.
- **Effect sizes** — rank-biserial correlation r for each pairwise comparison.
- **Benefit-envelope mapping** — identifies the operating-condition range where
  MP-CARE offers a statistically significant advantage (paper §VII).

All statistical procedures are implemented in `src/` (`stats_utils`) and
called from these scripts, ensuring the same code path is exercised by the
test suite (`tests/`).
