"""Phase 4 statistics (implementation_plan.md section 6.5 statistical protocol).

Pure-Python implementations (numpy/math only — scipy is deliberately NOT a
dependency). What is exact and what is approximated, stated up front:

- Mann-Whitney U: rank-based U statistic is EXACT (tie-corrected); the p-value
  uses the normal approximation with tie + continuity correction, which is
  accurate to ~3 decimal places for n >= 8 per group (our n = 30).
- Kruskal-Wallis: H statistic is EXACT (tie-corrected); the p-value uses the
  chi-square survival function via the Wilson-Hilferty cube-root normal
  approximation (df = k - 1). Accurate to ~1e-4 for df >= 2; good enough to
  compare against p < 0.05. Any result within 10x of the alpha threshold must
  be double-checked with a real statistics package before publication.
- Cohen's d: pooled-SD formula, exact.
- Rank-biserial correlation: exact from the U statistic.
- Holm-Bonferroni: exact step-down procedure.

Reference values used by tests/test_phase4.py:
- Mann-Whitney example from the scipy documentation (x1 vs x2 below):
  U = 4983.5, p ~= 0.00833 (two-sided normal approximation).
- Kruskal-Wallis hand-checkable case: identical groups -> H = 0, p = 1.
"""

from __future__ import annotations

import math
from statistics import NormalDist

_NORM = NormalDist()


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based) with midrank ties, stable order."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def mannwhitneyu(x: list[float], y: list[float]) -> dict:
    """Two-sided Mann-Whitney U with tie-corrected normal-approximation p."""
    x = [v for v in x if v is not None]
    y = [v for v in y if v is not None]
    if not x or not y:
        return {"U": None, "p": None, "n_x": len(x), "n_y": len(y)}
    n1, n2 = len(x), len(y)
    ranks = _ranks(list(x) + list(y))
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    u2 = n1 * n2 - u1
    u = min(u1, u2)
    mu = n1 * n2 / 2.0
    # tie correction on the pooled sample
    counts: dict[float, int] = {}
    for v in list(x) + list(y):
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t ** 3 - t for t in counts.values())
    n = n1 + n2
    sigma = math.sqrt(n1 * n2 / 12.0 * ((n + 1) - tie_term / (n * (n - 1))))
    if sigma == 0:
        p = 1.0 if u == mu else 0.0
    else:
        z = (u - mu + 0.5 * math.copysign(1.0, mu - u)) / sigma  # continuity
        p = 2.0 * _NORM.cdf(-abs(z))
    return {"U": u, "p": min(1.0, max(0.0, p)), "n_x": n1, "n_y": n2}


def kruskal_wallis(groups: list[list[float]]) -> dict:
    """Kruskal-Wallis H + chi-square-approximated p (df = k-1).

    Uses the Wilson-Hilferty transformation for the chi-square survival
    function; see module docstring for accuracy notes.
    """
    groups = [[v for v in g if v is not None] for g in groups]
    groups = [g for g in groups if g]
    k = len(groups)
    if k < 2:
        return {"H": None, "p": None, "df": None}
    ns = [len(g) for g in groups]
    n = sum(ns)
    all_vals = [v for g in groups for v in g]
    ranks = _ranks(all_vals)
    sums, idx = [0.0] * k, 0
    for gi, g in enumerate(groups):
        sums[gi] = sum(ranks[idx:idx + len(g)])
        idx += len(g)
    h = (12.0 / (n * (n + 1))) * sum(s ** 2 / ni for s, ni in zip(sums, ns)) \
        - 3.0 * (n + 1)
    # tie correction
    counts: dict[float, int] = {}
    for v in all_vals:
        counts[v] = counts.get(v, 0) + 1
    tie_term = sum(t ** 3 - t for t in counts.values())
    if tie_term >= n ** 3 - n:
        # every value identical (complete tie, e.g. all PDR = 1.0 at 0% loss):
        # zero variance across groups by definition
        return {"H": 0.0, "p": 1.0, "df": k - 1}
    if tie_term:
        h /= 1.0 - tie_term / (n ** 3 - n)
    df = k - 1
    p = _chi2_sf(h, df)
    return {"H": h, "p": p, "df": df}


def _chi2_sf(x: float, df: int) -> float:
    """P(X > x) for chi-square(df) via Wilson-Hilferty normal approximation."""
    if x <= 0:
        return 1.0
    # Wilson-Hilferty: (X/df)^(1/3) ~ Normal(1 - 2/(9df), sqrt(2/(9df)))
    z = ((x / df) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / math.sqrt(2.0 / (9.0 * df))
    return min(1.0, max(0.0, _NORM.cdf(-z)))


def cohens_d(x: list[float], y: list[float]) -> float | None:
    """Pooled-SD Cohen's d (positive => x > y)."""
    x = [v for v in x if v is not None]
    y = [v for v in y if v is not None]
    if len(x) < 2 or len(y) < 2:
        return None
    nx, ny = len(x), len(y)
    mx, my = sum(x) / nx, sum(y) / ny
    sx = math.sqrt(sum((v - mx) ** 2 for v in x) / (nx - 1))
    sy = math.sqrt(sum((v - my) ** 2 for v in y) / (ny - 1))
    sp = math.sqrt(((nx - 1) * sx ** 2 + (ny - 1) * sy ** 2) / (nx + ny - 2))
    if sp == 0:
        return 0.0 if mx == my else math.copysign(math.inf, mx - my)
    return (mx - my) / sp


def rank_biserial(x: list[float], y: list[float]) -> float | None:
    """Rank-biserial correlation from the Mann-Whitney U (positive => x > y)."""
    r = mannwhitneyu(x, y)
    if r["U"] is None:
        return None
    n1, n2 = r["n_x"], r["n_y"]
    # U here is min(u1, u2); recover the signed version from the data
    ranks = _ranks(list(x) + list(y))
    r1 = sum(ranks[:n1])
    u1 = r1 - n1 * (n1 + 1) / 2.0
    return 2.0 * u1 / (n1 * n2) - 1.0


def spearman_rho(x: list[float], y: list[float]) -> float | None:
    """Spearman rank correlation (Pearson on midrank ranks), ties handled."""
    x = [v for v in x if v is not None]
    y = [v for v in y if v is not None]
    n = min(len(x), len(y))
    if n < 3:
        return None
    x, y = x[:n], y[:n]
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def holm_bonferroni(pvals: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni step-down adjusted p-values (order preserved)."""
    idx = [i for i, p in enumerate(pvals) if p is not None]
    out: list[float | None] = [None] * len(pvals)
    m = len(idx)
    if m == 0:
        return out
    ranked = sorted(idx, key=lambda i: pvals[i])
    prev = 0.0
    for rank_pos, i in enumerate(ranked):
        adj = min(1.0, (m - rank_pos) * pvals[i])
        adj = max(adj, prev)          # enforce monotonicity
        out[i] = adj
        prev = adj
    return out


def mean_ci95(vals: list[float]) -> tuple[float, float] | None:
    """Mean and half-width of the t-based 95% CI (t crit from a small table)."""
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    if n < 2:
        return (mean, 0.0)
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    half = _t_crit95(n - 1) * math.sqrt(var / n)
    return (mean, half)


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
        8: 2.306, 9: 2.262, 10: 2.228, 12: 2.179, 15: 2.145, 20: 2.093,
        25: 2.064, 29: 2.045, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def _t_crit95(df: int) -> float:
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    for lo, hi in zip(keys, keys[1:]):
        if lo < df < hi:
            frac = (df - lo) / (hi - lo)
            return _T95[lo] * (1 - frac) + _T95[hi] * frac
    return 1.96
