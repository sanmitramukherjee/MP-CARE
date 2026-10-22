"""Composite route cost for MP-CARE (plan section 5.2.1).

    C(p) = w_d*D(p) + w_e*E(p) + w_t*L(p) + w_q*Q(p) + w_h*H(p)

All five terms are min-max normalized over the candidate set (plan: "all
terms are min-max normalized over the candidate set"); a term whose max ==
min across candidates contributes 0 (no information).

Term definitions (per plan table):
  D  total Euclidean path length (m)
  E  sum over edges of ETX_pred * hoplen, with hoplen = edge length / R_max
     (the plan's "ETX_pred(e) . hoplen(e)"; normalized by R_max so E stays a
     transmission count, not a distance)
  L  sum of per-hop predicted delay (propagation + airtime * ETX_pred +
     queueing term from the next hop's beaconed queue depth)
  Q  max queue occupancy ratio along the path
  H  hop count

Minimax variant (plan): C_mm = w_d*D + w_m*max_e ETX_pred(e) — guards the
weakest link. Enabled by setting w_m > 0 (then w_e is ignored).

Borrowed-from-AT-PP note: the minimax idea is cited to AT-PP/MT-PP in the
plan; our delta is multi-agent dynamics + prediction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DEFAULT_WEIGHTS = {"w_d": 1.0, "w_e": 1.0, "w_t": 0.5, "w_q": 0.5, "w_h": 0.3,
                   "w_m": 0.0}


@dataclass
class PathFeatures:
    path: list[int]
    distance_m: float
    etx_sum: float          # includes hoplen weighting per plan
    etx_max: float
    latency_s: float
    queue_max: float
    hops: int


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


class CostEngine:
    def __init__(self, weights: dict[str, float] | None = None):
        self.w = dict(DEFAULT_WEIGHTS)
        self.w.update(weights or {})

    def features(self, path: list[int], positions: dict, etx_pred_fn,
                 delay_pred_fn, queue_fn) -> PathFeatures:
        """Measure one candidate path.

        etx_pred_fn(a, b)   -> predicted ETX of edge (a, b)
        delay_pred_fn(a, b) -> predicted per-hop delay in seconds
        queue_fn(node)      -> queue occupancy ratio of a node (from beacons)
        """
        dist = 0.0
        etx_sum = 0.0
        etx_max = 0.0
        lat = 0.0
        qmax = 0.0
        for a, b in zip(path, path[1:]):
            e = etx_pred_fn(a, b)
            hoplen = math.dist(positions[a][:2], positions[b][:2])
            dist += hoplen
            etx_sum += e * (hoplen / max(1e-9, 300.0))   # plan: ETX * hoplen
            etx_max = max(etx_max, e)
            lat += delay_pred_fn(a, b)
            qmax = max(qmax, queue_fn(b))
        return PathFeatures(path=list(path), distance_m=dist, etx_sum=etx_sum,
                            etx_max=etx_max, latency_s=lat, queue_max=qmax,
                            hops=max(0, len(path) - 1))

    def cost(self, feats: PathFeatures, norm: dict[str, list[float]]) -> float:
        """Weighted cost given the candidate-set normalizations."""
        w = self.w
        if w.get("w_m", 0.0) > 0.0:      # minimax variant
            return (w["w_d"] * norm["D"][feats.idx]
                    + w["w_m"] * norm["E"][feats.idx])
        return (w["w_d"] * norm["D"][feats.idx]
                + w["w_e"] * norm["E"][feats.idx]
                + w["w_t"] * norm["L"][feats.idx]
                + w["w_q"] * norm["Q"][feats.idx]
                + w["w_h"] * norm["H"][feats.idx])

    def rank(self, feats: list[PathFeatures]) -> list[tuple[float, PathFeatures]]:
        """Rank candidates: returns [(cost, features)] ascending by cost."""
        if not feats:
            return []
        for i, f in enumerate(feats):
            f.idx = i
        norm = {
            "D": _minmax([f.distance_m for f in feats]),
            "E": _minmax([f.etx_sum for f in feats]),
            "L": _minmax([f.latency_s for f in feats]),
            "Q": _minmax([f.queue_max for f in feats]),
            "H": _minmax([float(f.hops) for f in feats]),
        }
        scored = [(self.cost(f, norm), f) for f in feats]
        scored.sort(key=lambda cf: (cf[0], cf[1].path))
        return scored
