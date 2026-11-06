"""LET-OLSR baseline (phase 9) — a prediction-augmented OLSR comparator.

Adapts the link-stability idea of Xie (2018, APCC)'s LET-OLSR
(A8 in bibliography.bib): next-hop selection weighs kinematic Link
Expiration Time (LET) alongside hop count, so the routing table prefers
stable links over merely short ones.

Design, disclosed:
- LET for each 1-hop link is estimated kinematically (Boleng-style):
  consecutive observed samples of the pair separation give a radial
  separation rate; LET = (Rmax - d) / rate for separating links, >= a
  "stable" cap for approaching links. The rate signal comes from the
  platform's per-tick position service (`observe_state`) — the same
  information regime the data plane already grants every candidate
  (geocast destination positions), so the baseline shares MP-CARE's
  information access rather than a weaker or stronger one.
- The base OLSR fixed-point routing table (RFC 3626 section 10 semantics
  from olsr_routing/core.py) is recomputed with FLOAT costs: 1-hop edges
  cost 1 + LET_WEIGHT * (1 - min(LET, LET_CAP)/LET_CAP), capped; entries
  from the topology table default to plain hop cost (interior links are
  not this node's measurements).
- The residual-energy weighting of the original LET-OLSR is omitted:
  the platform has no energy model. Disclosed as an adaptation, like the
  QLGR baseline.
"""
from __future__ import annotations

import math

from .core import OlsrAgent

LET_CAP_S = 30.0     # links with LET >= cap are "stable enough" (no penalty)
LET_WEIGHT = 1.0     # tie-break weight: stability vs. hop count (symmetric)
MIN_RATE = 1e-3      # m/s: below this a separating link is effectively stable


def edge_penalty(let_s: float | None) -> float:
    """Edge-cost ADDEND for a 1-hop link with estimated LET (None = unknown,
    no penalty). 0 for stable/approaching links, -> LET_WEIGHT as LET -> 0."""
    if let_s is None or let_s >= LET_CAP_S:
        return 0.0
    return LET_WEIGHT


class LetOlsrAgent(OlsrAgent):
    """OLSR agent whose next-hop selection prefers kinematically stable links."""

    def __init__(self, node_id: int, n_nodes: int, max_range_m: float = 300.0):
        super().__init__(node_id, n_nodes)
        self.max_range_m = float(max_range_m)
        self._obs: tuple[float, dict] | None = None   # (t, positions) of last tick
        self._let: dict[int, float | None] = {}       # neighbor -> LET estimate (s)

    # ------------------------------------------------------- kinematic LET
    def observe_state(self, t: float, positions: dict) -> None:
        """Per-tick ground-truth snapshot from the platform's position
        service. Updates the radial separation rate -> LET for each CURRENT
        1-hop neighbor. Honesty note: this only reads the same service the
        forwarding loop already uses; it does not look at any protocol state
        the LET-OLSR design does not define."""
        prev = self._obs
        self._obs = (t, positions)
        if prev is None:
            return
        t0, pos0 = prev
        dt = t - t0
        if dt <= 0:
            return
        me0, me1 = pos0.get(self.node_id), positions.get(self.node_id)
        if me0 is None or me1 is None:
            return
        for nb in list(self.neigh):
            a0, a1 = pos0.get(nb), positions.get(nb)
            if a0 is None or a1 is None:
                self._let[nb] = None
                continue
            d0 = math.dist(me0[:2], a0[:2])
            d1 = math.dist(me1[:2], a1[:2])
            rate = (d1 - d0) / dt                       # >0 = separating
            if rate <= MIN_RATE:                        # approaching/stable
                self._let[nb] = LET_CAP_S
            else:
                self._let[nb] = min((self.max_range_m - d1) / rate, LET_CAP_S)
        for nb in list(self._let):
            if nb not in self.neigh:
                self._let.pop(nb, None)

    # ------------------------------------------------------ routing table
    def _own_edge_cost(self, nb: int) -> float:
        return 1.0 + edge_penalty(self._let.get(nb))

    def next_hop_for(self, dst: int) -> int | None:
        """Same fixed-point structure as the base RFC-3626 routing table, but
        with the node's own 1-hop edges priced by stability instead of
        uniform unit cost. Unreachable -> None, like the base."""
        if dst == self.node_id:
            return None
        next_hop: dict[int, int] = {n: n for n in self.neigh}
        dist: dict[int, float] = {n: self._own_edge_cost(n) for n in self.neigh}
        entries = [(te.last_hop, te.dest) for te in self.topology.values()]
        for _ in range(self.n_nodes + 1):
            changed = False
            for last, dest in entries:
                if last in dist:
                    nd = dist[last] + 1.0    # interior edges: unknown LET
                    if dest not in dist or nd < dist[dest]:
                        dist[dest] = nd
                        next_hop[dest] = next_hop[last]
                        changed = True
            if not changed:
                break
        return next_hop.get(dst)
