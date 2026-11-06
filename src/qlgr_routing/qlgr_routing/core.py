"""QLGR — tabular Q-learning geographic routing baseline (improvement.md Task 1).

Adaptation of QLGR ("QLGR: A Q-learning-based Geographic FANET Routing
Algorithm Based on Multi-agent Reinforcement Learning", Qiu, Xie, Wang, Ye,
Yang, KSII Transactions on Internet and Information Systems 15(11), 2021,
DOI 10.3837/TIIS.2021.11.020). The published protocol is multi-agent tabular
Q-learning whose per-node tables with a one-hop-neighbor state; the exact
kernels are not reproduced from the paper (its full text was not readable at
implementation time), so this is an honest ADAPTATION inside the QLGR
protocol family, documented as follows (see paper Section "baselines"):

  State   : (progress_bin, etx_bin, speed_bin) of a candidate next hop
            relative to the packet destination and the current node —
            the (distance, ETX, speed) feature triple from improvement.md.
  Actions : one-hop neighbors (candidate forwarders), exactly as in QLGR's
            neighbor-based action space.
  Policy  : greedy argmax over Q-values at forwarding time; epsilon-greedy
            exploration during training runs.
  Reward  : r = +1 per successful unicast attempt, -1 per failed attempt,
            +5 terminal credit on end-to-end delivery; applied as an
            n-step (J = 4) discounted reward replay along the most recent
            per-packet state records (gamma = 0.9), which is the standard
            delayed-credit treatment of tabular Q-routing.

Like the GPSR baseline, selection uses the position service (geocast): the
forwarder knows the destination's current location. There is no control
plane: the table is learned purely from data-plane outcomes, so control
overhead is 0 by construction — the protocol is inference-cheap, and its
per-tick compute is measured by the same instrumentation as MP-CARE's.

Tables are per-state class tables (not per-neighbor), so a table trained on
one random geometric instance stays valid on fresh instances: merging across
training seeds is a plain state-wise mean (merge_tables()).
"""

from __future__ import annotations

import math
import random
from collections import deque

# discretization: 4 progress bins x 3 ETX bins x 3 receding-speed bins = 36 states
PROGRESS_BINS = 4   # 0: not closer / 1: <=25% closer / 2: <=50% / 3: >50%
ETX_BINS = 3        # 0: ETX <= 1.3 / 1: <= 2.5 / 2: worse-or-inf
SPEED_BINS = 3      # 0: approaching-or-slow / 1: receding <= 10 m/s / 2: > 10 m/s
DEFAULT_GAMMA = 0.9
DEFAULT_ALPHA = 0.5
DEFAULT_EPS = 0.15

# registry of the most recently constructed agents (test/training-side table
# capture): run_sim.run() drops its agent objects, so the campaign driver
# reads the learned tables out through this module-level batch list.
_BATCH: list["QlgrAgent"] = []


class QlgrAgent:
    """Per-node tabular Q-geographic router (multi-agent QLGR family)."""

    def __init__(self, node_id: int, n: int, *, learn: bool = True,
                 seed: int = 0, eps: float = DEFAULT_EPS,
                 alpha: float = DEFAULT_ALPHA,
                 gamma: float = DEFAULT_GAMMA):
        self.node_id = node_id
        self.n = n
        self.learn = bool(learn)
        self.eps = 0.0 if not self.learn else float(eps)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.rng = random.Random((node_id + 1) * 7919 + seed)   # deterministic
        self.q: dict[tuple[int, int, int], float] = {}
        # per-tick neighbor geometry (previous state for the radial-speed bin)
        self._pos_prev: dict[int, tuple[float, float, float]] = {}
        self._d_prev: dict[int, float] = {}
        self._t_prev: dict[int, float] = {}
        # per-packet learning + route-change bookkeeping
        self._rec: dict[int, deque] = {}
        self._last_nh: dict[int, int | None] = {}
        self.reroute_event = False          # route telemetry (same flag as GPSR)
        self.n_route_changes = 0
        self.n_explore = 0
        self.n_attempts = 0
        _BATCH.append(self)

    # ------------------------------------------------------------- serialize
    def load_table(self, table: dict) -> None:
        """Load a merged state -> preference mapping (keys '(p,e,s)')."""
        for k, v in (table or {}).items():
            try:
                s = tuple(int(x) for x in k.strip("()").split(","))
            except (ValueError, AttributeError):
                continue
            if len(s) == 3:
                self.q[s] = float(v)

    def to_dict(self) -> dict:
        return {f"({p},{e},{s})": round(v, 4) for (p, e, s), v in self.q.items()}

    # ------------------------------------------------------- protocol tick
    def observe(self, t: float, positions, links, snap) -> None:
        """Track neighbor geometry (radial receding speed of each neighbor)."""
        me = positions[self.node_id]
        for nb in links.get(self.node_id, ()):
            d = math.dist(me[:2], positions[nb][:2])
            tp = self._t_prev.get(nb)
            if tp is not None and t - tp > 1e-9 and nb in self._d_prev:
                v = (d - self._d_prev[nb]) / (t - tp)      # >0 = receding
                self._pos_prev[nb] = (v,)
            self._d_prev[nb] = d
            self._t_prev[nb] = t
        # prune stale entries (neighbors that vanished) — bounded bookkeeping
        if len(self._d_prev) > 2 * self.n:
            alive = set(links.get(self.node_id, ()))
            self._d_prev = {k: v for k, v in self._d_prev.items() if k in alive}
            self._t_prev = {k: v for k, v in self._t_prev.items() if k in alive}

    # ---------------------------------------------------- packet forwarding
    def forward(self, pkt, positions, links, snap) -> int | None:
        """Choose the next hop for one packet (Q-geographic rule)."""
        cur = self.node_id
        dst = pkt.dst_pos
        me = positions[cur]
        d_self = math.dist(me[:2], dst[:2])
        cands = []
        for nb in sorted(links.get(cur, ())):
            d_nb = math.dist(positions[nb][:2], dst[:2])
            gain = d_self - d_nb
            pb = 0 if gain <= 0.0 else int(min(PROGRESS_BINS - 1,
                                               4.0 * gain / max(d_self, 1.0)))
            etx = snap.etx_of(cur, nb)
            eb = 0 if (not math.isinf(etx) and etx <= 1.3) \
                else (1 if (not math.isinf(etx) and etx <= 2.5) else 2)
            v = self._pos_prev.get(nb, (0.0,))[0]
            vb = 2 if v > 10.0 else (1 if v > 2.0 else 0)
            s = (pb, eb, vb)
            cands.append((nb, s, self.q.get(s, 0.0), gain, etx))
        if not cands:
            return None                        # isolated: drop (GPSR parity)
        # record the packet's recent state (up to 4 hops of credit replay)
        self._rec.setdefault(pkt.pid, deque(maxlen=4))
        # exploration: epsilon-uniform random candidate (training only)
        if self.learn and self.rng.random() < self.eps:
            nxt, state, _, _, _ = self.rng.choice(cands)
            self.n_explore += 1
        else:
            # greedy argmax Q; ties break on progress gain, then ETX, then id
            cands.sort(key=lambda c: (-c[2], -c[3], c[4], c[0]))
            nxt, state, _, _, _ = cands[0]
        self._rec[pkt.pid].append(state)
        # route telemetry: same route-change definition as the AODV/OLSR rows
        prev = self._last_nh.get(pkt.pid)
        if prev is not None and prev != nxt:
            self.reroute_event = True
            self.n_route_changes += 1
        self._last_nh[pkt.pid] = nxt
        self.n_attempts += 1
        return nxt

    def observe_attempt(self, pkt, ok: bool, delivered: bool) -> None:
        """Data-plane learning feedback (called by the simulation driver)."""
        if not self.learn:
            return
        r = -1.0 if not ok else (5.0 if delivered else 1.0)
        recs = self._rec.get(pkt.pid)
        if not recs:
            return
        # J-step discounted reward replay over this packet's recent states
        for k in range(len(recs) - 1, -1, -1):
            s = recs[k]
            g = self.q.get(s, 0.0)
            self.q[s] = g + self.alpha * (self.gamma ** k * r - g)


def live_tables() -> dict[int, dict]:
    """Tables of the most recent construction batch (training capture)."""
    return {a.node_id: a.to_dict() for a in _BATCH}


def clear_batch() -> None:
    """Drop the registry references (bounded memory across trials)."""
    _BATCH.clear()


def merge_tables(tables: list[dict]) -> dict:
    """Mean of per-state tables across agents/seeds (class-space transfer)."""
    acc: dict[tuple, list[float]] = {}
    for t in tables:
        for k, v in t.items():
            try:
                s = tuple(int(x) for x in k.strip("()").split(","))
            except (ValueError, AttributeError):
                continue
            acc.setdefault(s, []).append(float(v))
    return {f"({p},{e},{s})": round(sum(vs) / len(vs), 4)
            for (p, e, s), vs in sorted(acc.items())}
