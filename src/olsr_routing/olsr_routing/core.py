"""OLSR protocol core (RFC 3626 subset) for the MP-CARE simulation.

Pure Python, deterministic, wired to the explicit network model through the
MessageBus. Implements the parts that matter for fair baseline comparison:

- HELLO exchange (1 s default interval in this implementation) to learn
  1-hop and symmetric 2-hop neighbors.
- MPR (Multi-Point Relay) selection: the greedy heuristic from RFC 3626 §8.3.1
  (pick the 1-hop neighbor that reaches the most still-uncovered 2-hop
  neighbors, repeat until covered; reachability = willingness-willing).
- TC (Topology Control) messages flooded by MPRs only — this is the overhead
  advantage OLSR is known for.
- Routing table via Dijkstra over the topology table (hop-count metric).
- Per-node control overhead accounting.

Deviations from RFC 3626 (documented, not silent):
- TCs advertise the FULL symmetric neighbor set, not only MPR selectors.
  Strict selector-only advertisement makes a leaf node (no one selects it)
  absent from every topology table, i.e. structurally unreachable at 5-node
  chain scale. Full-set advertisement with MPR-reduced FLOODING is the usual
  small-network mode and preserves the overhead advantage (only MPRs repeat
  TCs; smaller retransmission count), which is what we measure.
- No willingness levels (all WILL_DEFAULT), no link hysteresis, fixed message
  intervals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

HELLO_INTERVAL = 1.0          # s (RFC default 2 s; 1 s matches our beacon rate)
TC_INTERVAL = 5.0             # s (RFC default 5 s)
TOP_HOLD_TIME = 15.0          # s: topology entries expire after 3 TC intervals
NEIGH_HOLD_TIME = 3.0         # s: HELLO-derived entries expire after 3 HELLOs
MSG_SIZES = {"HELLO": 64, "TC": 80 + 8}   # TC grows with advertised links


@dataclass
class TopologyEntry:
    last_hop: int
    dest: int
    expires_t: float


class OlsrAgent:
    """One OLSR node. Control messages flow via the shared MessageBus."""

    def __init__(self, node_id: int, n_nodes: int):
        self.node_id = node_id
        self.n_nodes = n_nodes
        # HELLO-derived state
        self.neigh: dict[int, float] = {}            # 1-hop neighbor -> last heard t
        self.neigh_links = {}                        # nbr -> set(nbr's neighbors)
        self.two_hop: dict[int, dict[int, float]] = {}  # via -> {2hop -> last heard}
        # MPR state
        self.mpr_set: set[int] = set()               # my selected MPRs
        self.mpr_selectors: set[int] = set()         # who selected me
        # TC-derived topology SET: (last_hop, dest) -> expiry t. RFC 3626 keeps
        # ALL pairs — several advertisers may announce the same dest, and the
        # routing-table computation picks the shortest. (A dest-keyed dict that
        # overwrites is WRONG: it can leave only cyclic entries and make a
        # reachable node unroutable.)
        self.topology: dict[tuple[int, int], TopologyEntry] = {}
        self._seen_tc: set[tuple[int, int]] = set()  # (orig, seq) dedup
        self._last_hello_t: float = -1e9
        self._last_tc_t: float = -1e9
        self._tc_seq = 0
        self.next_hop_change: bool = False
        self.converged_t: float | None = None        # set by the sim loop
        self.pending_out: list[tuple[str, dict, int | None]] = []

    # ------------------------------------------------------ MPR selection
    def _select_mpr(self) -> set[int]:
        """RFC 3626 §8.3.1 greedy heuristic over current 2-hop knowledge."""
        N = set(self.neigh)
        N2: dict[int, set[int]] = {}
        for via, nbrs in self.two_hop.items():
            for x in nbrs:
                if x != self.node_id and x not in N:
                    N2.setdefault(x, set()).add(via)
        mprs: set[int] = set()
        covered: set[int] = set()
        # 1) nodes reachable through only one neighbor force that neighbor
        for x, vias in list(N2.items()):
            if len(vias) == 1:
                mprs.add(next(iter(vias)))
                covered.add(x)
        covered |= {x for x in N2 if N2[x] & mprs}
        # 2) greedy: pick the neighbor covering the most uncovered 2-hop nodes
        while covered < set(N2):
            best, best_gain = None, 0
            for n in sorted(N):
                if n in mprs:
                    continue
                gain = sum(1 for x in N2 if x not in covered and n in N2[x])
                if gain > best_gain:
                    best, best_gain = n, gain
            if best is None:
                break
            mprs.add(best)
            covered |= {x for x in N2 if best in N2[x]}
        return mprs

    # ------------------------------------------------------ HELLO
    def send_hello(self) -> dict:
        return {"type": "HELLO", "src": self.node_id,
                "neigh": sorted(self.neigh),
                "mprs": sorted(self.mpr_set),
                "size": MSG_SIZES["HELLO"] + 4 * len(self.neigh)}

    def _on_hello(self, msg, frm):
        t = msg["_t"]
        self.neigh[frm] = t
        self.neigh_links[frm] = set(msg.get("neigh", []))
        self.two_hop[frm] = {x: t for x in msg.get("neigh", [])
                             if x != self.node_id}
        if self.node_id in msg.get("mprs", []):
            self.mpr_selectors.add(frm)
        # refresh or expire
        cutoff = t - NEIGH_HOLD_TIME
        for n in list(self.neigh):
            if self.neigh[n] < cutoff:
                self.neigh.pop(n, None)
                self.neigh_links.pop(n, None)
        for via in list(self.two_hop):
            if via not in self.neigh:
                self.two_hop.pop(via, None)

    # ------------------------------------------------------ TC
    def send_tc(self) -> dict | None:
        # Deviation (see module docstring): advertise the full neighbor set so
        # every reachable node enters every topology table; MPRs still gate
        # the FLOODING (who retransmits), which is the overhead we compare.
        if not self.neigh:
            return None
        msg = {"type": "TC", "src": self.node_id, "seq": self._tc_seq,
               "selectors": sorted(self.neigh),
               "size": MSG_SIZES["TC"] + 4 * len(self.neigh)}
        self._tc_seq += 1
        return msg

    def _on_tc(self, msg, frm):
        t = msg["_t"]
        key = (msg["src"], msg["seq"])
        if key in self._seen_tc:
            return
        self._seen_tc.add(key)
        for d in msg.get("selectors", []):
            if d == self.node_id:
                continue
            # keep ALL (last_hop, dest) pairs; refresh expiry per origin
            self.topology[(msg["src"], d)] = TopologyEntry(
                last_hop=msg["src"], dest=d, expires_t=t + TOP_HOLD_TIME)
        # MPR forwarding rule: retransmit only if the last hop is my MPR selector
        if frm in self.mpr_selectors:
            fwd = dict(msg)
            self.pending_out.append(("bcast", fwd, None))

    # ------------------------------------------------------ messages / timers
    def on_message(self, msg: dict, frm: int):
        msg["_t"] = msg.get("_t", 0.0)
        if msg["type"] == "HELLO":
            self._on_hello(msg, frm)
        elif msg["type"] == "TC":
            self._on_tc(msg, frm)

    def tick(self, t: float, dt: float, links: dict):
        """Timers: HELLO/TC emission, MPR recompute, topology expiry."""
        self.next_hop_change = False
        if t - self._last_hello_t >= HELLO_INTERVAL:
            self._last_hello_t = t
            self.pending_out.append(("bcast", self.send_hello(), None))
            self.mpr_set = self._select_mpr()
        if t - self._last_tc_t >= TC_INTERVAL:
            self._last_tc_t = t
            tc = self.send_tc()
            if tc:
                self.pending_out.append(("bcast", tc, None))
        # topology expiry
        for k in list(self.topology):
            if self.topology[k].expires_t < t:
                self.topology.pop(k)

    def flush(self, bus, positions: dict, links: dict):
        for kind, msg, to in self.pending_out:
            bus.broadcast(msg, self.node_id, positions, links)
        self.pending_out.clear()    # ------------------------------------------------------ routing table
    def next_hop_for(self, dst: int) -> int | None:
        """RFC 3626 section 10 routing-table computation (fixed-point).

        next_hop(neighbor) = neighbor (base case, dist 1).
        For every topology entry (last_hop=A, dest=D): D is reachable through
        the route to A plus one hop, i.e. next_hop(D) = next_hop(A),
        dist(D) = dist(A) + 1. Iterate to a fixed point (<= N rounds).
        Returns None when dst is not in any known topology (unreachable).
        """
        if dst == self.node_id:
            return None
        next_hop: dict[int, int] = {n: n for n in self.neigh}
        dist: dict[int, int] = {n: 1 for n in self.neigh}
        entries = [(te.last_hop, te.dest) for te in self.topology.values()]
        for _ in range(self.n_nodes + 1):
            changed = False
            for last, dest in entries:
                if last in dist:
                    nd = dist[last] + 1
                    if dest not in dist or nd < dist[dest]:
                        dist[dest] = nd
                        next_hop[dest] = next_hop[last]
                        changed = True
            if not changed:
                break
        return next_hop.get(dst)
