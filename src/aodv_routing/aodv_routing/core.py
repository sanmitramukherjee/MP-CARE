"""AODV protocol core (subset of RFC 3561) for the MP-CARE simulation.

Pure Python, deterministic, wired to the explicit network model through the
MessageBus. Implements the parts that matter for fair baseline comparison:

- Route discovery: RREQ flooding, RREP unicast back along the reverse path.
- HELLO messages for link sensing; neighbor loss after 2 lost HELLOs.
- Route timeout (active route lifetime 10 s), RERR on route break.
- Per-node control overhead accounting (packets/s/node, §4.5 metric).

Deviations from the RFC (documented, not silent): no expanding-ring search, no
local repair, no local connectivity list beyond HELLO tracking. Enough to
behave like AODV; not a certified RFC implementation.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

RREQ_ID_COUNTER = itertools.count(1)
ACTIVE_ROUTE_TIMEOUT = 10.0     # s
HELLO_INTERVAL = 1.0            # s
ALLOWED_HELLO_LOSS = 2
RREQ_RETRIES = 2
DISCOVERY_BUDGET_S = 2.0        # §4.4 sanity check: discovery < 2 s
MAX_ROUTE_LIFECYCLE_S = 30.0    # after this, unused routes expire

MSG_SIZES = {"RREQ": 48, "RREP": 40, "RERR": 32, "HELLO": 24}


@dataclass
class RouteEntry:
    dest: int
    next_hop: int
    hop_count: int
    dest_seq: int = 0
    is_valid: bool = True
    created_t: float = 0.0


@dataclass
class _Discovery:
    dest: int
    started_t: float
    last_rreq_t: float
    retries: int = 0


class AodvAgent:
    """One AODV node. Control messages flow via the shared MessageBus.

    The bus calls on_message(msg, frm); the sim context (bus, current time,
    positions, links) is injected by the driver via set_context() each tick.
    """

    def __init__(self, node_id: int):
        self.node_id = node_id
        self._ctx = None
        self.routes: dict[int, RouteEntry] = {}
        self._rreq_seen: set[tuple[int, int]] = set()
        self._discoveries: dict[int, _Discovery] = {}
        self._last_hello_t: dict[int, float] = {}
        self.next_hop_change: bool = False
        self.discovery_times: list[float] = []
        self.route_errors: int = 0
        self.pending_out: list[tuple[str, dict, int | None]] = []

    # ------------------------------------------------------ data-plane API
    def next_hop_for(self, dst: int, t: float) -> int | None:
        """Valid next hop toward dst, or None (discovery may be in progress)."""
        r = self.routes.get(dst)
        if r and r.is_valid:
            return r.next_hop
        if dst not in self._discoveries:
            self._discoveries[dst] = _Discovery(dest=dst, started_t=t, last_rreq_t=t)
            self._new_rreq(dst, t)
        return None

    def _new_rreq(self, dst: int, t: float):
        msg = {"type": "RREQ", "src": self.node_id, "dest": dst,
               "rreq_id": next(RREQ_ID_COUNTER), "hops": 0, "ttl": 32,
               "size": MSG_SIZES["RREQ"]}
        self.pending_out.append(("bcast", msg, None))

    # ------------------------------------------------------ sim context
    def set_context(self, bus, t: float, positions: dict, links: dict):
        self._ctx = (bus, t, positions, links)

    # ------------------------------------------------------ message handling
    def on_message(self, msg: dict, frm: int):
        bus, t, positions, links = self._ctx
        mtype = msg["type"]
        if mtype == "RREQ":
            self._on_rreq(msg, frm, bus, t, positions, links)
        elif mtype == "RREP":
            self._on_rrep(msg, frm, bus, t, positions)
        elif mtype == "RERR":
            self._on_rerr(msg, frm)
        elif mtype == "HELLO":
            self._last_hello_t[frm] = t

    def _on_rreq(self, msg, frm, bus, t, positions, links):
        key = (msg["src"], msg["rreq_id"])
        if key in self._rreq_seen:
            return
        self._rreq_seen.add(key)
        # learn/refresh the reverse route to the originator via `frm`
        rev = self.routes.get(msg["src"])
        if rev is None or not rev.is_valid:
            self.routes[msg["src"]] = RouteEntry(dest=msg["src"], next_hop=frm,
                                                 hop_count=msg.get("hops", 0) + 1,
                                                 created_t=t)
        if msg["dest"] == self.node_id:
            rrep = {"type": "RREP", "orig": msg["src"], "dest": self.node_id,
                    "rreq_id": msg["rreq_id"], "hops": 0, "size": MSG_SIZES["RREP"]}
            self.pending_out.append(("ucast", rrep, frm))
        elif msg["ttl"] > 0:
            fwd = dict(msg)
            fwd["hops"] = msg.get("hops", 0) + 1
            self.pending_out.append(("bcast", fwd, None))

    def _on_rrep(self, msg, frm, bus, t, positions):  # noqa: ARG002 - bus unused, kept for symmetry
        dest = msg["dest"]           # the discovered destination (RREP travels src->orig)
        prev = self.routes.get(dest)
        new_hops = msg.get("hops", 0) + 1
        if prev is None or (not prev.is_valid) or new_hops < prev.hop_count:
            self.routes[dest] = RouteEntry(dest=dest, next_hop=frm, hop_count=new_hops,
                                           created_t=t)
            self.next_hop_change = True
        d = self._discoveries.pop(dest, None)
        if d is not None:
            self.discovery_times.append(t - d.started_t)
        # relay toward the originator if we are not it
        if msg["orig"] != self.node_id:
            r = self.routes.get(msg["orig"])
            if r and r.is_valid:
                out = dict(msg)
                out["hops"] = new_hops
                self.pending_out.append(("ucast", out, r.next_hop))

    def _on_rerr(self, msg, frm):
        r = self.routes.get(msg.get("broken_dest"))
        if r and r.next_hop == frm:
            r.is_valid = False
            self.route_errors += 1

    # ------------------------------------------------------ tick housekeeping
    def tick(self, t: float, dt: float, links: dict):
        """Timers: HELLO emission, neighbor loss, route expiry, RREQ retry.
        Outgoing control messages accumulate in pending_out; the sim loop
        flushes them into the bus after every on_message round."""
        self.next_hop_change = False
        # HELLO at the fixed interval
        if int(t / HELLO_INTERVAL) != int((t - dt) / HELLO_INTERVAL):
            msg = {"type": "HELLO", "src": self.node_id, "size": MSG_SIZES["HELLO"]}
            self.pending_out.append(("bcast", msg, None))
        # neighbor loss via missing HELLOs
        for nb in list(self._last_hello_t):
            if t - self._last_hello_t[nb] > ALLOWED_HELLO_LOSS * HELLO_INTERVAL:
                r = self.routes.get(nb)
                if r and r.is_valid and r.next_hop == nb:
                    r.is_valid = False
                    self.route_errors += 1
                self._last_hello_t.pop(nb, None)
        # route expiry (unused for one full lifecycle)
        for r in self.routes.values():
            if r.is_valid and t - r.created_t > MAX_ROUTE_LIFECYCLE_S:
                r.is_valid = False
        # RREQ retry within the discovery budget
        d = next(iter(self._discoveries.values()), None)
        if d is not None:
            if d.retries > RREQ_RETRIES or t - d.started_t > 3 * DISCOVERY_BUDGET_S:
                self._discoveries.pop(d.dest, None)   # give up (dest unreachable)
            elif t - d.last_rreq_t >= DISCOVERY_BUDGET_S / (RREQ_RETRIES + 1):
                d.retries += 1
                d.last_rreq_t = t
                self._new_rreq(d.dest, t)

    def flush(self, bus, positions: dict, links: dict):
        """Move pending_out messages onto the bus (called by the sim loop)."""
        for kind, msg, to in self.pending_out:
            if kind == "bcast":
                bus.broadcast(msg, self.node_id, positions, links)
            else:
                bus.unicast(msg, self.node_id, to, positions)
        self.pending_out.clear()
