"""Store-and-forward message layer for routing-protocol control traffic.

Used by the AODV and OLSR baselines (Phase 2) on top of the explicit network
model: every message — broadcast or unicast — is attempted per link with the
same physical PRR / MAC-retry model as data packets, so control-plane overhead
is measured under the same physics as the data plane.

Messages are dicts with at least: {"type", "src", "ttl"}.
Delivery is scheduled for the next tick (half-duplex radio abstraction);
successful reception decrements TTL at the receiver.

All randomness flows through the LinkModel RNG -> a run stays deterministic
for a fixed seed.
"""

from __future__ import annotations

import math


class MessageBus:
    """Per-tick message exchange between nodes over physical links.

    One instance is shared by all nodes. Each tick:
      1. nodes enqueue messages (broadcast -> one copy per neighbor link,
         unicast -> one copy on the specified next-hop link)
      2. `deliver()` attempts each copy with the link PRR model
      3. receivers' handler callbacks fire with (msg, from_id)
    """

    def __init__(self, link):
        self.link = link            # network_model.model.LinkModel
        self._pending: list[tuple[dict, int, int, float]] = []
        # (msg, from_id, to_id, distance_m)
        self.control_bytes_sent: dict[int, int] = {}   # per node
        self.control_msgs_sent: dict[int, int] = {}    # per node
        self.rx_dropped: int = 0

    # ---------------- send side ----------------
    def broadcast(self, msg: dict, from_id: int, positions: dict, links: dict):
        size = msg.get("size", 0)
        self.control_msgs_sent[from_id] = self.control_msgs_sent.get(from_id, 0) + 1
        self.control_bytes_sent[from_id] = self.control_bytes_sent.get(from_id, 0) + size
        for nb in sorted(links.get(from_id, ())):
            d = math.dist(positions[from_id][:2], positions[nb][:2])
            self._pending.append((dict(msg), from_id, nb, d))

    def unicast(self, msg: dict, from_id: int, to_id: int, positions: dict):
        size = msg.get("size", 0)
        self.control_msgs_sent[from_id] = self.control_msgs_sent.get(from_id, 0) + 1
        self.control_bytes_sent[from_id] = self.control_bytes_sent.get(from_id, 0) + size
        d = math.dist(positions[from_id][:2], positions[to_id][:2])
        self._pending.append((dict(msg), from_id, to_id, d))

    # ---------------- receive side ----------------
    def deliver(self, handlers: dict[int, object], t: float | None = None):
        """Attempt all pending transmissions; fire per-node handlers on success.

        handlers: node_id -> object with .on_message(msg, from_id) (or callable).
        t: current sim time, stamped into msg["_t"] so receiver-side timers
        (OLSR HELLO/TC expiry) see the delivery time.
        """
        pending, self._pending = self._pending, []
        for msg, frm, to, d in pending:
            if not self.link.unicast_success(d):
                continue  # lost on the channel
            h = handlers.get(to)
            if h is None:
                self.rx_dropped += 1
                continue
            # TTL: only messages that CARRY one are hop-limited (HELLO ttl=1,
            # TC/RREQ ttl>0). Messages without a ttl field are not hop-limited
            # (unicast RREP traveling a fixed reverse path).
            if "ttl" in msg:
                if msg["ttl"] <= 0:
                    continue
                msg["ttl"] -= 1
            if t is not None:
                msg["_t"] = t
            if hasattr(h, "on_message"):
                h.on_message(msg, frm)
            elif callable(h):
                h(msg, frm)

    # ---------------- stats ----------------
    def control_overhead_per_node(self, n_nodes: int, duration_s: float) -> float:
        """Average control packets per node per second (§4.5 metric)."""
        total_msgs = sum(self.control_msgs_sent.values())
        return total_msgs / max(1, n_nodes) / max(1e-9, duration_s)
