"""Packet-level 802.11 DCF shared-airtime model (opt-in).

The research engine advances in 0.1 s ticks, which is far too coarse to
represent 802.11 slot times directly. This module therefore runs a small
frame-level scheduler *inside* each tick: every unicast attempt is scheduled
with carrier sense, DIFS and exponential backoff, and occupies the medium for
its frame-exchange duration. Frames whose busy periods overlap at a receiver
collide. Time is continuous in seconds; the tick boundary only decides when
protocol logic runs.

What is modelled
----------------
* CSMA/CA: DIFS + random backoff from CW in {32, 64, ..., 1024}.
* Four retransmissions (five attempts), CW doubling per attempt.
* Frame airtime: PHY header + payload at 11 Mbps + SIFS + ACK.
* Physical carrier sense over a configurable range (default 600 m, 2x the
  decode range), with a per-node NAV.
* Half-duplex: a node cannot receive while it is transmitting.
* Hidden terminals: a frame collides at a receiver if another transmitter
  the receiver can sense overlaps it in time.
* Capture: when two frames overlap at a receiver, the earlier-scheduled
  frame is treated as captured and the later one is lost. This is a
  documented simplification; it avoids retroactively changing a result that
  a protocol has already consumed.

What is *not* modelled
----------------------
Virtual carrier sense (RTS/CTS NAV), MCS/rate adaptation, block ACK,
frame aggregation, adjacent-channel interference, and per-node transmit
power. The model is a single-channel DCF abstraction, not a waveform model.

Determinism
-----------
All randomness is drawn from the LinkModel's seeded RNG. With
``NetworkParams.dcf_enabled = False`` this module is never constructed and
the legacy per-attempt Bernoulli path is used unchanged, preserving
byte-identical historical results.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DcfParams:
    slot_s: float = 9e-6
    difs_s: float = 28e-6
    sifs_s: float = 10e-6
    cw_min: int = 32
    cw_max: int = 1024
    retries: int = 4
    data_rate_bps: float = 11e6
    phy_header_s: float = 192e-6
    ack_s: float = 304e-6
    cs_range_m: float = 600.0
    default_frame_bytes: int = 1000
    # Synthetic same-channel background load (aggregate frames/s, 1 kB each).
    # Represents video/telemetry traffic that is not part of the delivery
    # metric but consumes airtime. 0 disables it.
    bg_frames_per_s: float = 0.0
    bg_frame_bytes: int = 1000


@dataclass
class _Frame:
    start: float
    end: float
    tx: int
    rx: int


class DcfChannel:
    """Frame-level CSMA/CA scheduler shared by every node in one trial."""

    def __init__(self, rng, params: DcfParams | None = None):
        self.rng = rng
        self.p = params or DcfParams()
        self.now = 0.0
        self.positions: dict[int, tuple[float, float, float]] = {}
        self._frames: list[_Frame] = []
        self._bg_intervals: list[tuple[float, float]] = []
        self._nav: dict[int, float] = {}
        self._cs_cache: dict[int, tuple[int, ...]] = {}
        # instrumentation
        self.frames_scheduled = 0
        self.airtime_s = 0.0
        self.collisions = 0
        self.broadcast_frames = 0
        self.broadcast_rx_attempts = 0
        self.broadcast_rx_blocked = 0
        self.broadcast_rx_ok = 0
        self.broadcast_collided_frames = 0
        self.phy_failures = 0
        self.retry_attempts = 0
        self.packets_ok = 0
        self.packets_failed = 0
        self.deferred = 0
        self.bg_frames = 0
        self.bg_airtime_s = 0.0

    # ------------------------------------------------------------------ tick
    def begin_tick(self, t: float, positions: dict) -> None:
        """Advance the scheduler to a new protocol tick."""
        self.now = float(t)
        self.positions = {int(k): tuple(v) for k, v in positions.items()}
        # Keep only frames that can still overlap future attempts. Frames are
        # sub-millisecond unless the medium is saturated, so this is cheap.
        horizon = max(self.now, self.now - 1.0)
        self._frames = [f for f in self._frames if f.end > horizon]
        self._cs_cache.clear()
        self._bg_intervals = self._make_background(t)

    def _make_background(self, t: float) -> list[tuple[float, float]]:
        """Lay synthetic background frames evenly through the coming tick.

        They are aggregate airtime, not protocol packets: no ACK, no retry,
        no delivery metric. Protocol frames defer around them through the
        DIFS/backoff path, which is the effect under test.
        """
        rate = self.p.bg_frames_per_s
        if rate <= 0.0 or not self.positions:
            return []
        dt = 0.1   # protocol tick (the engine is tick-driven)
        n_frames = int(round(rate * dt))
        if n_frames <= 0:
            return []
        dur = self.frame_time_s(self.p.bg_frame_bytes, ack=False)
        out: list[tuple[float, float]] = []
        # leave headroom so a full-length frame always fits inside the tick
        span = max(1e-6, dt - dur)
        for i in range(n_frames):
            start = t + (i + self.rng.random()) * span / n_frames
            out.append((start, start + dur))
        self.bg_frames += n_frames
        self.bg_airtime_s += n_frames * dur
        self.airtime_s += n_frames * dur
        return out

    def _cs(self, node: int) -> tuple[int, ...]:
        """Nodes within physical carrier-sense range (includes `node`)."""
        hit = self._cs_cache.get(node)
        if hit is not None:
            return hit
        p0 = self.positions.get(node)
        if p0 is None:
            out: tuple[int, ...] = (node,)
        else:
            rng = self.p.cs_range_m
            out = tuple(n for n, q in self.positions.items()
                        if math.dist(p0, q) <= rng)
        self._cs_cache[node] = out
        return out

    def frame_time_s(self, size_bytes: int, ack: bool = True) -> float:
        payload = 8.0 * max(1, int(size_bytes)) / self.p.data_rate_bps
        tail = (self.p.sifs_s + self.p.ack_s) if ack else 0.0
        return self.p.phy_header_s + payload + tail

    # --------------------------------------------------------------- attempt
    def attempt(self, a: int, b: int, per_attempt_prr: float,
                size_bytes: int = 0) -> bool:
        """Try to deliver one packet from a to b through DCF.

        Returns True only if one attempt both avoided a collision and passed
        the physical reception draw. `per_attempt_prr` is the data-frame
        reception probability (the caller passes PRR^2 to include the ACK).
        Every scheduled frame consumes airtime, including failed ones.
        """
        if per_attempt_prr <= 0.0:
            return False
        size = size_bytes or self.p.default_frame_bytes
        dur = self.frame_time_s(size)
        cs_a = self._cs(a)

        for attempt in range(self.p.retries + 1):
            self.retry_attempts += 1
            cw = min(self.p.cw_max, self.p.cw_min * (2 ** attempt))
            backoff = self.rng.randrange(cw) * self.p.slot_s
            # physical carrier sense at the *transmitter* only: a hidden
            # node cannot know that the receiver is already busy, which is
            # exactly how hidden-terminal collisions arise.
            earliest = max(self.now, self._nav.get(a, 0.0))
            start = earliest + self.p.difs_s + backoff
            end = start + dur
            # defer around synthetic background airtime (at most a few
            # iterations: the intervals are laid out in increasing order)
            for _ in range(3):
                hit = [b1 for b0, b1 in self._bg_intervals
                       if start < b1 and end > b0]
                if not hit:
                    break
                start = max(hit) + self.p.difs_s
                end = start + dur
            if earliest > self.now + 1e-9:
                self.deferred += 1   # waited for a busy medium (not backoff)

            # capture rule: an overlapping frame already on the medium wins;
            # this attempt is a collision at the receiver if the earlier
            # transmitter is one the receiver can actually sense.
            collided = False
            for f in self._frames:
                if f.end <= start or f.start >= end:
                    continue
                if f.tx == b:
                    collided = True       # receiver is transmitting: half-duplex
                    break
                if f.tx != a and f.tx in self._cs(b):
                    collided = True       # hidden/overlapping transmitter
                    break

            # schedule the frame (busy medium), whether or not it succeeds
            self._frames.append(_Frame(start, end, a, b))
            self.frames_scheduled += 1
            self.airtime_s += dur
            for n in cs_a:
                self._nav[n] = max(self._nav.get(n, 0.0), end)
            self._nav[a] = max(self._nav.get(a, 0.0), end)
            self._nav[b] = max(self._nav.get(b, 0.0), end)

            if collided:
                self.collisions += 1
                continue
            if self.rng.random() < per_attempt_prr:
                self.packets_ok += 1
                return True
            self.phy_failures += 1
            # failed reception: retry after the medium is free
        self.packets_failed += 1
        return False

    def attempt_broadcast(self, a: int, prr_by_rx: dict[int, float],
                          size_bytes: int = 0) -> dict[int, bool]:
        """One 802.11 broadcast frame (no ACK, no retransmission).

        A broadcast is a single transmission received (or not) by every
        neighbour. Collisions are evaluated per receiver, so a hidden
        transmitter can destroy the frame at one neighbour while another
        receives it. The frame occupies the medium once, which is the point:
        modelling a broadcast as N unicast frames inflates control overhead
        by the node degree.
        """
        if not prr_by_rx:
            return {}
        size = size_bytes or self.p.default_frame_bytes
        dur = self.frame_time_s(size, ack=False)
        cw = self.p.cw_min
        backoff = self.rng.randrange(cw) * self.p.slot_s
        earliest = max(self.now, self._nav.get(a, 0.0))
        start = earliest + self.p.difs_s + backoff
        end = start + dur
        if earliest > self.now + 1e-9:
            self.deferred += 1
        self._frames.append(_Frame(start, end, a, -1))
        self.frames_scheduled += 1
        self.broadcast_frames += 1
        self.airtime_s += dur
        for n in self._cs(a):
            self._nav[n] = max(self._nav.get(n, 0.0), end)
        self._nav[a] = max(self._nav.get(a, 0.0), end)

        out: dict[int, bool] = {}
        cs_a = set(self._cs(a))
        any_blocked = False
        for rx, prr_attempt in prr_by_rx.items():
            self.broadcast_rx_attempts += 1
            if prr_attempt <= 0.0:
                out[rx] = False
                continue
            blocked = False
            for f in self._frames[:-1]:
                if f.end <= start or f.start >= end:
                    continue
                if f.tx == rx:
                    blocked = True           # half-duplex receiver
                    break
                # a concurrent transmitter the receiver can hear destroys it
                if f.tx != a and f.tx in self._cs(rx):
                    blocked = True
                    break
            if blocked:
                self.broadcast_rx_blocked += 1
                any_blocked = True
            ok = (not blocked) and (self.rng.random() < prr_attempt)
            if ok:
                self.broadcast_rx_ok += 1
            out[rx] = ok
        if any_blocked:
            self.broadcast_collided_frames += 1
        return out

    # ----------------------------------------------------------- diagnostics
    def summary(self) -> dict:
        return {
            "frames_scheduled": self.frames_scheduled,
            "airtime_s": round(self.airtime_s, 6),
            "collisions": self.collisions,
            "broadcast_frames": self.broadcast_frames,
            "broadcast_rx_attempts": self.broadcast_rx_attempts,
            "broadcast_rx_blocked": self.broadcast_rx_blocked,
            "broadcast_rx_ok": self.broadcast_rx_ok,
            "broadcast_collided_frames": self.broadcast_collided_frames,
            "phy_failures": self.phy_failures,
            "retry_attempts": self.retry_attempts,
            "packets_ok": self.packets_ok,
            "packets_failed": self.packets_failed,
            "deferred": self.deferred,
            "bg_frames": self.bg_frames,
            "bg_airtime_s": round(self.bg_airtime_s, 6),
        }
