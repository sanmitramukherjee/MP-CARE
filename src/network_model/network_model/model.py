"""Explicit swarm communication model (implementation_plan.md section 3.3).

Pure-Python, deterministic, parameterized — the same module is used by the
headless MVP (scripts/run_sim.py) and the ROS 2 node (network_model/node_ros.py).

Model chain:
    distance d
      -> log-distance path loss  PL(d) = PL0 + 10 * alpha * log10(d)
      -> SNR(d) = Ptx - PL(d) - noise_floor
      -> PRR(d) = 1 / (1 + exp(-k * (SNR(d) - SNR50)))      [sigmoid fit]
      -> ETX    = 1 / (PRR_fwd * PRR_rev)                   [symmetric link]

Defaults follow plan section 3.3: alpha=2.5, PL0=46.4 dB, Ptx=20 dBm,
noise floor=-90 dBm, 802.11b 11 Mbps, beacon interval 1 s, max range ~300 m.
SNR50/k place the PRR knee at the sensitivity edge: SNR(300 m) = 1.66 dB, so
snr50=1.7 dB with k=0.9/dB gives PRR~50% at max range and near-1 inside ~200 m
(the 802.11b 11 Mbps cliff). Unicast counts up to 4 MAC retransmissions
(data+ACK per attempt), as in real 802.11.

No idealized connectivity graphs: out-of-range links have ETX = inf and
cannot carry packets. Injected loss/latency multiply on top of the physics.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .dcf import DcfChannel, DcfParams


@dataclass
class NetworkParams:
    """Parameters of plan section 3.3 (defaults = plan defaults)."""
    alpha: float = 2.5              # path-loss exponent (sweep 2.0-4.0)
    pl0_db: float = 46.4            # reference loss at 1 m
    tx_power_dbm: float = 20.0
    noise_floor_dbm: float = -90.0
    bandwidth_mbps: float = 11.0    # 802.11b
    snr50_db: float = 1.7           # PRR sigmoid: 50% point (at ~max range)
    k_per_db: float = 0.9           # PRR sigmoid: steepness
    mac_retries: int = 4            # 802.11 unicast retransmission limit
    max_range_m: float = 300.0      # beyond this ETX = inf
    beacon_interval_s: float = 1.0  # neighbor discovery period
    injected_loss: float = 0.0      # 0.0-1.0 extra drop probability
    injected_latency_ms: float = 0.0
    queue_size: int = 64            # drop-tail queue length
    rng_seed: int = 1               # deterministic channel randomness
    # --- log-normal shadow fading (campaigns set these; default = off) ---
    # sigma 0.0 reproduces the deterministic LoS baseline exactly (same RNG
    # stream, same deliveries). sigma 2-4 dB brackets the standardized 3GPP
    # LoS value; see run_phase9.py header for the sourcing.
    shadow_sigma_db: float = 0.0
    shadow_tau_s: float = 5.0       # Gauss-Markov decorrelation time constant
    # --- 802.11 DCF shared-airtime model (opt-in; default off) ---
    # When enabled, every unicast attempt is scheduled with carrier sense,
    # DIFS and exponential backoff, and occupies the medium for its frame
    # exchange. Default off preserves the historical per-attempt Bernoulli
    # path byte-for-byte.
    dcf_enabled: bool = False
    dcf_cs_range_m: float = 600.0   # physical carrier-sense range (2x decode)
    dcf_bg_frames_per_s: float = 0.0  # synthetic same-channel load (aggregate)
    dcf_bg_frame_bytes: int = 1000


def path_loss_db(d_m: float, p: NetworkParams) -> float:
    """Log-distance path loss; distances below 1 m use the 1 m reference."""
    d = max(float(d_m), 1.0)
    return p.pl0_db + 10.0 * p.alpha * math.log10(d)


def snr_db(d_m: float, p: NetworkParams) -> float:
    return p.tx_power_dbm - path_loss_db(d_m, p) - p.noise_floor_dbm


def prr(d_m: float, p: NetworkParams) -> float:
    """Packet reception ratio from the SNR sigmoid (section 3.3)."""
    if d_m > p.max_range_m:
        return 0.0
    s = snr_db(d_m, p)
    return 1.0 / (1.0 + math.exp(-p.k_per_db * (s - p.snr50_db)))


def etx(d_m: float, p: NetworkParams) -> float:
    """ETX = 1/(PRR_fwd * PRR_rev). Returns math.inf out of range (section 3.5:
    'ETX -> infinity when out of range')."""
    pr = prr(d_m, p)
    if pr <= 0.0 or d_m > p.max_range_m:
        return math.inf
    return 1.0 / (pr * pr)


class LinkModel:
    """Stateful channel for one tick: deterministic RNG, injected loss/latency.

    `set_link_loss(a, b, p)` adds a per-pair impairment (e.g. interference on
    one link) applied to every attempt inside the MAC retry loop — this is how
    tests inject degradation that only *measurements* (not the physics-from-
    distance snapshot) can detect, which is exactly what MP-CARE's triggers
    must react to.
    """

    def __init__(self, params: NetworkParams | None = None):
        self.p = params or NetworkParams()
        self.rng = random.Random(self.p.rng_seed)
        self._link_loss: dict[frozenset, float] = {}
        # per-link shadow offset state, dB (log-normal -> Gaussian in dB).
        # x_{t+1} = rho*x_t + sqrt(1-rho^2)*sigma*eps, rho = exp(-dt/tau):
        # ensemble std sigma at all times, correlation rho between ticks.
        self._shadow: dict[frozenset, float] = {}
        self.dcf: DcfChannel | None = (
            DcfChannel(self.rng, DcfParams(
                cs_range_m=self.p.dcf_cs_range_m,
                bg_frames_per_s=self.p.dcf_bg_frames_per_s,
                bg_frame_bytes=self.p.dcf_bg_frame_bytes))
            if self.p.dcf_enabled else None)

    def begin_tick(self, t: float, positions: dict) -> None:
        """Advance the DCF scheduler to the current protocol tick."""
        if self.dcf is not None:
            self.dcf.begin_tick(t, positions)

    def set_link_loss(self, a: int, b: int, loss: float):
        """Inject extra per-attempt loss on the pair {a, b} (0..1)."""
        self._link_loss[frozenset((a, b))] = float(loss)

    def clear_link_losses(self):
        self._link_loss.clear()

    def link_loss(self, a: int | None, b: int | None) -> float:
        if a is None or b is None:
            return 0.0
        return self._link_loss.get(frozenset((a, b)), 0.0)

    def reseed(self, seed: int):
        self.p.rng_seed = seed
        self.rng = random.Random(seed)
        self._shadow.clear()
        if self.dcf is not None:
            self.dcf.rng = self.rng
            self.dcf._nav.clear()
            self.dcf._frames.clear()

    def step_shadow(self, pairs: list[tuple[int, int]], dt: float):
        """Advance the Gauss-Markov shadow state one tick for every pair
        currently in range (new pairs start at ~N(0, sigma) via a fresh
        sample). No-op when sigma == 0; the RNG stream is untouched so the
        deterministic-LoS baseline matches the historical runs exactly."""
        sig = self.p.shadow_sigma_db
        if sig <= 0.0 or not pairs:
            return
        rho = math.exp(-dt / self.p.shadow_tau_s)
        amp = sig * math.sqrt(1.0 - rho * rho)
        for key in map(frozenset, pairs):
            prev = self._shadow.get(key, 0.0)
            self._shadow[key] = rho * prev + amp * self.rng.gauss(0.0, 1.0)

    def unicast_success(self, d_m: float, a: int | None = None,
                        b: int | None = None,
                        size_bytes: int = 0) -> bool:
        """One unicast delivery. MAC retries act on the PHYSICAL per-attempt
        loss only (data+ACK, ~PRR^2); `injected_loss` and any per-pair
        impairment apply at the PACKET level (to the burst as a whole).

        Rationale: "packet loss" in the literature and in our Phase 4 sweeps
        means per-packet drop rate. Per-attempt injection would be absorbed
        by the retry loop (30% injected -> 0.3% packet loss), making loss
        sweeps and the T1 trigger test meaningless.

        When `dcf_enabled` is set, the retry loop is replaced by the
        frame-level DCF scheduler in `network_model.dcf`; the packet-level
        injected-loss terms still apply once, after a successful exchange."""
        pr = self._prr_attempt(d_m, a, b)
        if pr <= 0.0 or d_m > self.p.max_range_m:
            return False
        p1 = pr * pr
        if self.dcf is not None and a is not None and b is not None:
            ok = self.dcf.attempt(a, b, p1, size_bytes)
            if not ok:
                return False
            return self.rng.random() < ((1.0 - self.p.injected_loss)
                                        * (1.0 - self.link_loss(a, b)))
        success_p = (1.0 - (1.0 - p1) ** (self.p.mac_retries + 1)) \
            * (1.0 - self.p.injected_loss) * (1.0 - self.link_loss(a, b))
        return self.rng.random() < success_p

    def broadcast_success(self, dist_by_rx: dict[int, float], a: int,
                          size_bytes: int = 0) -> dict[int, bool]:
        """One broadcast transmission, per-receiver outcome.

        Under DCF this is a single frame (no ACK, no retry) scheduled once,
        with the physical reception drawn independently per receiver.
        Without DCF it falls back to the legacy per-neighbour unicast draws,
        preserving historical behaviour exactly.
        """
        if self.dcf is None:
            return {rx: self.unicast_success(d, a, rx, size_bytes=size_bytes)
                    for rx, d in dist_by_rx.items()}
        prrs = {rx: self._prr_attempt(d, a, rx) for rx, d in dist_by_rx.items()}
        out = self.dcf.attempt_broadcast(a, prrs, size_bytes)
        keep = (1.0 - self.p.injected_loss)
        return {rx: bool(ok and self.rng.random()
                         < keep * (1.0 - self.link_loss(a, rx)))
                for rx, ok in out.items()}

    def _prr_attempt(self, d_m: float, a: int | None, b: int | None) -> float:
        """Per-attempt PRR: the physical sigmoid, shifted by the pair's
        current shadow offset (dB) when shadowing is active. The offset is
        the tick's slow state and persists across the MAC retry attempts,
        as log-normal shadowing does over sub-second timescales. Pairs with
        no (a, b) context fall back to the physical PRR."""
        pr_base = prr(d_m, self.p)
        if not self._shadow or a is None or b is None:
            return pr_base
        if d_m > self.p.max_range_m or pr_base <= 0.0:
            return pr_base
        off = self._shadow.get(frozenset((a, b)), 0.0)
        s = snr_db(d_m, self.p) + off
        return 1.0 / (1.0 + math.exp(-self.p.k_per_db * (s - self.p.snr50_db)))

    def latency_ms(self, d_m: float) -> float:
        """Transmission latency: airtime + propagation + injected latency.
        airtime_ms = 8000 bits / (bandwidth_mbps * 1000)  (~0.73 ms for 1 kB @ 11 Mbps)."""
        if d_m > self.p.max_range_m:
            return math.inf
        airtime_ms = 8000.0 / (self.p.bandwidth_mbps * 1000.0)
        prop_ms = d_m / 3e8 * 1000.0
        return airtime_ms + prop_ms + self.p.injected_latency_ms


class DropTailQueue:
    """Fixed-size drop-tail queue (plan section 3.3 'queue model')."""

    def __init__(self, size: int = 64):
        self.size = size
        self._items: list = []
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def depth(self) -> int:
        return len(self._items)

    def push(self, item) -> bool:
        if len(self._items) >= self.size:
            self.dropped += 1
            return False  # tail drop
        self._items.append(item)
        return True

    def pop(self):
        return self._items.pop(0) if self._items else None


class NetworkSnapshot:
    """One topology evaluation: pairwise distances/ETX for the current positions.

    This is what the ROS network-model node publishes as /swarm/topology and
    what the agents' neighbor tables consume.
    """

    def __init__(self, positions: dict[int, tuple[float, float, float]],
                 link: LinkModel, t_s: float):
        self.t_s = t_s
        self.link = link
        p = link.p
        self.ids = sorted(positions)
        self.dist: dict[tuple[int, int], float] = {}
        self.etx: dict[tuple[int, int], float] = {}
        for i, a in enumerate(self.ids):
            for b in self.ids[i + 1:]:
                d = math.dist(positions[a], positions[b])
                self.dist[(a, b)] = d
                self.dist[(b, a)] = d
                e = etx(d, p)
                self.etx[(a, b)] = e
                self.etx[(b, a)] = e

    def neighbors(self, uav_id: int) -> list[int]:
        """IDs of UAVs with a finite-ETX link to uav_id."""
        return [b for (a, b), e in self.etx.items()
                if a == uav_id and math.isfinite(e)]

    def etx_of(self, a: int, b: int) -> float:
        return self.etx.get((a, b), math.inf)
