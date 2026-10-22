"""Per-UAV neighbor table for MP-CARE (plan section 5.3).

Maintains 1-hop neighbor knowledge from beacons: last-seen time, EWMA-smoothed
link ETX, velocity estimate (from consecutive positions), and radial velocity
(rate of change of distance — the input the LinkPredictor needs).

O(1) state per neighbor, no global topology table (plan section 5.2.4).
Also detects neighbor loss (T4): a neighbor heard in the past but silent for
NEIGHBOR_TIMEOUT beacons is reported as lost exactly once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

NEIGHBOR_TIMEOUT_S = 2.5      # ~2.5 beacon intervals: declared gone
EWMA_ALPHA = 0.3              # weight of the newest ETX sample


@dataclass
class NeighborInfo:
    uav_id: int
    last_seen_t: float
    last_pos: tuple[float, float, float]
    etx_ewma: float                      # smoothed ETX toward this neighbor
    prev_dist: float | None = None       # distance at the previous beacon
    radial_vel: float = 0.0              # + receding / - approaching (m/s)
    recent_delivery_rate: float = 1.0    # EWMA of our own deliveries to them
    last_advertised_nbrs: tuple = ()     # their neighbor list (beacon payload)
    queue_ratio: float = 0.0             # their queue fullness (beacon payload)


class NeighborTable:
    def __init__(self, uav_id: int):
        self.uav_id = uav_id
        self.neighbors: dict[int, NeighborInfo] = {}
        self.lost_neighbors: list[int] = []      # T4 events, drained by the trigger
        self._my_pos: tuple[float, float, float] | None = None

    def update_from_beacon(self, frm: int, pos: tuple[float, float, float],
                           etx_to_them: float, t: float):
        """One received beacon: refresh entry, update velocity + EWMA ETX.
        Radial velocity = rate of change of dist(me, them) between their last
        two beacons (positive = receding, negative = approaching)."""
        info = self.neighbors.get(frm)
        if info is None:
            self.neighbors[frm] = NeighborInfo(uav_id=frm, last_seen_t=t,
                                               last_pos=pos, etx_ewma=etx_to_them)
            return
        if self._my_pos is not None:
            d_prev = math.dist(self._my_pos[:2], info.last_pos[:2])
            d_now = math.dist(self._my_pos[:2], pos[:2])
            info.radial_vel = (d_now - d_prev) / max(1e-6, t - info.last_seen_t)
        info.last_pos = pos
        info.last_seen_t = t
        info.etx_ewma = (EWMA_ALPHA * etx_to_them
                         + (1.0 - EWMA_ALPHA) * info.etx_ewma)

    def set_my_position(self, pos: tuple[float, float, float]):
        self._my_pos = pos

    def observe_delivery(self, to_id: int, success: bool):
        """Our own data-plane experience with a neighbor (T1-style evidence)."""
        info = self.neighbors.get(to_id)
        if info is None:
            return
        target = 1.0 if success else 0.0
        info.recent_delivery_rate = 0.4 * target + 0.6 * info.recent_delivery_rate

    def tick(self, t: float):
        """Expire silent neighbors; report each loss once (T4)."""
        for nid in list(self.neighbors):
            if t - self.neighbors[nid].last_seen_t > NEIGHBOR_TIMEOUT_S:
                self.neighbors.pop(nid)
                self.lost_neighbors.append(nid)

    def ids(self) -> list[int]:
        return sorted(self.neighbors)

    def etx_to(self, nid: int) -> float:
        info = self.neighbors.get(nid)
        return info.etx_ewma if info else math.inf

    def radial_velocity(self, nid: int) -> float:
        info = self.neighbors.get(nid)
        return info.radial_vel if info else 0.0

    def drain_lost(self) -> list[int]:
        lost, self.lost_neighbors = self.lost_neighbors, []
        return lost
