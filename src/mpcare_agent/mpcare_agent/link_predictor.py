"""Mobility-predictive link cost (plan section 5.2.2, formulas verbatim).

ETX prediction uses the radial velocity measured by the NeighborTable:

    ETX_pred(e_ij, tau) = ETX_current(e_ij) * f(d_dot_ij)
    f(v) = 1 + beta * tanh(gamma * v)      (<1 approaching, >1 receding)

Link Expiration Time:

    LET_ij = (R_max - d_ij) / |d_dot_ij|    (inf when approaching, d_dot <= 0)

The ETX + LET combination exists at the network layer (P-OLSR lineage); the
claimed contribution is using it in the navigation loop (plan section 5.2.2
note) — the planner scores routes with the link state expected at arrival,
not the snapshot.
"""

from __future__ import annotations

import math


class LinkPredictor:
    def __init__(self, max_range_m: float, beta: float = 1.0, gamma: float = 0.15):
        self.max_range_m = max_range_m
        self.beta = beta
        self.gamma = gamma

    def f_velocity(self, radial_vel_mps: float) -> float:
        """Velocity scaling factor f(v): <1 approaching, >1 receding."""
        return 1.0 + self.beta * math.tanh(self.gamma * radial_vel_mps)

    def etx_pred(self, etx_current: float, radial_vel_mps: float,
                 expected_arrival_s: float = 0.0) -> float:
        """Predicted ETX of a link with the given current ETX and radial velocity.

        expected_arrival_s > 0 additionally projects how much the distance will
        have changed by the time traffic reaches this hop (plan note: 'link
        quality at the expected arrival time'), using the same f() over the
        velocity projected forward.
        """
        if math.isinf(etx_current):
            return math.inf
        f = self.f_velocity(radial_vel_mps)
        if expected_arrival_s > 0.0:
            # degradation/projection accrues over the time traffic takes to arrive
            f = f ** (1.0 + min(3.0, expected_arrival_s))
        return etx_current * f

    def let_s(self, dist_m: float, radial_vel_mps: float) -> float:
        """Link expiration time in seconds (inf while approaching)."""
        if radial_vel_mps <= 0.0:
            return math.inf
        if dist_m >= self.max_range_m:
            return 0.0
        return (self.max_range_m - dist_m) / radial_vel_mps
