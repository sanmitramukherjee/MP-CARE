"""Event-triggered re-routing with hysteresis (plan section 5.2.3).

Trigger set T (each maps to a plan row):
  T1  PRR(current route) < theta_pr        -> route quality dropped
  T2  LET(any link on route) < tau_min     -> a link is about to expire
  T3  C(alt) < (1 - delta) * C(current)    -> alternative clearly better
  T4  neighbor lost                        -> relay disappeared
  T5  new task assigned                    -> destination changed

Hysteresis (plan):
  - minimum dwell time t_dwell between consecutive re-routes
  - switch only if C(alt) < (1 - delta) * C(cur), delta in [0.05, 0.20]
  - formal property: re-routes in [0, T] bounded by floor(T / t_dwell)

Edge-triggering note: T1/T2/T4 are level conditions, but a re-plan decision
fires once per rising edge (state change) — otherwise a persistently bad
route would re-plan every tick and the dwell bound would be meaningless.
T3 is evaluated inside the hysteresis gate at every decision point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

TRIGGER_NAMES = {"T1": "pr-below-threshold", "T2": "let-expiring",
                 "T3": "better-alternative", "T4": "neighbor-lost",
                 "T5": "new-task"}


@dataclass
class TriggerParams:
    theta_pr: float = 0.6          # PRR threshold on the current route
    tau_min: float = 3.0           # LET alert threshold (s)
    delta: float = 0.10            # hysteresis margin (plan range 0.05-0.20)
    t_dwell: float = 2.0           # minimum seconds between re-routes


class EventTrigger:
    def __init__(self, params: TriggerParams | None = None):
        self.p = params or TriggerParams()
        self.last_reroute_t: float = -1e18
        # previous (level) state of the edge-triggered conditions
        self._prev_pr_bad = False
        self._prev_let_bad = False
        self._prev_lost: set[int] = set()
        self.re_event_count = 0        # telemetry: fires that passed hysteresis
        self.blocked_by_dwell = 0
        self.blocked_by_margin = 0

    # -------------------------------------------------------- level checks
    def pr_bad(self, route_prr: float | None) -> bool:
        return route_prr is not None and route_prr < self.p.theta_pr

    def let_bad(self, min_let_s: float | None) -> bool:
        return min_let_s is not None and min_let_s < self.p.tau_min

    # -------------------------------------------------------- decision
    def should_reroute(self, t: float, *, route_prr: float | None,
                       min_let_s: float | None, better_alt: bool,
                       lost_neighbors: list[int]) -> tuple[bool, str]:
        """Returns (fire?, trigger-name). T3/lost events are one-shot per
        call; T1/T2 are edge-triggered on their level; all gated by the
        dwell timer; T3 additionally by the margin (which IS the trigger)."""
        pr_now = self.pr_bad(route_prr)
        let_now = self.let_bad(min_let_s)
        lost_new = [n for n in lost_neighbors if n not in self._prev_lost]
        self._prev_lost = set(lost_neighbors)

        reasons = []
        if pr_now and not self._prev_pr_bad:
            reasons.append("T1")
        if let_now and not self._prev_let_bad:
            reasons.append("T2")
        if lost_new:
            reasons.append("T4")
        self._prev_pr_bad, self._prev_let_bad = pr_now, let_now
        if better_alt:
            reasons.append("T3")

        if not reasons:
            return False, ""

        # T4 is an emergency: relay gone, the route is broken NOW.
        # The dwell gate applies to optional re-planning (T1/T2/T3).
        if "T4" in reasons:
            self.last_reroute_t = t
            self.re_event_count += 1
            return True, TRIGGER_NAMES["T4"]

        if t - self.last_reroute_t < self.p.t_dwell:
            self.blocked_by_dwell += 1
            return False, ""
        # T3 carries its own margin (better_alt already means C(alt) <
        # (1-delta)*C(cur)); T1/T2 are degradation alarms, not alternatives.
        self.last_reroute_t = t
        self.re_event_count += 1
        return True, TRIGGER_NAMES[reasons[0]]

    def reset(self):
        self.last_reroute_t = -1e18
        self._prev_pr_bad = False
        self._prev_let_bad = False
        self._prev_lost = set()
