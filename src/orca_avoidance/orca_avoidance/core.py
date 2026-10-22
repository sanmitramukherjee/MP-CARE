"""Communication-aware collision avoidance — Phase 5 Extension A (plan §7.1).

Base algorithm: ORCA (Optimal Reciprocal Collision Avoidance), van den Berg
et al., "Reciprocal n-body Collision Avoidance" (ISRR 2011). Implementation
honesty: this module implements the 1-D (line-of-sight) projection of the
ORCA constraint — each pair shares responsibility 50/50 for keeping their
tau-horizon projected positions outside the combined-radius disc, applied
along the separation axis. This preserves ORCA's guarantees on the
separation axis (the direction that matters for head-on/convoy conflicts on
corridors) but is NOT the full 2-D half-plane ORCA; the paper must describe
it as an axis-projected ORCA variant. Pure and deterministic: identical
inputs -> identical velocities, no RNG, no ROS.

Connectivity penalty (the §7.1 extension): the preferred velocity is
v_pref + lambda * sum over routing neighbors of the outward unit vector
scaled by max(0, d_ij(v_pref) - R_safe) — i.e. steer back toward any
routing neighbor the preferred velocity would push beyond the safe range.
Constraints are still enforced AFTER the preference, so safety (the
benchmark's second criterion) cannot regress: the ORCA step bounds how much
any connectivity pull can change the velocity, per pair, per tick.Discrete-time form (the standard ORCA derivation): with horizon tau = 5 dt,
the pair (i, j) must satisfy d_ij(t+tau) >= R, i.e. the separation rate
satisfies d_dot >= (R - d_ij)/tau; each side applies half the required boost
(reciprocity). See constrain() and connectivity_ok().
R_safe = 1.0 m (collision threshold), plan default."""

from __future__ import annotations

import math


def unit(vx: float, vy: float) -> tuple[float, float]:
    d = math.hypot(vx, vy)
    if d < 1e-12:
        return (0.0, 0.0)
    return (vx / d, vy / d)


def connect_penalty(v_pref: tuple[float, float],
                    neighbors: dict[int, tuple[float, float]],
                    my_pos: tuple[float, float],
                    max_range_m: float,
                    lam: float = 0.8,
                    lookahead_s: float = 1.0) -> tuple[float, float]:
    """§7.1 soft constraint: pull v_pref toward routing neighbors that the
    preferred velocity would push beyond max_range. lambda=0 reproduces the
    plain-ORCA baseline exactly.

    Sign convention (verified by test): d/dt(dist to neighbor) =
    -(v . n) with n = unit(neighbor - me), so the predicted distance at the
    model lookahead is d_pref = d - (v . n) * lookahead_s; a pull fires when
    d_pref > max_range, and it adds velocity ALONG +n (toward the neighbor),
    opposing the recession.

    lookahead_s: how far ahead the link-loss prediction runs (the §7.1 argmin
    evaluates d_ij(v) at the velocity the agent is ABOUT to fly). 1.0 s =
    one-step default (kinematic prediction); the corridor benchmark uses
    3.0 s (model lookahead) because at UAV cruise speeds a 1 s horizon only
    reacts once the link is already at the range edge — measured +2.5% rel
    connectivity in the first draft (below the 5% criterion), +lookahead
    documented in NOTES/Phase_5.md."""
    vx, vy = v_pref
    t_arr = float(lookahead_s)
    for nid, npos in neighbors.items():
        cur = math.dist(my_pos, npos)
        nx, ny = unit(npos[0] - my_pos[0], npos[1] - my_pos[1])
        d_pref = cur - (vx * nx + vy * ny) * t_arr
        over = max(0.0, d_pref - max_range_m)
        if over > 0.0:
            vx += lam * over * nx
            vy += lam * over * ny
    return (vx, vy)


def constrain(v: tuple[float, float],
              my_pos: tuple[float, float],
              other_pos: tuple[float, float],
              v_other: tuple[float, float],
              dt: float, tau: float, R: float) -> tuple[float, float]:
    """One ORCA constraint (axis form): keep the tau-horizon pair distance
    >= R. With n = unit(other - me) the separation rate is
    d_dot = (v_other - v) . n, so d(t+tau) = d + d_dot * tau >= R requires
        d_dot >= (R - d) / tau.
    When violated, u = (R-d)/tau - d_dot > 0 is the required separation
    boost; reciprocity means THIS agent applies half of it (the other half
    is the pair partner's duty, applied in its own constraint).
    """
    ex, ey = unit(other_pos[0] - my_pos[0], other_pos[1] - my_pos[1])
    d = math.dist(my_pos, other_pos)
    d_dot = (v_other[0] - v[0]) * ex + (v_other[1] - v[1]) * ey
    u = (R - d) / tau - d_dot
    if u <= 0.0:
        return v                                   # constraint satisfied
    return (v[0] - 0.5 * u * ex, v[1] - 0.5 * u * ey)


def connectivity_ok(v: tuple[float, float], my_pos: tuple[float, float],
                    other_pos: tuple[float, float], v_other: tuple[float, float],
                    dt: float, tau: float, R: float) -> bool:
    """The benchmark's per-pair predicate (plan §7.1, criterion 1):
    the tau-horizon pair distance stays >= R given BOTH adjusted velocities."""
    ex, ey = unit(other_pos[0] - my_pos[0], other_pos[1] - my_pos[1])
    d = math.dist(my_pos, other_pos)
    d_dot = (v_other[0] - v[0]) * ex + (v_other[1] - v[1]) * ey
    return d + d_dot * tau >= R - 1e-9


def orca_step(i: int, positions: dict[int, tuple[float, float]],
              v_pref: dict[int, tuple[float, float]], v_cur: dict[int, tuple[float, float]],
              radius: float = 0.5, max_speed: float = 10.0,
              dt: float = 0.1, tau_factor: float = 5.0,
              passes: int = 3) -> dict[int, tuple[float, float]]:
    """One synchronous ORCA tick for all agents (deterministic id order).

    radius: per-agent physical radius (pairwise collision disc = 2*radius);
    R = 2*radius = 1.0 m = the plan's R_safe.

    passes: Gauss-Seidel sweeps over the pairwise constraints per tick.
    One pass is NOT a fixed point under multi-threat interference (a rear
    closing threat demands speed-up while frontal threats demand slow-down:
    corrections partially cancel and the horizon guarantee erodes tick over
    tick — measured in the cross benchmark, NOTES/Phase_5.md). Repeated
    sweeps re-apply each pair constraint against the CURRENT velocities and
    converge to the axis-projected fixed point; 3 passes hold the corridor
    scenario tick-safe. Single-pair geometries are unchanged (extra sweeps
    are no-ops once u <= 0)."""
    tau = tau_factor * dt
    R = 2.0 * radius
    v_out: dict[int, tuple[float, float]] = {}
    for a in sorted(positions):
        v = v_pref[a]
        for b in sorted(positions):
            if b == a:
                continue
            v = constrain(v, positions[a], positions[b],
                          v_cur[b], dt, tau, R)
        sp = math.hypot(*v)
        if sp > max_speed:
            v = (v[0] * max_speed / sp, v[1] * max_speed / sp)
        v_out[a] = v
    for _ in range(max(0, passes - 1)):       # Gauss-Seidel refinement
        for a in sorted(positions):
            v = v_out[a]
            for b in sorted(positions):
                if b == a:
                    continue
                v = constrain(v, positions[a], positions[b],
                              v_out[b], dt, tau, R)
            sp = math.hypot(*v)
            if sp > max_speed:
                v = (v[0] * max_speed / sp, v[1] * max_speed / sp)
            v_out[a] = v
    return v_out


def orca_step_comms(i: int, positions: dict[int, tuple[float, float]],
                    v_pref: dict[int, tuple[float, float]],
                    v_cur: dict[int, tuple[float, float]],
                    routing_neighbors: dict[int, dict[int, tuple[float, float]]],
                    max_range_m: float = 300.0, lam: float = 0.8,
                    radius: float = 0.5, max_speed: float = 10.0,
                    dt: float = 0.1, tau_factor: float = 5.0,
                    lookahead_s: float = 1.0, passes: int = 3
                    ) -> dict[int, tuple[float, float]]:
    """ORCA tick WITH the §7.1 connectivity penalty folded into v_pref
    (per agent, using that agent's routing-neighbor view — decentralized).
    Safety ordering is unchanged: penalties first, constraints after
    (`passes` Gauss-Seidel sweeps, see orca_step)."""
    tau = tau_factor * dt
    R = 2.0 * radius
    v_out: dict[int, tuple[float, float]] = {}
    for a in sorted(positions):
        vp = connect_penalty(v_pref[a], routing_neighbors.get(a, {}),
                             positions[a], max_range_m, lam, lookahead_s)
        v = vp
        for b in sorted(positions):
            if b == a:
                continue
            v = constrain(v, positions[a], positions[b],
                          v_cur[b], dt, tau, R)
        sp = math.hypot(*v)
        if sp > max_speed:
            v = (v[0] * max_speed / sp, v[1] * max_speed / sp)
        v_out[a] = v
    for _ in range(max(0, passes - 1)):
        for a in sorted(positions):
            v = v_out[a]
            for b in sorted(positions):
                if b == a:
                    continue
                v = constrain(v, positions[a], positions[b],
                              v_out[b], dt, tau, R)
            sp = math.hypot(*v)
            if sp > max_speed:
                v = (v[0] * max_speed / sp, v[1] * max_speed / sp)
            v_out[a] = v
    return v_out


class OrcaAgent:
    """One UAV's ORCA state for scenario integration (deterministic)."""

    def __init__(self, uav_id: int, radius: float = 0.5,
                 max_speed: float = 10.0, lam: float = 0.8,
                 max_range_m: float = 300.0, lookahead_s: float = 1.0):
        self.uav_id = uav_id
        self.radius = radius
        self.max_speed = max_speed
        self.lam = lam
        self.max_range_m = max_range_m
        self.lookahead_s = lookahead_s
        self.pos: tuple[float, float] = (0.0, 0.0)
        self.vel: tuple[float, float] = (0.0, 0.0)
        self.routing_neighbors: dict[int, tuple[float, float]] = {}

    def step(self, v_pref: tuple[float, float], positions, v_cur,
             dt: float = 0.1, tau_factor: float = 5.0):
        tau = tau_factor * dt
        R = 2.0 * self.radius
        v = connect_penalty(v_pref, self.routing_neighbors, self.pos,
                            self.max_range_m, self.lam,
                            self.lookahead_s) if self.routing_neighbors \
            else v_pref
        for b in sorted(positions):
            if b == self.uav_id:
                continue
            v = constrain(v, self.pos, positions[b], v_cur[b], dt, tau, R)
        sp = math.hypot(*v)
        if sp > self.max_speed:
            v = (v[0] * self.max_speed / sp, v[1] * self.max_speed / sp)
        self.vel = v
        self.pos = (self.pos[0] + v[0] * dt, self.pos[1] + v[1] * dt)
        return v
