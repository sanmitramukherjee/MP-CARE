"""Tests for Phase 5 Extension A: ORCA + connectivity penalty (plan §7.1).

Pure-Python module tests (Windows, no ROS): the same code the WSL/Linux
benchmark executes. Exit criteria under test (plan §7.3):
  - connectivity preservation: connectivity(ORCA+penalty) > connectivity(ORCA)
  - safety maintained: collision rate(ORCA+penalty) <= collision rate(ORCA)
plus determinism and edge cases. The full corridor benchmark lives in
scripts/run_phase5.py; these tests pin the mechanism directly.
"""
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/orca_avoidance", "src/network_model"):
    sys.path.insert(0, str(ROOT / p))

from orca_avoidance.core import (connect_penalty, constrain, connectivity_ok,
                                 orca_step, orca_step_comms)  # noqa: E402


def test_penalty_is_zero_when_inside_range():
    me, nbr = (0.0, 0.0), (200.0, 0.0)
    out = connect_penalty((1.0, 0.0), {1: nbr}, me, 300.0, lam=0.8)
    assert out == (1.0, 0.0)            # 200 + 1*dt << 300: no pull


def test_penalty_pulls_back_toward_edge_leaving_range():
    me, nbr = (0.0, 0.0), (299.0, 0.0)
    # moving AWAY from the neighbor (v.x = -2): d_pref = 299 - (-2) = 301 > 300
    # -> pull toward +x (the neighbor) cancels part of the recession
    out = connect_penalty((-2.0, 0.0), {1: nbr}, me, 300.0, lam=0.5)
    assert math.isclose(out[0], -2.0 + 0.5 * (301.0 - 300.0), abs_tol=1e-9)
    # approaching the neighbor (d_pref shrinks) must never trigger the penalty
    assert connect_penalty((2.0, 0.0), {1: nbr}, me, 300.0, lam=0.5) == (2.0, 0.0)


def test_penalty_zero_when_lambda_zero_baseline_exact():
    me, nbr = (0.0, 0.0), (299.0, 0.0)
    a = connect_penalty((2.0, 1.0), {1: nbr}, me, 300.0, lam=0.0)
    assert a == (2.0, 1.0)


def test_constrain_allows_receding_motion():
    # moving apart: constraint must never touch the velocity
    v = (1.0, 0.0)
    out = constrain(v, (0.0, 0.0), (10.0, 0.0), (1.0, 0.0), dt=0.1, tau=0.5, R=1.0)
    assert out == v


def test_constrain_stops_head_on_before_collision():
    # closing head-on at 4 m/s combined from 1 m apart: both sides apply half
    # the correction; the predicate holds with BOTH adjusted velocities
    out_a = constrain((2.0, 0.0), (0.0, 0.0), (1.0, 0.0), (-2.0, 0.0),
                      dt=0.1, tau=0.5, R=1.0)
    out_b = constrain((-2.0, 0.0), (1.0, 0.0), (0.0, 0.0), (2.0, 0.0),
                      dt=0.1, tau=0.5, R=1.0)
    assert connectivity_ok(out_a, (0.0, 0.0), (1.0, 0.0), out_b, 0.1, 0.5, 1.0)
    assert out_a[0] < 2.0 and out_b[0] > -2.0    # approaches softened


def test_constrain_applies_exactly_half_the_required_boost():
    """Pair predicate holds only when BOTH sides apply their half (each
    computing u against the other's pre-adjustment velocity) — verified by
    hand: d=0.6, closing at 1.0 m/s relative, tau=0.5, R=1 -> u=1.3 each,
    both slow by 0.65 along the axis -> horizon distance = R exactly."""
    va = constrain((1.0, 1.0), (0.0, 0.0), (0.6, 0.0), (0.5, 0.0), 0.1, 0.5, 1.0)
    vb = constrain((0.5, 0.0), (0.6, 0.0), (0.0, 0.0), (1.0, 1.0), 0.1, 0.5, 1.0)
    assert connectivity_ok(va, (0.0, 0.0), (0.6, 0.0), vb, 0.1, 0.5, 1.0)


def test_orca_step_no_overlap_one_tick():
    pos = {0: (0.0, 0.0), 1: (0.6, 0.0)}
    vp = {0: (1.0, 0.0), 1: (-1.0, 0.0)}
    vc = {0: (1.0, 0.0), 1: (-1.0, 0.0)}
    out = orca_step(2, pos, vp, vc, dt=0.1, tau_factor=5.0)
    # tau-horizon separation must hold for both
    for a in (0, 1):
        assert connectivity_ok(out[a], pos[a], pos[1 - a], out[1 - a], 0.1, 0.5, 1.0)


def test_comms_variant_preserves_safety():
    pos = {0: (0.0, 0.0), 1: (0.6, 0.0), 2: (50.0, 0.0)}
    vp = {0: (1.0, 0.0), 1: (-1.0, 0.0), 2: (-1.0, 0.0)}
    vc = dict(vp)
    nbrs = {0: {2: pos[2]}, 2: {0: pos[0]}}   # 0-2 is the routing pair near range
    out = orca_step_comms(3, pos, vp, vc, nbrs, max_range_m=60.0, lam=0.8)
    for a in (0, 1):
        assert connectivity_ok(out[a], pos[a], pos[1 - a], out[1 - a], 0.1, 0.5, 1.0)


def test_comms_penalty_holds_link_that_plain_orca_breaks():
    """The §7.1 mechanism end-to-end: routing pair at d0=290 m (inside the
    300 m range); agent 1's preferred velocity recedes at 15 m/s. Plain
    ORCA (no collision risk) lets it fly: d crosses 300 m and the link
    dies. With the penalty, agent 1's velocity is pulled to the stable
    fixed point v* = 300 - d (d approaches 300 strictly from below, so the
    link never breaks). 2 s rollout, per-tick safety asserts."""
    dt, tau, rng = 0.1, 0.5, 300.0
    pos0 = {0: (0.0, 0.0), 1: (290.0, 0.0)}
    vp0 = {0: (0.0, 0.0), 1: (15.0, 0.0)}

    def rollout(comms: bool):
        p = dict(pos0)
        v = {i: (0.0, 0.0) for i in pos0}
        for _ in range(20):
            if comms:
                nbrs = {1: {0: p[0]}, 0: {1: p[1]}}   # routing neighbors
                v = orca_step_comms(2, p, vp0, v, nbrs, max_range_m=rng,
                                    lam=2.0, max_speed=20.0, dt=dt)
            else:
                v = orca_step(2, p, vp0, v, max_speed=20.0, dt=dt)
            assert connectivity_ok(v[0], p[0], p[1], v[1], dt, tau, 1.0)
            p = {i: (p[i][0] + v[i][0] * dt, p[i][1] + v[i][1] * dt)
                 for i in p}
        return math.dist(p[0], p[1])

    d_plain = rollout(comms=False)
    d_comms = rollout(comms=True)
    assert d_plain > rng                      # plain ORCA: link broken
    assert d_comms <= rng                     # comms variant: link held
    assert d_comms < d_plain


def test_determinism_same_inputs_same_output():
    pos = {0: (0.0, 0.0), 1: (0.6, 0.0), 2: (5.0, 5.0)}
    vp = {i: (1.0, 0.5) for i in pos}
    vc = {i: (0.0, 0.0) for i in pos}
    nbrs = {i: {j: pos[j] for j in pos if j != i} for i in pos}
    a = orca_step_comms(3, pos, vp, vc, nbrs)
    b = orca_step_comms(3, pos, vp, vc, nbrs)
    assert a == b
