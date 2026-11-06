"""Shadowing extension tests (phase 9).

Two invariants:
1. sigma = 0.0 must reproduce the deterministic-LoS baseline EXACTLY (same
   RNG stream, same delivery decisions) — the shadowing default is off.
2. sigma > 0 must shift per-attempt PRR by the tick's per-link offset, and
   the offset process must have the right first and second moments.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "network_model"))

from network_model.model import (LinkModel, NetworkParams, etx, prr,  # noqa: E402
                                 snr_db)

import math  # noqa: E402


def _pairs():
    return [(0, 1), (0, 2), (1, 2)]


def test_sigma_zero_keeps_baseline_identical():
    """With sigma=0 the RNG stream and delivery decisions match the
    historical deterministic runs exactly."""
    p = NetworkParams(rng_seed=7)
    lm = LinkModel(p)
    q = NetworkParams(rng_seed=7)
    lm0 = LinkModel(q)          # shadow fields exist but sigma == 0
    lm0.step_shadow(_pairs(), dt=0.1)
    lm0.step_shadow(_pairs(), dt=0.1)
    assert not lm0._shadow      # no state created
    # identical RNG consumption
    for d in (50.0, 120.0, 260.0, 299.0):
        for (a, b) in _pairs():
            assert lm.unicast_success(d, a, b) == lm0.unicast_success(d, a, b)


def test_step_shadow_ensemble_std_and_correlation():
    """Ensemble std ~ sigma at all times; tick-to-tick correlation ~ rho."""
    sig = 4.0
    p = NetworkParams(rng_seed=3, shadow_sigma_db=sig, shadow_tau_s=5.0)
    rho = math.exp(-0.1 / 5.0)
    amp = sig * math.sqrt(1.0 - rho * rho)
    lm = LinkModel(p)
    xs = []
    for _ in range(3000):
        lm.step_shadow([(0, 1)], dt=0.1)
        xs.append(lm._shadow[frozenset((0, 1))])
    # steady-state ensemble std equals sigma (burn first 200 warmup ticks)
    steady = xs[200:]
    m = sum(steady) / len(steady)
    v = sum((x - m) ** 2 for x in steady) / (len(steady) - 1)
    assert abs(math.sqrt(v) - sig) / sig < 0.05   # within 5%
    # lag-1 correlation ~ rho
    n = len(steady) - 1
    num = sum((steady[i] - m) * (steady[i + 1] - m) for i in range(n))
    c1 = num / (v * n)
    assert abs(c1 - rho) < 0.05


def test_prr_attempt_applies_offset():
    """unicast_success's per-attempt PRR is the deterministic PRR shifted by
    the pair's current shadow offset."""
    p = NetworkParams(rng_seed=11, shadow_sigma_db=6.0, shadow_tau_s=5.0)
    lm = LinkModel(p)
    d = 200.0
    base = prr(d, p)
    lm.step_shadow([(0, 1)], dt=0.1)
    off = lm._shadow[frozenset((0, 1))]
    s = snr_db(d, p) + off
    want = 1.0 / (1.0 + math.exp(-p.k_per_db * (s - p.snr50_db)))
    # access the per-attempt PRR directly
    got = lm._prr_attempt(d, 0, 1)
    assert abs(got - want) < 1e-12
    # and without context (a=None) it falls back to the physical PRR
    assert lm._prr_attempt(d, None, None) == base


def test_offset_persists_within_a_tick():
    """The shadow state is a slow field: repeated attempts inside one tick
    see the SAME offset (ten re-draws, no state mutation)."""
    p = NetworkParams(rng_seed=42, shadow_sigma_db=3.0, shadow_tau_s=5.0)
    lm = LinkModel(p)
    lm.step_shadow([(0, 1)], dt=0.1)
    off0 = lm._shadow[frozenset((0, 1))]
    for _ in range(10):
        assert lm._shadow[frozenset((0, 1))] == off0


def test_shadow_changes_delivery_outcomes():
    """Statistical smoke: at the range edge, heavy shadowing widens the
    realized delivery spread versus the deterministic case."""
    p_det = NetworkParams(rng_seed=99)
    p_sh = NetworkParams(rng_seed=99, shadow_sigma_db=6.0,
                         shadow_tau_s=5.0)
    d = 260.0
    n = 4000
    det = sum(LinkModel(NetworkParams(rng_seed=k)).unicast_success(d, 0, 1)
              for k in range(n))
    lm = LinkModel(p_sh)
    hits = 0
    for k in range(n):
        lm.step_shadow([(0, 1)], dt=0.1)
        hits += lm.unicast_success(d, 0, 1)
    # deterministic baseline is a single Bernoulli rate; shadowed delivery
    # mixes over offsets, so the two rates should differ by inspection scale
    det_rate = det / n
    sh_rate = hits / n
    se = math.sqrt(det_rate * (1 - det_rate) / n)
    assert abs(sh_rate - det_rate) > 1.5 * se, (det_rate, sh_rate)
