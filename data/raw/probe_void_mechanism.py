#!/usr/bin/env python3
"""Throwaway probe (data/raw): diagnose MP-CARE's PDR deficit at scale BEFORE
designing the perimeter-recovery fix (no-hallucination rule: measure the
mechanism first).

Wraps MpcareRouter.next_hop_for with a counting subclass injected into
run_sim, then re-runs exact phase6v1 trials (same seeds, scale scenario):
  total_calls      : next_hop_for invocations (packet-hop decisions)
  none_returns     : returned None (packet stalled this tick)
  isolated_none    : None with EMPTY neighbor table (physically alone -> no
                     protocol can help; overlap with the 6 disconnected seeds)
  void_none        : None with neighbors PRESENT but none strictly closer to
                     the destination (greedy local minimum -> recoverable by
                     perimeter-style walking IF the local mesh continues)
Also: share of none-returns inside the static window (t < 30 s).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/network_model", "src/mpcare_agent", "scripts"):
    sys.path.insert(0, str(ROOT / p))

import run_sim  # noqa: E402
from mpcare_agent.router import MpcareRouter  # noqa: E402


class InstrumentedRouter(MpcareRouter):
    STATS = {"calls": 0, "none": 0, "isolated": 0, "void": 0, "none_t30": 0}

    def next_hop_for(self, dst: int, t: float):
        InstrumentedRouter.STATS["calls"] += 1
        r = super().next_hop_for(dst, t)
        if r is None:
            s = InstrumentedRouter.STATS
            s["none"] += 1
            if t < run_sim.HOVER_UNTIL:
                s["none_t30"] += 1
            if not self.table.ids():
                s["isolated"] += 1
            else:
                s["void"] += 1
        return r


run_sim.MpcareRouter = InstrumentedRouter

CASES = [(3002, 30), (3001, 50), (3000, 50), (3003, 50), (3022, 50)]
for seed, n in CASES:
    InstrumentedRouter.STATS = dict.fromkeys(InstrumentedRouter.STATS, 0)
    r = run_sim.run(seed, ROOT / "data" / "raw", duration=150.0,
                    write_csv=False, algorithm="mpcare", scenario="scale",
                    n_uav=n)
    s = InstrumentedRouter.STATS
    pct = (lambda k: f"{100.0 * s[k] / s['none']:.0f}%" if s["none"] else "n/a")
    print(f"seed={seed} N={n:<3} pdr={r['pdr']:.3f} deg={r['avg_degree']} | "
          f"calls={s['calls']} none={s['none']} "
          f"(isolated {pct('isolated')}, void {pct('void')}), "
          f"none during t<30s: {s['none_t30']}")
