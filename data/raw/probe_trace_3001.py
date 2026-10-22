#!/usr/bin/env python3
"""Throwaway trace: WHERE does a packet die on seed 3001 @ N=50 despite the
face walk? Log every mpcare forwarding decision for t in [5, 13] (packet 1's
whole life: created t=5, buffer 5 s -> dead by t=10) and reconstruct.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/network_model", "src/mpcare_agent", "scripts"):
    sys.path.insert(0, str(ROOT / p))

import run_sim  # noqa: E402
from mpcare_agent.router import MpcareRouter  # noqa: E402

LOG: list[str] = []
_orig = MpcareRouter.next_hop_for


def traced(self, dst, t, visited=None, prev=None):
    r = _orig(self, dst, t, visited=visited, prev=prev)
    if t < 13.0 and (self.node_id == 0 or (r is not None and LOG and LOG[-1].endswith(f"-> {self.node_id}"))):
        LOG.append(f"t={t:5.1f} node={self.node_id:<3} dst={dst} "
                   f"deg={self.table.ids().__len__():<2} "
                   f"visited={len(visited) if visited else 0:<3} "
                   f"-> {r}" + ("" if r is not None else "  (STALL)"))
    return r


MpcareRouter.next_hop_for = traced
r = run_sim.run(3001, ROOT / "data" / "raw", duration=150.0, write_csv=False,
                algorithm="mpcare", scenario="scale", n_uav=50)
print(f"pdr={r['pdr']} recovery={r['void_recovery_events']}")
print(f"log lines: {len(LOG)}")
for line in LOG[:90]:
    print(line)
