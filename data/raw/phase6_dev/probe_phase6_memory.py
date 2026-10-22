#!/usr/bin/env python3
"""Throwaway probe (data/raw, NOT part of the campaign): memory-model sanity.

Q1: does MpcareRouter per-node work grow with N at fixed density (plan 8.3
claims O(deg))? Measure per-router tick+beacon time at N=10/30/50 in the
S2/S4/S6 boxes at the plan's density (area/N held at ~57.8 km^2/UAV).
Q2: what does a REDUCED N=50 trial cost end-to-end (to extrapolate the
campaign runtime before committing to a trial count)?
"""
from __future__ import annotations

import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in ("src/network_model", "src/mpcare_agent", "scripts"):
    sys.path.insert(0, str(ROOT / p))

from network_model.model import LinkModel, NetworkParams, NetworkSnapshot  # noqa: E402
from mpcare_agent.router import MpcareRouter, SimHooks  # noqa: E402
import run_sim  # noqa: E402

BOXES = {10: (700.0, 700.0), 30: (1200.0, 1200.0), 50: (1700.0, 1700.0)}

print("== Q1: per-router tick+beacon cost vs N (same density) ==")
prev = None
for n, box in BOXES.items():
    params = NetworkParams(rng_seed=7)
    link = LinkModel(params)
    m = random.Random(7)
    pos = {i: (m.uniform(50, box[0] - 50), m.uniform(50, box[1] - 50), 40.0)
           for i in range(n)}
    routers = [MpcareRouter(i, n, max_range_m=params.max_range_m) for i in range(n)]
    t_ticks = t_beacon = 0.0
    REPS = 60
    for rep in range(REPS):
        t = rep * 1.0
        snap = NetworkSnapshot(pos, link, t)
        links = {u: set(snap.neighbors(u)) for u in pos}
        hooks = SimHooks(pos, link, snap)
        t0 = time.perf_counter()
        for r in routers:
            r.bind(hooks)
            r._own_queue_ratio = 0.0
            r.tick(t)
        t_ticks += time.perf_counter() - t0
        t0 = time.perf_counter()
        for r in routers:
            payload = r.beacon_payload(pos[r.node_id], r._own_queue_ratio, t)
            for nb in sorted(links.get(r.node_id, ())):
                d = math.dist(pos[r.node_id][:2], pos[nb][:2])
                if link.unicast_success(d, r.node_id, nb):
                    routers[nb].receive_beacon(payload, r.node_id,
                                               snap.etx_of(r.node_id, nb), t)
        t_beacon += time.perf_counter() - t0
    deg = sum(len(r.table.ids()) for r in routers) / n
    per = (t_ticks + t_beacon) / REPS / n * 1000.0
    ratio = f"  x{per / prev:.2f} vs prev" if prev else ""
    print(f"N={n:<3} avg_hood={deg:5.1f}  per-router tick+beacon "
          f"{per:6.3f} ms{ratio}")
    prev = per

print("\n== Q2: reduced full-trial cost at N=50 (mpcare + gpsr) ==")
for algo in ("mpcare", "gpsr"):
    t0 = time.perf_counter()
    r = run_sim.run(7000, ROOT / "data" / "raw", duration=30.0,
                    write_csv=False, algorithm=algo, scenario="scale",
                    n_uav=50)
    dt = time.perf_counter() - t0
    print(f"{algo:<7} 30s-sim N=50: {dt:6.1f}s wall "
          f"(-> 90s sim ~= {dt * 3:.0f}s) pdr={r['pdr']} "
          f"conv={r.get('full_conv_s')} deg={r.get('avg_degree')} "
          f"beacons={r.get('beacons_per_node_s')}")
