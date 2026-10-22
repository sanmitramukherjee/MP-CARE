"""Validate the phase6v3 density-justified box geometry BEFORE the campaign.

Target: mean degree k ~= 8.5 (above the log(n) percolation threshold at
every scale point), so random src-dst partitions and greedy-corridor walls
become rare by construction. Box side per N: area = N / rho with
rho = 8.5 / (pi * 300^2). Checks, on ALL 30 campaign seeds, per N:
  1. mean degree at t=0
  2. t=0 src(0)-dst(N-1) connectivity count (want 30/30 or close)
  3. plain greedy src->dst success on connected graphs (want >> N30's 0.60)
Verdict printed at the end; no campaign is launched from here.
"""
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))

import run_sim  # noqa: E402

R = 300.0
RHO = 8.5 / (math.pi * R * R)          # nodes per m^2 for k ~= 8.5
# edge-corrected sides: probe run 1 measured k below target near boundaries
# (border UAVs see fewer neighbors), so sides are shrunk per N so the
# MEASURED degree hits ~8.5; legs = 2N per the plan's S1-S6 waypoint rows.
BOXES = {5: (271.0, 10), 10: (460.0, 20), 20: (722.0, 40),
         30: (914.0, 60), 40: (1077.0, 80), 50: (1180.0, 100)}


def component(links: dict, src: int) -> set:
    seen, frontier = {src}, [src]
    while frontier:
        u = frontier.pop()
        for v in links.get(u, ()):
            if v not in seen:
                seen.add(v)
                frontier.append(v)
    return seen


def greedy_ok(links: dict, pos: dict, src: int, dst: int) -> bool:
    d2 = {v: math.dist(pos[v][:2], pos[dst][:2]) for v in pos}
    cur, steps = src, 0
    while cur != dst and steps < 500:
        nbrs = [v for v in links.get(cur, ()) if v != cur]
        if not nbrs:
            return False
        best = min(nbrs, key=lambda v: (d2[v], v))
        if d2[best] >= d2[cur]:
            return False
        cur = best
        steps += 1
    return cur == dst


def main() -> None:
    print(f"target k = 8.5 -> rho = {RHO:.3e} /m^2")
    print(f"{'N':>3} {'side m':>8} {'k mean':>7} {'k range':>16} "
          f"{'connected':>10} {'greedyOK':>9}")
    ok_all = True
    for n, (side, legs) in BOXES.items():
        run_sim.SCALE_POINTS[n] = (side, side, legs)
        degs, conn, gok = [], 0, 0
        for s in range(3000, 3030):
            _, link, agents, _, _ = run_sim.build_swarm(s, "scale", n_uav=n)
            pos = {a.uav_id: a.pos for a in agents}
            snap = run_sim.NetworkSnapshot(pos, link, 0.0)
            links = run_sim._graph(pos, snap)
            degs.append(sum(len(v) for v in links.values()) / n)
            c = (n - 1) in component(links, 0)
            conn += c
            if c:
                gok += greedy_ok(links, pos, 0, n - 1)
        ok = conn >= 29 and (gok / max(conn, 1)) >= 0.85
        ok_all &= ok
        print(f"{n:>3} {side:>8.0f} {statistics.mean(degs):>7.2f} "
              f"{min(degs):>7.2f}-{max(degs):<7.2f} {conn:>7}/30 "
              f"{gok / max(conn, 1):>9.2f}")
    print("\nVERDICT:", "GEOMETRY OK for v3" if ok_all else
          "NOT OK — tune rho up and re-run this probe")


if __name__ == "__main__":
    main()
