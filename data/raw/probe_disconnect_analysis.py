"""Phase 6 diagnostic: what ARE the "disconnected" seeds?

Measured evidence only — this probe reuses the REAL scenario builder
(run_sim.build_swarm) and the REAL agent movement, so the graphs here are
bit-identical to the ones the campaign ran on. No re-implementation.

Questions answered:
  Q1. At t=0 (the static 30 s hover window in which ALL 50 benchmark packets
      are created): is UAV0 (src) connected to UAV N-1 (dst)?
  Q2. Do the t=0-disconnected seeds match the measured all-algorithms-PDR=0
      seeds from phase6v2_trials.csv EXACTLY? (falsifiable check)
  Q3. For each dead seed: how big are the two islands, how far apart (the
      "bridge gap" = closest pair of nodes between the src and dst islands),
      and did the islands EVER merge during the 150 s flight?
"""
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))

import run_sim  # noqa: E402


def component(links: dict, src: int) -> set:
    seen, frontier = {src}, [src]
    while frontier:
        u = frontier.pop()
        for v in links.get(u, ()):
            if v not in seen:
                seen.add(v)
                frontier.append(v)
    return seen


def static_graph(seed: int, n: int):
    _, link, agents, _, _ = run_sim.build_swarm(seed, "scale", n_uav=n)
    positions = {a.uav_id: a.pos for a in agents}
    snap = run_sim.NetworkSnapshot(positions, link, 0.0)
    return run_sim._graph(positions, snap), agents


def flight_merges(seed: int, n: int, duration: float = 150.0, dt: float = 0.1):
    """Sample the src-dst component once per second during the whole flight."""
    _, link, agents, _, _ = run_sim.build_swarm(seed, "scale", n_uav=n)
    merges: list[float] = []
    arrival = [None] * n
    for tick in range(int(duration / dt)):
        t = tick * dt
        if t >= run_sim.HOVER_UNTIL:
            for a in agents:
                wp = a.mission.current_wp()
                if wp and a.move_toward(wp, dt):
                    a.mission.advance()
                    if arrival[a.uav_id] is None:
                        arrival[a.uav_id] = t
        if t >= run_sim.HOVER_UNTIL and (tick % 10 == 0):   # 1 Hz samples
            positions = {a.uav_id: a.pos for a in agents}
            snap = run_sim.NetworkSnapshot(positions, link, t)
            if (n - 1) in component(run_sim._graph(positions, snap), 0):
                merges.append(round(t, 1))
    return merges


def measured_dead_seeds() -> dict[int, set[int]]:
    rows = list(csv.DictReader(
        open(ROOT / "data" / "processed" / "phase6v2_trials.csv", encoding="utf-8")))
    dead: dict[int, set[int]] = {}
    for r in rows:
        n, seed = int(r["N"]), int(r["seed"])
        dead.setdefault(n, set())
        per_seed = [x for x in rows if x["N"] == r["N"] and x["seed"] == r["seed"]]
        if all(float(x["pdr"]) == 0.0 for x in per_seed):
            dead[n].add(seed)
    return dead


def main() -> None:
    dead = measured_dead_seeds()
    print("measured all-algo-PDR=0 seeds from phase6v2 (per N):")
    for n in sorted(dead):
        print(f"  N={n}: {sorted(dead[n]) if dead[n] else 'none'}")
    print()

    mismatch = 0
    for n in (5, 10, 20, 30, 40, 50):
        print(f"== N={n} (box {run_sim.SCALE_POINTS[n][:2]}, "
              f"{run_sim.SCALE_POINTS[n][2]} legs) ==")
        for seed in range(3000, 3030):
            links, agents = static_graph(seed, n)
            comp0 = component(links, 0)
            dst = n - 1
            connected = dst in comp0
            mean_deg = sum(len(v) for v in links.values()) / n
            flag = ""
            if not connected:
                compd = component(links, dst)
                gap = min(math.dist(agents[u].pos[:2], agents[v].pos[:2])
                          for u in comp0 for v in compd)
                sizes = f"src island {len(comp0)}/{n}, dst island {len(compd)}/{n}"
                flag = f"  DISCONNECTED at t=0: {sizes}, bridge gap {gap:.0f} m"
                if seed in dead.get(n, set()):
                    m = flight_merges(seed, n)
                    flag += (f" | merged during flight at t={m[:4]}..."
                             if m else " | NEVER merged in 150 s")
                else:
                    flag += "  << !! t=0 connected but measured PDR=0 (FALSIFIES story)"
            elif seed in dead.get(n, set()):
                flag = "  << !! measured PDR=0 but t=0 connected (FALSIFIES story)"
                mismatch += 1
            if flag:
                print(f"  seed {seed}: deg={mean_deg:.2f}{flag}")
        # set equality per N
        t0_disc = set()
        for seed in range(3000, 3030):
            links, agents = static_graph(seed, n)
            if (n - 1) not in component(links, 0):
                t0_disc.add(seed)
        eq = "MATCH" if t0_disc == dead.get(n, set()) else "MISMATCH"
        if t0_disc != dead.get(n, set()):
            mismatch += 1
        print(f"  t=0-disconnected {sorted(t0_disc)} vs measured-dead "
              f"{sorted(dead.get(n, set()))}: {eq}\n")
    print("VERDICT:", "all sets match — dead seeds are pure physics"
          if mismatch == 0 else f"{mismatch} MISMATCHES — investigate!")


if __name__ == "__main__":
    main()
