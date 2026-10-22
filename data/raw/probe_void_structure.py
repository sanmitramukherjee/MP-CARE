"""Phase 6 probe: WHY is connected-seed PDR worse at N=30/40 than N=50?

Two candidate mechanisms, measured directly on the REAL static graphs
(t=0, built with the real scenario code):
  M1 void propensity: for every node v (w.r.t. the actual dst N-1 and its
     position), is v a GREEDY LOCAL MINIMUM (all neighbors farther from dst
     than v itself)? A packet entering such a node must face-walk. Also:
     plain greedy-forwarding success rate from src to dst per graph.
  M2 path hardness: BFS hop count src->dst (routing difficulty baseline).

Cross-referenced with measured v2 outcomes on the same seeds (connected
seeds only): mpcare void_recovery_events per sent packet, and the
mpcare-vs-gpsr PDR gap per N. If M1/M2 do NOT separate N=30/40 from N=50,
the connected-seed dip is sample noise + selection, and the partitioned
seeds are the real story.
"""
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))

import run_sim  # noqa: E402

DISC = {5: {3014, 3018}, 10: {3026}, 20: {3006, 3008, 3025, 3027},
        30: {3013, 3016, 3021, 3022, 3025}, 40: {3000, 3014, 3024, 3027},
        50: {3000, 3001, 3002, 3004, 3005, 3006, 3010, 3011, 3012, 3015,
             3022, 3029}}


def static_graph(seed: int, n: int):
    _, link, agents, _, _ = run_sim.build_swarm(seed, "scale", n_uav=n)
    pos = {a.uav_id: a.pos for a in agents}
    snap = run_sim.NetworkSnapshot(pos, link, 0.0)
    return run_sim._graph(pos, snap), pos


def component(links: dict, src: int) -> set:
    seen, frontier = {src}, [src]
    while frontier:
        u = frontier.pop()
        for v in links.get(u, ()):
            if v not in seen:
                seen.add(v)
                frontier.append(v)
    return seen


def bfs_hops(links: dict, src: int, dst: int) -> int | None:
    if src == dst:
        return 0
    dist = {src: 0}
    frontier = [src]
    while frontier:
        nxt = []
        for u in frontier:
            for v in links.get(u, ()):
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if v == dst:
                        return dist[v]
                    nxt.append(v)
        frontier = nxt
    return None


def greedy_success(links: dict, pos: dict, src: int, dst: int,
                   max_steps: int = 500) -> tuple[bool, int]:
    """Plain greedy-cf forwarding src->dst on the static graph."""
    d2 = {v: math.dist(pos[v][:2], pos[dst][:2]) for v in pos}
    cur, steps = src, 0
    while cur != dst and steps < max_steps:
        nbrs = [v for v in links.get(cur, ()) if v != cur]
        if not nbrs:
            return False, steps
        best = min(nbrs, key=lambda v: (d2[v], v))
        if d2[best] >= d2[cur]:
            return False, steps          # stuck: greedy local minimum
        cur = best
        steps += 1
    return cur == dst, steps


def v2_connected_stats(n: int, connected: set[int]) -> dict:
    rows = list(csv.DictReader(
        open(ROOT / "data" / "processed" / "phase6v2_trials.csv", encoding="utf-8")))
    sel = [r for r in rows if int(r["N"]) == n and int(r["seed"]) in connected]
    out = {}
    for a in ("mpcare", "gpsr", "olsr", "aodv"):
        out[a] = statistics.mean(float(r["pdr"]) for r in sel if r["algorithm"] == a)
    ev = [int(float(r["void_recovery_events"])) for r in sel
          if r["algorithm"] == "mpcare"]
    sent = [int(float(r["packets_sent"])) for r in sel
            if r["algorithm"] == "mpcare"]
    out["void_events_per_sent"] = (sum(ev) / sum(sent)) if sum(sent) else 0.0
    return out


def main() -> None:
    print(f"{'N':>3} {'conn':>5} {'deg':>5} {'bfs':>6} {'minima/g':>9} "
          f"{'greedyOK':>9} {'ev/sent':>8} | {'mpcare':>7} {'gpsr':>7} "
          f"{'olsr':>7} {'aodv':>7}")
    for n in (10, 20, 30, 40, 50):
        connected = {s for s in range(3000, 3030)} - DISC[n]
        degs, bfss, minima, gok = [], [], [], 0
        for s in sorted(connected):
            links, pos = static_graph(s, n)
            dst = n - 1
            d2 = {v: math.dist(pos[v][:2], pos[dst][:2]) for v in pos}
            degs.append(sum(len(v) for v in links.values()) / n)
            bfss.append(bfs_hops(links, 0, dst))
            stuck = sum(1 for v in pos
                        if v != dst and d2[v] > 0 and
                        (not links.get(v) or
                         min(d2[w] for w in links[v]) >= d2[v]))
            minima.append(stuck)
            ok, _ = greedy_success(links, pos, 0, dst)
            gok += ok
        st = v2_connected_stats(n, connected)
        print(f"{n:>3} {len(connected):>5} {statistics.mean(degs):>5.2f} "
              f"{statistics.mean(bfss):>6.1f} {statistics.mean(minima):>9.1f} "
              f"{gok / len(connected):>9.2f} {st['void_events_per_sent']:>8.3f} | "
              f"{st['mpcare']:>7.3f} {st['gpsr']:>7.3f} {st['olsr']:>7.3f} "
              f"{st['aodv']:>7.3f}")
    print()
    print("cols: conn = t=0 src-dst-connected seeds; deg = mean degree; "
          "bfs = mean BFS hops src->dst; minima/g = greedy local-minimum nodes "
          "per graph (w.r.t. dst); greedyOK = fraction of connected graphs "
          "where plain greedy reaches dst at t=0; ev/sent = measured v2 mpcare "
          "void-walk entries per sent packet; then measured v2 connected-seed PDR.")


if __name__ == "__main__":
    main()
