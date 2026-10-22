"""Phase 6 probe: does DENSITY (not N) explain the N=30/40 PDR gap?

Runs 8 previously-losing seeds at N=30 on the SAME box family but with the
S4 box (1200x1200, k~5.2, percolation edge) vs a denser 1000x1000 box
(k~8.2, above the classical log(n) percolation threshold). Uses the FROZEN
router, no code changes. Answers: is the remaining ceiling a routing defect
or a scenario-density artifact?
"""
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))

import run_sim  # noqa: E402

SEEDS = [3000, 3001, 3002, 3013, 3016, 3021, 3024, 3027]
ALGOS = ("mpcare", "olsr", "aodv", "gpsr")


def mean_degree(seed: int, n: int, box: tuple[float, float]) -> float:
    import random
    m0 = random.Random(seed)
    start = [(m0.uniform(40.0, box[0] - 40.0), m0.uniform(40.0, box[1] - 40.0), 40.0)
             for _ in range(n)]
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            if ((start[i][0] - start[j][0]) ** 2 + (start[i][1] - start[j][1]) ** 2) ** 0.5 <= 300.0:
                edges += 1
    return 2.0 * edges / n


def connected0(seed: int, n: int, box) -> bool:
    sys.modules["run_sim"].SCALE_POINTS[n] = (box[0], box[1], 60)
    _, link, agents, _, _ = run_sim.build_swarm(seed, "scale", n_uav=n)
    pos = {a.uav_id: a.pos for a in agents}
    snap = run_sim.NetworkSnapshot(pos, link, 0.0)
    links = run_sim._graph(pos, snap)
    seen, frontier = {0}, [0]
    while frontier:
        u = frontier.pop()
        for v in links.get(u, ()):
            if v not in seen:
                seen.add(v)
                frontier.append(v)
    return (n - 1) in seen


def main() -> None:
    for label, box in (("S4 box 1200x1200 (k~5.2)", (1200.0, 1200.0)),
                       ("dense 1000x1000 (k~8.2)", (1000.0, 1000.0))):
        run_sim.SCALE_POINTS[30] = (box[0], box[1], 60)
        conns = [connected0(s, 30, box) for s in SEEDS]
        print(f"== {label}: t=0 src-dst connected {sum(conns)}/{len(SEEDS)} seeds ==")
        for a in ALGOS:
            vals = []
            for s in SEEDS:
                r = run_sim.run(s, ROOT / "data" / "raw", duration=150.0,
                                write_csv=False, algorithm=a, scenario="scale",
                                n_uav=30)
                vals.append(r["pdr"])
            print(f"  {a:<7} pdr mean={statistics.mean(vals):.3f} "
                  f"(min {min(vals):.2f}, zeros {sum(1 for v in vals if v == 0)})")
        print()


if __name__ == "__main__":
    main()
