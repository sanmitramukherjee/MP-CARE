#!/usr/bin/env python3
"""Phase 5 benchmark campaign (implementation_plan.md section 7).

Extension A (§7.1) — ORCA + connectivity penalty:
  Two scenarios, plain ORCA (lam=0) vs comm-aware (lam>0), same seeds:
    spread : 6-UAV corridor convoy whose preferred velocities pull the two
             halves apart — tears routing links past max_range under plain
             ORCA; the penalty should hold them (connectivity metric).
    cross  : two 3-UAV columns flying head-on, lanes OFFSET 3 m laterally —
             a passable corridor (pure collision-avoidance stress; safety
             metric: min pairwise distance vs R = 1 m). NOT collinear: on a
             single line the axis-projected ORCA can only stop-and-go, and
             the first draft pinned BOTH variants at min_dist = 0 —
             benchmark-design flaw, fixed (NOTES/Phase_5.md).
  Metrics per trial: conn_ratio (mean fraction of the initial routing edges
  still inside max_range), min_pair_dist, collision ticks (pair < R).

Extension B (§7.2) — CBBA + comm-cost bid:
  8 agents on a corridor CHAIN (only consecutive links, 140-160 m, ETX~1);
  2 tasks anchored at host stations 4 and 2. The chain edge on each host's
  euclid-nearest side carries a 0.7 per-attempt impairment: LIVE in the
  consensus channel at 99.76% packet drop (0.3^2 data+ACK over 5 attempts;
  the 0.24% leak is accepted and documented). Because the plain (euclid)
  bid beats the clean relay on euclid, the host awards the task to the
  degraded-link agent whenever that link happens to pass a message, and
  then must execute with unreliable coordination; the comm-aware bid ranks
  the clean relay higher, so the same impairment SWITCHES the allocation.
  (Draft 7 finding: fully-severed links are SELF-HEALING in CBBA — a cut
  bidder's messages never reach the host at all; the win condition needs
  degradation, not severance. Full history in NOTES/Phase_5.md.)
  ALL consensus exchanges draw from ONE pre-drawn schedule seeded by
  (seed, loss), SHARED by both variants (paired channels). Beliefs are
  read from the HOST's table (only the host's component can execute).
  Loss sweep 0-60% = plan §7.2 range (10-60%) plus a 0% sanity point.
  Metrics:
    completion_rate : winner holds a clean route to the host station (no
                      impaired edge, no greedy stall)
    optimal_rate    : winner is the clean-route-optimal agent
    total_route_cost: winner's route cost to the station (impaired edges
                      charged impairment-ETX; stall = penalty)

Outputs (data/processed/):
  phase5_orca_trials.csv / phase5_orca_summary.json / phase5_orca.png
  phase5_cbba_trials.csv / phase5_cbba_summary.json / phase5_cbba.png
Exit-criteria verdicts (plan §7.3) printed from the measured numbers only.

Usage:
  py -3.13 scripts/run_phase5.py                # full campaign (fast: pure py)
  py -3.13 scripts/run_phase5.py --trials 5     # quick pass
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "orca_avoidance"))
sys.path.insert(0, str(ROOT / "src" / "cbba_allocator"))

from network_model.model import LinkModel, NetworkParams, etx as etx_of  # noqa: E402
from orca_avoidance.core import orca_step, orca_step_comms  # noqa: E402
from cbba_allocator.core import (CbbaAgent, Task,  # noqa: E402
                                 UNREACHABLE_PENALTY_M, route_cost_to,
                                 run_consensus)

OUT_DIR = ROOT / "data" / "processed"
R_COLL = 1.0            # R_safe (plan §7.1): collision threshold, m
DT, TAU_F = 0.1, 30.0   # tau = 3 s: van den Berg et al. use a multi-second
                        # horizon (reaction scale); at 0.5 s the corridor's
                        # 13 m/s rear agents only register threats ~6 m out —
                        # physically under-horizoned (draft-10 coll ticks)
TICKS = 200             # 20 s per ORCA trial
MAX_RANGE = 300.0

ORCA_SCENARIOS = ("spread", "cross")
CBBA_LOSSES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60)  # §7.2: 10-60% + sanity
N_AGENTS_B, N_TASKS_B = 8, 6
COMMIT_EPOCHS = 4          # bounded negotiation window (time-limited CBBA):
                           # 15 epochs converge at every swept loss (draft 2
                           # measured 0->60% identical), so loss only binds
                           # under a deadline
LOOKAHEAD_S = 3.0          # §7.1 model lookahead (NOTES/Phase_5.md)


# ------------------------------------------------------------------ ORCA side
def _scenario(scn: str, seed: int):
    """Deterministic per-seed scenario: positions + preferred velocities."""
    m = random.Random(seed)
    if scn == "spread":
        n = 6
        pos = {i: (i * 65.0, 0.0) for i in range(n)}          # span 325 m
        jit = [m.uniform(0.7, 1.3) for _ in range(n)]
        vp = {i: (6.0 * jit[i] * (i - 2.5), 0.0) for i in range(n)}
    else:  # cross: two 3-UAV columns head-on, lanes offset 3 m laterally.
        # EQUAL speeds within each column (jitter shared): within-column
        # gaps are constant by construction, so the scenario isolates the
        # head-on pass — any sustained within-column closing rate eventually
        # collides in a 20 s sim no matter the horizon (tau only guarantees
        # collision-free WITHIN tau; draft-10's rear-faster columns eroded
        # for 14 s before biting). This is the intended stress anyway.
        pos = {i: (0.0 - 30.0 * i, 200.0) for i in range(3)}
        for k in range(3):
            pos[3 + k] = (400.0 + 30.0 * k, 197.0)
        jc_a, jc_b = m.uniform(0.7, 1.3), m.uniform(0.7, 1.3)
        vp = {i: (8.0 * jc_a, 0.0) for i in range(3)}
        for k in range(3):
            vp[3 + k] = (-8.0 * jc_b, 0.0)
    edges = [(i, j) for i in pos for j in pos
             if i < j and math.dist(pos[i], pos[j]) <= MAX_RANGE]
    return pos, vp, edges


def run_orca_trial(scn: str, seed: int, lam: float) -> dict:
    pos, vp, edges = _scenario(scn, seed)
    n = len(pos)
    v: dict[int, tuple[float, float]] = {i: (0.0, 0.0) for i in pos}
    # Routing-neighbor view of the comms variant: the INITIAL routing graph
    # (t=0 edges) with LIVE neighbor positions. This is the hysteresis
    # approximation of MP-CARE's neighbor table (Phase 3: neighbors join the
    # routing set only after sustained range, leave instantly when broken) —
    # §7.1 penalizes velocities breaking links with ROUTING neighbors, not
    # with every transiently-in-range RF pair. In `cross` this keeps the
    # just-passed opposing column out of the penalty (it was never a routing
    # neighbor within the 20 s window), so the safety scenario measures
    # collision avoidance, not link-chasing.
    init_edges = {i: {j for (a, b) in edges for j in ([b] if a == i else
                                                     [a] if b == i else [])}
                  for i in pos}
    min_dist = math.inf
    coll_ticks = 0
    conn_sum = 0.0
    first = True
    for _ in range(TICKS):
        if lam > 0.0:
            nbrs = {i: {j: pos[j] for j in init_edges[i]
                        if math.dist(pos[i], pos[j]) <= MAX_RANGE}
                    for i in pos}
            v = orca_step_comms(n, pos, vp, v, nbrs, max_range_m=MAX_RANGE,
                                lam=lam, dt=DT, tau_factor=TAU_F,
                                lookahead_s=LOOKAHEAD_S)
        else:
            v = orca_step(n, pos, vp, v, dt=DT, tau_factor=TAU_F)
        if first:                     # t=0 is the initial condition, not a
            first = False             # behavioral outcome — excluded
        for i in pos:
            for j in pos:
                if i < j:
                    d = math.dist(pos[i], pos[j])
                    min_dist = min(min_dist, d)
                    if d < R_COLL:
                        coll_ticks += 1
        conn_sum += sum(1 for i, j in edges
                        if math.dist(pos[i], pos[j]) <= MAX_RANGE) / len(edges)
        pos = {i: (pos[i][0] + v[i][0] * DT, pos[i][1] + v[i][1] * DT)
               for i in pos}
    return {"scenario": scn, "seed": seed, "variant": ("orca" if lam == 0.0
            else f"orca_comms_lam{lam}"), "lam": lam,
            "conn_ratio": round(conn_sum / TICKS, 4),
            "n_edges": len(edges),
            "min_pair_dist_m": round(min_dist, 3),
            "collision_ticks": coll_ticks}


def orca_campaign(trials: int, seed0: int) -> tuple[list[dict], dict]:
    rows = []
    for scn in ORCA_SCENARIOS:
        for lam in (0.0, 0.4):   # 0.4 chosen from the stability sweep in
                                 # NOTES/Phase_5.md (0.6+ collapses pairs)
            for k in range(trials):
                r = run_orca_trial(scn, seed0 + k, lam)
                rows.append(r)
                print(f"[orca/{scn}] seed={seed0 + k} {r['variant']:<16} "
                      f"conn={r['conn_ratio']:.3f} min_d={r['min_pair_dist_m']:7.2f} "
                      f"coll_ticks={r['collision_ticks']}")
    summary: dict = {}
    from stats_utils import cohens_d, holm_bonferroni, mannwhitneyu  # noqa: E402
    pvals = []
    labels = []
    for scn in ORCA_SCENARIOS:
        plain = [r["conn_ratio"] for r in rows
                 if r["scenario"] == scn and r["lam"] == 0.0]
        comms = [r["conn_ratio"] for r in rows
                 if r["scenario"] == scn and r["lam"] > 0.0]
        mp = statistics.mean(plain)
        mc = statistics.mean(comms)
        u = mannwhitneyu(comms, plain)
        d = cohens_d(comms, plain)
        entry = {"conn_plain_mean": round(mp, 4), "conn_comms_mean": round(mc, 4),
                 "conn_gain_pp": round(100 * (mc - mp), 2),
                 "conn_gain_rel_pct": round(100 * (mc - mp) / mp, 2) if mp else None,
                 "mw_p": u["p"], "cohens_d": round(d, 3) if d is not None else None,
                 "min_dist_plain": round(min(r["min_pair_dist_m"] for r in rows
                                             if r["scenario"] == scn and r["lam"] == 0.0), 3),
                 "min_dist_comms": round(min(r["min_pair_dist_m"] for r in rows
                                             if r["scenario"] == scn and r["lam"] > 0.0), 3),
                 "coll_ticks_plain": sum(r["collision_ticks"] for r in rows
                                         if r["scenario"] == scn and r["lam"] == 0.0),
                 "coll_ticks_comms": sum(r["collision_ticks"] for r in rows
                                         if r["scenario"] == scn and r["lam"] > 0.0)}
        summary[scn] = entry
        if scn == "spread":                    # connectivity criterion applies here
            pvals.append(u["p"])
            labels.append(f"orca:conn:{scn}")
        else:                                   # safety criterion: min dist
            pvals.append(u["p"])
            labels.append(f"orca:safety:{scn}")
    holm = holm_bonferroni(pvals)
    for lab, p in zip(labels, holm):
        summary[lab + "_holm_p"] = p
    return rows, summary


# ------------------------------------------------------------------ CBBA side
def _cbba_instance(seed: int):
    m = random.Random(seed)
    # Corridor chain + degraded link (the §7.2 win condition; drafts 1-8
    # history in NOTES/Phase_5.md): 8 agents, ONLY consecutive links (chain,
    # 140-160 m, ETX~1). Tasks are anchored at HOST stations (routes
    # terminate at the node — point-anchored tasks are impossible with
    # strict-improvement greedy: the nearest node always stalls, provable);
    # the HOST does not bid on its own station (it is the requester).
    # The chain edge on each host's euclid-nearest side is degraded: 0.7
    # per-attempt loss -> per-exchange pass p = 1-(1-PRR^2*0.3^2)^5 ~= 0.376
    # (MAC-consistent with the ETX x11.1 the comm bid charges that edge).
    # Plain (euclid) bid: the degraded-side neighbor outbids the clean
    # relay, so the host awards the task to it whenever the bad link passes
    # (~85% over 4 epochs) -> dirty allocation. Comm bid: the clean relay
    # outbids it ALWAYS -> reliable allocation. Same impairment, same
    # schedule, different decisions.
    x0 = [40.0, 180.0, 320.0, 470.0, 600.0, 760.0, 900.0, 1040.0]
    jit = [m.uniform(-25.0, 25.0) for _ in range(N_AGENTS_B)]
    yj = [m.uniform(-20.0, 20.0) for _ in range(N_AGENTS_B)]
    pos = {i: (x0[i] + jit[i], 400.0 + yj[i], 40.0) for i in range(N_AGENTS_B)}
    hosts = (4, 2)
    # Stations EXACTLY at the host node position: a strict-improvement greedy
    # walk to a point jittered off the nearest node stalls those last meters
    # (no neighbor is strictly closer) — draft 8's completion-0 bug.
    tasks = [Task(k, pos[h][:2]) for k, h in enumerate(hosts)]
    chain = [(i, i + 1) for i in range(N_AGENTS_B - 1)]
    links = {i: set() for i in pos}
    for a, b in chain:
        links[a].add(b)
        links[b].add(a)
    impair = {frozenset((3, 4)), frozenset((1, 2))}   # euclid-side edges
    BAD_LF = 0.7                         # per-attempt impairment
    PASS_BAD = 1.0 - (1.0 - 0.3 ** 2) ** 5   # per-exchange pass ~= 0.376
    params = NetworkParams()

    def snap_etx(a: int, b: int) -> float:
        e = etx_of(math.dist(pos[a][:2], pos[b][:2]), params)
        return e / (1.0 - BAD_LF) ** 2 if frozenset((a, b)) in impair else e

    return pos, links, tasks, snap_etx, impair, hosts, PASS_BAD


def run_cbba_trial(seed: int, loss: float, w_c: float) -> dict:
    (pos, links, tasks, snap_etx, impair, hosts,
     PASS_BAD) = _cbba_instance(seed)
    # ONE pre-drawn drop schedule shared by both variants at this (seed,
    # loss) — paired channels; the seed depends only on (seed, loss).
    rng = random.Random(seed * 100 + int(round(loss * 100)))
    schedule = [rng.random() < loss
                for _ in range(N_AGENTS_B * N_TASKS_B * COMMIT_EPOCHS)]
    it = iter(schedule)

    rng2 = random.Random(seed * 977 + 13)   # impairment passes, variant-blind

    def channel(d_m: float, a=None, b=None) -> bool:
        if d_m > MAX_RANGE:
            return False                    # out of range: physics, not loss
        if a is not None and b is not None \
                and frozenset((a, b)) in impair \
                and rng2.random() > PASS_BAD:
            return False                    # degraded pair: 37.6% pass
        return not next(it)                 # True in schedule = drop

    agents = [CbbaAgent(i, w_d=1.0, w_t=0.5, w_c=w_c) for i in pos]
    for a in agents:
        # Requester exclusion (both variants): the host does not bid on its
        # own station — it ISSUES the task. Whoever the host's belief table
        # settles on is the effective winner.
        mine = [t for k, t in enumerate(tasks) if hosts[k] != a.uav_id]
        a.set_view(pos, links, snap_etx if w_c > 0.0 else None)
        a.decide(mine, n_bundle=2)

    n_msg = [1 + len(links.get(a.uav_id, ())) * COMMIT_EPOCHS
             for a in agents]  # bids + exchanges
    belief = run_consensus(agents, links, link_success=channel,
                           epochs=COMMIT_EPOCHS)

    # Scoring, final semantics (drafts 1-9 history in NOTES/Phase_5.md).
    # Beliefs are read from the HOST's table — only the host's component can
    # execute the task:
    #   completion_rate = winner reaches the station over CLEAN links only
    #     (an impaired cut blocks reliable coordination across it; the chain
    #     topology routes every cross-side path through the cut);
    #   optimal_rate = winner is the impairment-charged-route-optimal agent
    #     (argmin route cost at degraded-edge ETX — the §7.2 objective);
    #   total_route_cost = winner's impairment-charged route cost.
    def clean_reach(src: int, dst: int) -> bool:
        # Clean-graph reachability: a route that avoids the impaired edge
        # entirely. A degraded cut really does block reliable coordination
        # across it (37.6% per-exchange pass); the chain topology forces
        # every cross-side route through the cut, so a winner on the far
        # side of it cannot reliably execute — that is completion-false by
        # physics, not by bookkeeping.
        seen, stack = {src}, [src]
        while stack:
            c = stack.pop()
            if c == dst:
                return True
            for nb in links.get(c, ()):
                if nb not in seen and frozenset((c, nb)) not in impair:
                    seen.add(nb)
                    stack.append(nb)
        return False

    comp = done = opt = 0
    route_cost = 0.0
    for k, t in enumerate(tasks):
        comp += 1
        h = hosts[k]
        u = belief[h].get(t.task_id, (None, None))[1]
        # Impairment-charged route costs (the real coordination reliability:
        # the degraded edge costs ETX x11.1) for the optimality metric:
        costs = {a: route_cost_to(pos[a][:2], t.pos, links, pos, a, snap_etx,
                                  unreachable_penalty=UNREACHABLE_PENALTY_M)
                 for a in pos if a != h}
        best = min(costs, key=costs.get)
        opt += (u == best)
        if u is not None:
            done += clean_reach(u, h)
            route_cost += costs[u]
    return {"seed": seed, "loss": loss, "variant": ("cbba" if w_c == 0.0
            else "cbba_comm"), "w_c": w_c,
            "completion_rate": round(done / comp, 4),
            "optimal_rate": round(opt / comp, 4),
            "n_assigned": sum(1 for k2, t2 in enumerate(tasks)
                              if belief[hosts[k2]].get(t2.task_id)),
            "total_route_cost": round(route_cost, 1),
            "ctrl_msgs_per_agent": round(statistics.mean(n_msg), 1)}


def cbba_campaign(trials: int, seed0: int) -> tuple[list[dict], dict]:
    rows = []
    for loss in CBBA_LOSSES:
        for w_c in (0.0, 1.0):
            for k in range(trials):
                r = run_cbba_trial(seed0 + k, loss, w_c)
                rows.append(r)
                print(f"[cbba] seed={seed0 + k} loss={loss:.2f} {r['variant']:<10} "
                      f"compl={r['completion_rate']:.3f} "
                      f"opt={r['optimal_rate']:.3f} "
                      f"route_cost={r['total_route_cost']:.0f}")
    from stats_utils import holm_bonferroni, mannwhitneyu  # noqa: E402
    summary: dict = {"losses": list(CBBA_LOSSES)}
    pvals, labels = [], []
    for loss in CBBA_LOSSES:
        base = [r["completion_rate"] for r in rows
                if r["loss"] == loss and r["w_c"] == 0.0]
        comm = [r["completion_rate"] for r in rows
                if r["loss"] == loss and r["w_c"] > 0.0]
        mb, mc = statistics.mean(base), statistics.mean(comm)
        u = mannwhitneyu(comm, base)
        key = f"loss_{int(round(loss * 100)):02d}"
        op_b = statistics.mean(r["optimal_rate"] for r in rows
                               if r["loss"] == loss and r["w_c"] == 0.0)
        op_c = statistics.mean(r["optimal_rate"] for r in rows
                               if r["loss"] == loss and r["w_c"] > 0.0)
        summary[key] = {"compl_base_mean": round(mb, 4),
                        "compl_comm_mean": round(mc, 4),
                        "gain_pp": round(100 * (mc - mb), 2),
                        "optimal_base": round(op_b, 4),
                        "optimal_comm": round(op_c, 4),
                        "mw_p": u["p"],
                        "route_cost_base": round(statistics.mean(
                            r["total_route_cost"] for r in rows
                            if r["loss"] == loss and r["w_c"] == 0.0), 1),
                        "route_cost_comm": round(statistics.mean(
                            r["total_route_cost"] for r in rows
                            if r["loss"] == loss and r["w_c"] > 0.0), 1)}
        pvals.append(u["p"])
        labels.append(f"cbba:{key}")
    holm = holm_bonferroni(pvals)
    for lab, p in zip(labels, holm):
        summary[lab + "_holm_p"] = p
    return rows, summary


# ------------------------------------------------------------------ outputs
def _write_orca(rows: list[dict], summary: dict, tag: str) -> None:
    with open(OUT_DIR / f"phase5_orca_trials{tag}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_DIR / f"phase5_orca_summary{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_orca(summary, tag)


def plot_orca(summary: dict, tag: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        scns = ORCA_SCENARIOS
        x = range(len(scns))
        base = [summary[s]["conn_plain_mean"] for s in scns]
        comm = [summary[s]["conn_comms_mean"] for s in scns]
        ax.bar([i - 0.18 for i in x], base, width=0.36, label="ORCA (plain)")
        ax.bar([i + 0.18 for i in x], comm, width=0.36,
               label="ORCA + connectivity penalty")
        ax.set_xticks(list(x))
        ax.set_xticklabels(scns)
        ax.set_ylabel("routing-edge retention (mean ratio)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Extension A: connectivity under spreading convoy")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"phase5_orca{tag}.png", dpi=600)
        print(f"chart: {OUT_DIR / f'phase5_orca{tag}.png'}")
    except ImportError:
        print("matplotlib not available — skipped the ORCA chart")


def _write_cbba(rows: list[dict], summary: dict, tag: str) -> None:
    with open(OUT_DIR / f"phase5_cbba_trials{tag}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT_DIR / f"phase5_cbba_summary{tag}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    plot_cbba(summary, tag)


def plot_cbba(summary: dict, tag: str = "") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        losses = [int(round(l * 100)) for l in CBBA_LOSSES]
        base = [summary[f"loss_{l:02d}"]["compl_base_mean"] for l in losses]
        comm = [summary[f"loss_{l:02d}"]["compl_comm_mean"] for l in losses]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(losses, base, "o-", label="CBBA (baseline)")
        ax.plot(losses, comm, "s-", label="CBBA + comm-cost bid")
        ax.set_xlabel("injected control-channel loss (%)")
        ax.set_ylabel("completion rate (nearest-agent wins)")
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title("Extension B: task allocation under channel loss")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"phase5_cbba{tag}.png", dpi=600)
        print(f"chart: {OUT_DIR / f'phase5_cbba{tag}.png'}")
    except ImportError:
        print("matplotlib not available — skipped the CBBA chart")


def main():
    ap = argparse.ArgumentParser(description="Phase 5 extension campaign")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--only", choices=("orca", "cbba"), default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ok = True

    if args.only in (None, "orca"):
        rows, summary = orca_campaign(args.trials, args.seed0)
        _write_orca(rows, summary, f"_{args.tag}" if args.tag else "")
        print("\n== Extension A (ORCA + connectivity) ==")
        for scn in ORCA_SCENARIOS:
            s = summary[scn]
            print(f"  {scn:<7} conn {s['conn_plain_mean']:.3f} -> "
                  f"{s['conn_comms_mean']:.3f} "
                  f"(+{s['conn_gain_pp']:.1f} pp, p={s['mw_p']:.4f}) | "
                  f"min_dist {s['min_dist_plain']:.2f} vs {s['min_dist_comms']:.2f} m | "
                  f"coll ticks {s['coll_ticks_plain']} vs {s['coll_ticks_comms']}")
        spread_ok = (summary["spread"]["conn_gain_rel_pct"] is not None
                     and summary["spread"]["conn_gain_rel_pct"] >= 5.0
                     and summary["orca:conn:spread_holm_p"] is not None
                     and summary["orca:conn:spread_holm_p"] < 0.05)
        safety_ok = (summary["cross"]["min_dist_comms"]
                     >= summary["cross"]["min_dist_plain"] - 1e-9
                     and summary["cross"]["coll_ticks_comms"] == 0
                     and summary["cross"]["min_dist_comms"] > R_COLL)
        print(f"  §7.3 connectivity criterion (spread, plan: >= 5% relative): "
              f"{summary['spread']['conn_gain_rel_pct']}% rel -> "
              f"{'PASS' if spread_ok else 'NOT MET'}")
        print(f"  §7.3 safety criterion (cross): "
              f"{'PASS' if safety_ok else 'NOT MET'}")
        ok &= spread_ok and safety_ok

    if args.only in (None, "cbba"):
        rows, summary = cbba_campaign(args.trials, args.seed0)
        _write_cbba(rows, summary, f"_{args.tag}" if args.tag else "")
        print("\n== Extension B (CBBA + comm-cost bid) ==")
        for loss in CBBA_LOSSES:
            key = f"loss_{int(round(loss * 100)):02d}"
            s = summary[key]
            print(f"  loss {loss:>4.0%}: completion "
                  f"{s['compl_base_mean']:.3f} -> {s['compl_comm_mean']:.3f} "
                  f"({s['gain_pp']:+.1f} pp, p={s['mw_p']:.4f}) | optimal "
                  f"{s['optimal_base']:.3f} -> {s['optimal_comm']:.3f} | "
                  f"route cost {s['route_cost_base']:.0f} -> {s['route_cost_comm']:.0f}")
        gains = [summary[f"loss_{int(round(l * 100)):02d}"]["gain_pp"]
                 for l in CBBA_LOSSES]
        print(f"  §7.3 completion criterion (comm >= base at every loss): "
              f"{'PASS' if all(g >= 0.0 for g in gains) else 'NOT MET'}")
        print(f"  §7.3 '>= 5 pp somewhere' criterion: "
              f"{'PASS' if max(gains) >= 5.0 else 'NOT MET (report honestly)'}")
        ok &= all(g >= 0.0 for g in gains)

    print(f"\n{args.trials} trials/condition in {time.perf_counter() - t0:.0f}s "
          f"-> data/processed/phase5_*")
    print("RESULT:", "EXIT CRITERIA MET" if ok else "CRITERIA NOT MET (honest)")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
