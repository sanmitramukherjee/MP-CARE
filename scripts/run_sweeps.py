#!/usr/bin/env python3
"""Phase 4 stress-test suite (implementation_plan.md section 6).

Experiments implemented (section 6.2; experiment 7 = Phase 5 task arrival):
  exp2_loss      loss   in {0, 5, 10, 15, 20, 25, 30}%   chain scenario
  exp3_latency   delay  in {0, 10, 25, 50, 100, 150, 200} ms, chain scenario
  exp5_mobility  speed  in {5, 10, 15, 20, 25} m/s       mobility scenario
  exp6_failure   k in {0, 1, 2, 3} UAVs killed           grid scenario (n = 10)
  envelope       loss x speed grid, MP-CARE vs GPSR      mobility (RQ6, graph 4.7)

Per experiment:
  data/processed/<name>.csv            per-trial rows (seeds recorded: 6.5)
  data/processed/<name>_summary.json   means +- 95% CI, KW, post-hoc MW + Holm,
                                       rank-biserial, Cohen's d, 6.3 benchmarks
  data/processed/<name>_g41..g47.png   section 6.4 graphs

Statistics (section 6.5): >= 30 trials/condition; per condition Kruskal-Wallis
across algorithms, post-hoc Mann-Whitney U (MP-CARE vs each baseline) with
Holm-Bonferroni correction, rank-biserial + Cohen's d effect sizes, t-based
95% CIs. Seeds are SHARED across algorithms within a trial index (paired
design -> the reported margin is the paired mean difference; the MW test is
run as independent-samples, i.e. conservative).

Usage:
  py -3.13 scripts/run_sweeps.py                 # full campaign (~10 min)
  py -3.13 scripts/run_sweeps.py --only exp2,envelope
  py -3.13 scripts/run_sweeps.py --trials 3 --tag _smoke   # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))
sys.path.insert(0, str(ROOT / "src" / "logger"))
sys.path.insert(0, str(ROOT / "src" / "aodv_routing"))
sys.path.insert(0, str(ROOT / "src" / "olsr_routing"))

import run_sim  # noqa: E402
from stats_utils import (mannwhitneyu, kruskal_wallis, holm_bonferroni,  # noqa: E402
                         cohens_d, rank_biserial, mean_ci95)

OUT = ROOT / "data" / "processed"

ALGO_COLOR = {"astar_dist": "#8c8c8c", "gpsr": "#ed7d31", "aodv": "#4472c4",
              "olsr": "#70ad47", "astar_etx": "#ffc000", "mpcare": "#c00000"}

ROW_KEYS = ["seed", "trial", "algorithm", "condition", "chain_pdr", "pdr",
            "avg_latency_ms", "last_delivery_s", "min_chain_hops",
            "control_pkts_per_node_s", "first_route_s", "converged_s",              "recovery_s", "reroutes", "reroute_events", "let_audit_n",
            "let_median_rel_err", "let_median_rel_err_vle15",
            "let_median_rel_err_vle15_act2", "let_spearman",
            "loss", "latency_ms", "speed_mps", "kills",
            "scenario"]

KILL_POOL = (1, 2, 3, 6, 7, 8)   # grid mid-column nodes (never src 0 / dst 9)


def _load_qlgr_tables(needed: list[str]) -> dict:
    """Load the Phase-8 trained QLGR tables (diamond per-node + dynamic
    global). Returns {} when absent (then QLGR would silently train online,
    breaking the inference-only protocol: fail loudly instead)."""
    if "qlgr" not in needed:
        return {}
    tables: dict = {}
    for scen in ("diamond", "mobility"):
        p = OUT / f"phase8_qlgr_{scen}.json"
        if p.exists():
            tables[scen] = json.loads(p.read_text(encoding="utf-8"))["table"]
    dyn = OUT / "phase8_qlgr_dynamic.json"
    if dyn.exists():
        tables["dynamic"] = json.loads(dyn.read_text(encoding="utf-8"))["table"]
    if not tables.get("diamond") or not tables.get("dynamic"):
        raise SystemExit("missing Phase-8 QLGR tables — run "
                         "scripts/run_phase8.py --train first (inference-only "
                         "evaluation protocol)")
    return tables


def _kills_for(seed: int, k: int) -> list[tuple[float, int]]:
    """Deterministic per-seed kill schedule: k mid-grid UAVs at t = 15 + 5j s
    (all inside the traffic window, which ends at t = 30)."""
    rng = random.Random(seed * 31 + 7)
    ids = rng.sample(list(KILL_POOL), k)
    return [(15.0 + 5.0 * j, nid) for j, nid in enumerate(sorted(ids))]


def _flatten(r: dict, condition: str, params: dict) -> dict:
    row = {"seed": r["seed"], "algorithm": r["algorithm"], "condition": condition}
    for k in ("chain_pdr", "pdr", "avg_latency_ms", "last_delivery_s",
              "min_chain_hops", "control_pkts_per_node_s", "first_route_s",
              "converged_s", "recovery_s", "let_audit_n", "let_median_rel_err",
              "let_median_rel_err_vle15", "let_median_rel_err_vle15_act2",
              "let_spearman"):
        row[k] = r.get(k)
    ev = r.get("trigger_events") or {}
    row["reroute_events"] = json.dumps(ev) if ev else ""
    row["reroutes"] = sum(ev.values())
    row.update(params)
    return row


def run_experiment(name: str, conditions: list[dict], algorithms: list[str],
                   trials: int, seed0: int, duration: float, base: dict,
                   sanity_must_hold: bool = True) -> list[dict]:
    """Run one experiment grid: conditions x algorithms x trials."""
    required = ["qlgr"] if "qlgr" in algorithms else []
    tables = _load_qlgr_tables(required)
    rows: list[dict] = []
    t_start = time.perf_counter()
    for ci, cond in enumerate(conditions):
        label = cond.pop("label")
        for algo in algorithms:
            for trial in range(trials):
                seed = seed0 + trial
                kw = dict(base)
                kw.update(cond)
                kills = kw.pop("kills", None)
                if kills:
                    kw["kill_node"] = (_kills_for(seed, kills)
                                       if isinstance(kills, int) else kills)
                params = {"loss": kw.get("injected_loss") or 0.0,
                          "latency_ms": kw.get("injected_latency_ms") or 0.0,
                          "speed_mps": kw.get("speed"),
                          "kills": (f"{len(kw['kill_node'])}@" +
                                    ",".join(str(t) for t, _ in kw["kill_node"]))
                                  if kw.get("kill_node") else "0",
                          "scenario": kw.get("scenario", "")}
                if algo == "qlgr" and tables:
                    scen = kw.get("scenario", "")
                    table = (tables.get("diamond") if scen == "diamond"
                             else tables.get("dynamic", {}).get("global"))
                    if table:
                        # Phase 8 protocol: inference-only evaluation with the
                        # Phase-8-trained tables (never online learning inside
                        # evaluation trials)
                        kw["qlgr_opts"] = {"learn": False, "seed": seed,
                                           "eps": 0.0, "table": table}
                r = run_sim.run(seed, ROOT / "data" / "raw", duration=duration,
                                write_csv=False, algorithm=algo, **kw)
                if sanity_must_hold and not (r["stability_ok"] and r["waypoint_ok"]
                                             and r["network_ok"]):
                    print(f"  !! SANITY FAIL {name}/{label}/{algo} seed={seed}: "
                          f"stab={r['stability_ok']} wp={r['waypoint_ok']} "
                          f"net={r['network_ok']}", file=sys.stderr)
                rows.append(_flatten(r, label, params))
            print(f"[{name}] {label:<14} {algo:<11} {trials} trials done "
                  f"({time.perf_counter() - t_start:.0f}s elapsed)")
        cond["label"] = label     # restore for stats pass
    return rows


# ---------------------------------------------------------------- statistics

def per_algo_stats(rows: list[dict], algo: str, metric: str) -> dict:
    vals = [r[metric] for r in rows
            if r["algorithm"] == algo and isinstance(r[metric], (int, float))]
    ci = mean_ci95(vals)
    if ci is None:
        return {"n": 0}
    return {"n": len(vals), "mean": round(ci[0], 4), "ci95": round(ci[1], 4),
            "std": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals)}


def condition_stats(rows: list[dict], conditions: list[str],
                    algorithms: list[str], metric: str,
                    reference: str = "mpcare") -> dict:
    """Per condition: per-algo mean+-CI, KW across algos, post-hoc MW
    (reference vs each baseline) with Holm correction + effect sizes."""
    out: dict = {}
    for label in conditions:
        cr = [r for r in rows if r["condition"] == label]
        per_algo = {a: per_algo_stats(cr, a, metric) for a in algorithms}
        groups = [[r[metric] for r in cr
                   if r["algorithm"] == a and isinstance(r[metric], (int, float))]
                  for a in algorithms]
        kw = kruskal_wallis(groups)
        posthoc: dict = {}
        pvals: list = []
        pairs: list[str] = []
        for a in algorithms:
            if a == reference:
                continue
            mv = [r[metric] for r in cr if r["algorithm"] == reference
                  and isinstance(r[metric], (int, float))]
            bv = [r[metric] for r in cr if r["algorithm"] == a
                  and isinstance(r[metric], (int, float))]
            mw = mannwhitneyu(mv, bv)
            paired = (statistics.mean(mv) - statistics.mean(bv)) \
                if mv and bv else None
            posthoc[a] = {"mw_p": (round(mw["p"], 5) if mw["p"] is not None else None),
                          "paired_margin": round(paired, 4) if paired is not None else None,
                          "rank_biserial": rank_biserial(mv, bv),
                          "cohens_d": (round(cohens_d(mv, bv), 3)
                                       if cohens_d(mv, bv) is not None else None)}
            pvals.append(mw["p"])
            pairs.append(a)
        adjusted = holm_bonferroni(pvals)
        for a, p in zip(pairs, adjusted):
            posthoc[a]["mw_p_holm"] = round(p, 5) if p is not None else None
        out[label] = {"per_algo": per_algo, "kruskal_wallis":
                      {"H": round(kw["H"], 3) if kw["H"] is not None else None,
                       "p": round(kw["p"], 6) if kw["p"] is not None else None,
                       "df": kw["df"]},
                      "posthoc_vs_mpcare": posthoc}
    return out


def benchmarks(rows: list[dict], conditions: list[str], name: str) -> dict:
    """Section 6.3 publishable-result checks, evaluated on the actual data.
    Each check runs only in the experiment designed for it."""
    b: dict = {}

    def cond_pdr(label, algo):
        return per_algo_stats([r for r in rows if r["condition"] == label],
                              algo, "chain_pdr")

    if name.startswith("exp2"):
        # Benchmark: MP-CARE PDR > 0.80 at 20% loss AND A* dist < 0.60 there.
        # NOTE: this presumes a topology with path diversity — on the bare
        # chain there are no alternate routes, so ALL routers follow the same
        # (1-loss)^hops law and the check degrades to a physics statement.
        # Run on chain AND diamond and compare.
        if "loss_0.20" in conditions:
            m = cond_pdr("loss_0.20", "mpcare")
            s = cond_pdr("loss_0.20", "astar_dist")
            sc = rows[0]["scenario"] if rows else "?"
            b[f"exp2_pdr80_at_20pct_loss[{sc}]"] = {
                "mpcare_mean": m.get("mean"), "astar_dist_mean": s.get("mean"),
                "mpcare_above_080": bool(m.get("mean", 0) > 0.80),
                "astar_dist_below_060": bool(s.get("mean", 1) < 0.60)}
    if name == "exp3_latency":
        # Needs control-plane latency in the model -> honest n/a
        b["exp3_reroute_faster_at_50ms"] = {
            "result": "n/a", "reason": "control messages are not delayed by "
            "injected_latency_ms in the current model (data-plane only), so "
            "the periodic-vs-event re-route latency cannot be measured here"}
    if name == "exp5_mobility":
        # median LET relative error < 30% for v <= 15 m/s (section 6.3). The
        # raw number is dominated by the 1 s beacon quantization of break time
        # (see NOTES/Phase_4.md); we also report the quantization-filtered
        # variant and the rank correlation (does the predictor at least ORDER
        # links correctly?).
        vals = [r["let_median_rel_err_vle15"] for r in rows
                if r.get("let_median_rel_err_vle15") is not None]
        vals_q = [r["let_median_rel_err_vle15_act2"] for r in rows
                  if r.get("let_median_rel_err_vle15_act2") is not None]
        sp = [r["let_spearman"] for r in rows
              if r.get("let_spearman") is not None]
        b["exp5_let_err_below_30pct_vle15"] = {
            "median_rel_err_across_runs": round(statistics.median(vals), 3)
                if vals else None,
            "median_rel_err_filtered_act_ge_2s": round(statistics.median(vals_q), 3)
                if vals_q else None,
            "median_spearman_rho": round(statistics.median(sp), 3) if sp else None,
            "n_runs": len(vals),
            "pass": bool(vals) and statistics.median(vals) < 0.30}
    if name == "exp6_failure":
        # route recovery < 10 s after a single node failure
        rec = [r["recovery_s"] for r in rows
               if r["condition"] == "kill_1"
               and isinstance(r["recovery_s"], (int, float))]
        b["exp6_recovery_below_10s_single_kill"] = {
            "median_recovery_s": round(statistics.median(rec), 2) if rec else None,
            "n": len(rec),
            "pass": bool(rec) and statistics.median(rec) < 10.0}
    return b


# ------------------------------------------------------------------- graphs

def _style_axes(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)


def _series(rows, conditions, algos, metric):
    xs, series = {}, {}
    for label in conditions:
        cr = [r for r in rows if r["condition"] == label]
        for a in algos:
            st = per_algo_stats(cr, a, metric)
            if st.get("n"):
                series.setdefault(a, ([], [], []))
                series[a][0].append(label)
                series[a][1].append(st["mean"])
                series[a][2].append(st["ci95"])
    return series


def _line_plot(path, rows, conditions, algos, metric, xlabel, title, ylab):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ser = _series(rows, conditions, algos, metric)
    for a in algos:
        if a not in ser:
            continue
        xs = [float(c.split("_")[-1]) for c in ser[a][0]]
        m, ci = ser[a][1], ser[a][2]
        ax.plot(xs, m, marker="o", ms=4, color=ALGO_COLOR.get(a), label=a)
        ax.fill_between(xs, [x - c for x, c in zip(m, ci)],
                        [x + c for x, c in zip(m, ci)],
                        color=ALGO_COLOR.get(a), alpha=0.15)
    _style_axes(ax, xlabel, ylab, title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=600)
    plt.close(fig)


def graph_41_44(rows, name, loss_labels, algos):
    _line_plot(OUT / f"{name}_g41_pdr_vs_loss.png", rows, loss_labels, algos,
               "chain_pdr", "injected packet loss", "Graph 4.1: PDR vs packet loss",
               "PDR (mean ± 95% CI)")
    _line_plot(OUT / f"{name}_g42_completion_vs_loss.png", rows, loss_labels,
               algos, "last_delivery_s", "injected packet loss",
               "Graph 4.2: last-packet completion time vs loss\n(waypoint mission "
               "time is routing-independent by construction — see notes)",
               "last delivery (s)")
    _line_plot(OUT / f"{name}_g43_joint_vs_loss.png", rows, loss_labels, algos,
               "joint_objective", "injected packet loss",
               "Graph 4.3: joint objective (PDR / completion time) vs loss", "objective")
    # 4.4: re-route counts
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ser = _series(rows, loss_labels, algos, "reroutes")
    for a in algos:
        if a not in ser:
            continue
        xs = [float(c.split("_")[-1]) for c in ser[a][0]]
        ax.plot(xs, ser[a][1], marker="s", ms=4, color=ALGO_COLOR.get(a), label=a)
    _style_axes(ax, "injected packet loss", "re-routes per run",
                "Graph 4.4: re-route count vs loss (event-triggered vs periodic)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_g44_reroutes_vs_loss.png", dpi=600)
    plt.close(fig)


def graph_45(rows, name, speed_labels, algos):
    _line_plot(OUT / f"{name}_g45_pdr_vs_speed.png", rows, speed_labels, algos,
               "chain_pdr", "flight speed (m/s)", "Graph 4.5: PDR vs UAV velocity",
               "PDR (mean ± 95% CI)")


def graph_46(rows, name, kill_labels, algos):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    data, labels_, positions_, colors = [], [], [], []
    for i, a in enumerate(algos):
        vals = [r["recovery_s"] for r in rows
                if r["condition"] == "kill_1" and r["algorithm"] == a
                and isinstance(r["recovery_s"], (int, float))]
        if vals:
            data.append(vals)
            labels_.append(a)
            positions_.append(i)
            colors.append(ALGO_COLOR.get(a, "#888"))
    if data:
        bp = ax.boxplot(data, positions=positions_, widths=0.55, patch_artist=True,
                        showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks(positions_)
        ax.set_xticklabels(labels_)
    _style_axes(ax, "algorithm", "route recovery time (s)",
                "Graph 4.6: recovery after 1 node failure (grid, n=10)")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_g46_recovery_box.png", dpi=600)
    plt.close(fig)


def graph_47(env_rows, name, loss_labels, speed_labels):
    """Benefit envelope: cell = mean PDR margin (mpcare - gpsr), star = MW p<0.05.
    loss_labels/speed_labels are the raw value lists; cell condition labels are
    rebuilt exactly as run_experiment built them: f'L{l:.2f}_S{s:.0f}'."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    margin = [[0.0] * len(speed_labels) for _ in loss_labels]
    stars = [[""] * len(speed_labels) for _ in loss_labels]
    for i, ll in enumerate(loss_labels):
        for j, sl in enumerate(speed_labels):
            cell = [r for r in env_rows
                    if r["condition"] == f"L{ll:.2f}_S{sl:.0f}"]
            m = [r["chain_pdr"] for r in cell if r["algorithm"] == "mpcare"]
            g = [r["chain_pdr"] for r in cell if r["algorithm"] == "gpsr"]
            if m and g:
                margin[i][j] = statistics.mean(m) - statistics.mean(g)
                mw = mannwhitneyu(m, g)
                if mw["p"] is not None and mw["p"] < 0.05:
                    stars[i][j] = "*" if margin[i][j] > 0 else "*-"
    fig, ax = plt.subplots(figsize=(7, 4.8))
    im = ax.imshow(margin, cmap="RdBu", vmin=-0.5, vmax=0.5, origin="lower",
                   aspect="auto")
    ax.set_xticks(range(len(speed_labels)), [f"{s:.0f}" for s in speed_labels])
    ax.set_yticks(range(len(loss_labels)), [f"{int(l * 100)}%" for l in loss_labels])
    for i in range(len(loss_labels)):
        for j in range(len(speed_labels)):
            ax.text(j, i, f"{margin[i][j]:+.2f}{stars[i][j]}",
                    ha="center", va="center", fontsize=9)
    ax.set_xlabel("flight speed (m/s)")
    ax.set_ylabel("injected packet loss")
    ax.set_title("Graph 4.7: benefit envelope — PDR margin (MP-CARE − GPSR)\n"
                 "* = Mann-Whitney p < 0.05 (Holm across cells not applied)",
                 fontsize=9)
    fig.colorbar(im, ax=ax, label="PDR margin")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_g47_benefit_envelope.png", dpi=600)
    plt.close(fig)


# ----------------------------------------------------------------- campaign

LOSS_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
LAT_MS_VALUES = [0.0, 10.0, 25.0, 50.0, 100.0, 150.0, 200.0]
SPEED_VALUES = [5.0, 10.0, 15.0, 20.0, 25.0]
KILL_VALUES = [0, 1, 2, 3]


def build_experiments(args):
    n = args.trials
    s0 = args.seed0
    # Phase 8 (Task 1): exp2b/exp5 gain the QLGR learned baseline (+full algo
    # list on the diamond); exp2/exp3/envelope keep their legacy rosters so
    # their published numbers stay byte-identical to the current paper.
    algos4 = ["astar_dist", "gpsr", "olsr", "mpcare"]
    algos7 = list(run_sim.ALGORITHMS)
    algos6 = ["astar_dist", "gpsr", "aodv", "olsr", "qlgr", "mpcare"]
    speed_ext = list(SPEED_VALUES) + [30.0, 40.0, 50.0]   # Task 3a: aerial speeds
    exps = {}
    exps["exp2_loss"] = dict(
        conditions=[{"label": f"loss_{v:.2f}", "scenario": "chain",
                     "injected_loss": v} for v in LOSS_VALUES],
        algorithms=algos4, duration=90.0, base={},
        stats_metric="chain_pdr", graph_fn="loss")
    exps["exp2b_loss_diamond"] = dict(
        # same sweep on the diamond: WITH path diversity the quality-aware
        # routers can actually route around loss (the section 6.3 benchmark
        # presumes exactly this; the chain result documents the no-diversity
        # physics floor)
        conditions=[{"label": f"loss_{v:.2f}", "scenario": "diamond",
                     "injected_loss": v} for v in LOSS_VALUES],
        algorithms=algos7, duration=90.0, base={},
        stats_metric="chain_pdr", graph_fn="loss")
    exps["exp3_latency"] = dict(
        conditions=[{"label": f"lat_{v:.0f}", "scenario": "chain",
                     "injected_latency_ms": v} for v in LAT_MS_VALUES],
        algorithms=algos4, duration=90.0, base={},
        stats_metric="avg_latency_ms", graph_fn=None)
    exps["exp5_mobility"] = dict(
        conditions=[{"label": f"speed_{v:.0f}", "scenario": "mobility",
                     "speed": v} for v in speed_ext],
        algorithms=algos6, duration=60.0, base={},
        stats_metric="chain_pdr", graph_fn="speed")
    exps["exp6_failure"] = dict(
        conditions=[{"label": f"kill_{k}", "scenario": "grid", "kills": k}
                    for k in KILL_VALUES],
        algorithms=list(run_sim.ALGORITHMS), duration=90.0, base={"n_uav": 10},
        stats_metric="chain_pdr", graph_fn="kill")
    exps["envelope"] = dict(
        conditions=[{"label": f"L{l:.2f}_S{s:.0f}", "scenario": "mobility",
                     "injected_loss": l, "speed": s}
                    for l in (0.0, 0.10, 0.20, 0.30) for s in SPEED_VALUES],
        algorithms=["gpsr", "mpcare"], duration=60.0, base={},
        stats_metric="chain_pdr", graph_fn="envelope")
    if args.quick:
        for e in exps.values():
            e["conditions"] = e["conditions"][:2]
    return exps


def main():
    ap = argparse.ArgumentParser(description="Phase 4 stress-test suite (plan §6)")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--only", default="",
                    help="comma list: exp2_loss,exp3_latency,exp5_mobility,"
                         "exp6_failure,envelope")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--quick", action="store_true",
                    help="2 conditions per experiment (smoke test)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    exps = build_experiments(args)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        exps = {k: v for k, v in exps.items() if k in keep}

    all_benchmarks: dict = {}
    for name, spec in exps.items():
        print(f"\n=== {name}: {len(spec['conditions'])} conditions x "
              f"{len(spec['algorithms'])} algorithms x {args.trials} trials ===")
        conds = [dict(c) for c in spec["conditions"]]
        rows = run_experiment(name, conds, spec["algorithms"], args.trials,
                              args.seed0, spec["duration"], spec["base"])

        # derived metric: the section 6.2 joint objective PDR / completion time
        for r in rows:
            if isinstance(r["last_delivery_s"], (int, float)) and r["last_delivery_s"]:
                r["joint_objective"] = round(
                    r["chain_pdr"] / max(r["last_delivery_s"] - run_sim.SEND_FROM, 1.0), 5)
            else:
                r["joint_objective"] = None

        with open(OUT / f"{name}{tag}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=ROW_KEYS + ["joint_objective"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)

        labels = [c["label"] for c in spec["conditions"]]
        algos = spec["algorithms"]
        metric = spec["stats_metric"]
        summary = {
            "experiment": name, "n_trials": args.trials, "seed0": args.seed0,
            "duration_s": spec["duration"], "algorithms": algos,
            "conditions": labels,
            "primary_metric": metric,
            "primary_stats": condition_stats(rows, labels, algos, metric),
            "pdr_stats": condition_stats(rows, labels, algos, "chain_pdr"),
            "benchmarks": {},
        }
        if name.startswith(("exp2", "exp3", "exp5", "exp6")):
            summary["benchmarks"] = benchmarks(rows, labels, name)
            all_benchmarks.update({f"{name}.{k}": v
                                   for k, v in summary["benchmarks"].items()})
        with open(OUT / f"{name}{tag}_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # graphs
        if spec["graph_fn"] == "loss":
            graph_41_44(rows, name, labels, algos)
        elif spec["graph_fn"] == "speed":
            graph_45(rows, name, labels, algos)
        elif spec["graph_fn"] == "kill":
            graph_46(rows, name, labels, algos)
        elif spec["graph_fn"] == "envelope":
            graph_47(rows, name,
                     [c["injected_loss"] for c in spec["conditions"]],
                     SPEED_VALUES)
        print(f"[{name}] wrote {name}{tag}.csv/_summary.json"
              + (f" + graphs ({spec['graph_fn']})" if spec["graph_fn"] else ""))

        # console headline
        ref = "mpcare"
        print(f"\n  {'condition':<16}", end="")
        for a in algos:
            print(f"{a:<14}", end="")
        print()
        for label in labels:
            print(f"  {label:<16}", end="")
            cr = [r for r in rows if r["condition"] == label]
            for a in algos:
                st = per_algo_stats(cr, a, metric if metric != "chain_pdr" or name != "exp3_latency" else "chain_pdr")
                print(f"{st.get('mean', float('nan')):.3f}±{st.get('ci95', 0):.3f}   "
                      if st.get("n") else f"{'-':<14}", end="")
            print()
        ph = summary["primary_stats"]
        hl = [(a, d["mw_p_holm"], d["paired_margin"])
              for a, d in ph[labels[-1]]["posthoc_vs_mpcare"].items()
              if d["mw_p_holm"] is not None]
        if hl:
            print(f"  post-hoc vs mpcare @{labels[-1]} (Holm): "
                  + ", ".join(f"{a}: p={p:g}, margin={m:+.3f}" for a, p, m in hl))

    if all_benchmarks:
        print("\n=== section 6.3 benchmark checks ===")
        for k, v in all_benchmarks.items():
            if isinstance(v, dict) and ("pass" in v or "result" in v):
                verdict = v.get("result") or ("PASS" if v.get("pass") else "MISS")
                print(f"  [{verdict}] {k}")
        with open(OUT / f"phase4_benchmarks{tag}.json", "w", encoding="utf-8") as f:
            json.dump(all_benchmarks, f, indent=2)
    print("\nDONE — data in", OUT)


if __name__ == "__main__":
    main()
