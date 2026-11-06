#!/usr/bin/env python3
"""Phase 8 — modern-baseline + high-mobility campaigns (improvement.md Tasks 1, 3).

  1. QLGR (modern learned baseline): tabular Q-geographic routing (QLGR
     family; adaptation of Qiu et al. 2021, KSII TIIS 15(11)) — trained on
     seeds 900-909 with epsilon-greedy exploration, tables merged state-wise,
     then evaluated inference-only on seeds 1000-1029 (30 fresh seeds).
     Zero control packets by construction; per-tick compute measured with
     the same instrumentation as MP-CARE's.
     Tables: a per-node table for the role-fixed diamond; ONE global
     'dynamic' table (class-space transfer; merged over the mobility and
     patrol training runs) used for mobility/patrol evaluation.
  2. High-speed mobility sweep: 30/40/50 m/s added to 5-25 m/s (Task 3a) —
     the speed-activation test of the prediction layer.
  3. Patrol experiment: 10 UAVs on crossing patrol lanes in a 1000 x 1000 m
     box (non-diamond dynamic geometry, Task 3b).

Outputs (data/processed/): phase8_qlgr_*.json, phase8_<name>.csv +
phase8_<name>_summary.json + phase8_<name>_pdr.png.

Usage:
  py -3.13 scripts/run_phase8.py --train                    # training runs
  py -3.13 scripts/run_phase8.py --only diamond,mobility,patrol
  py -3.13 scripts/run_phase8.py --trials 3 --tag _smoke    # smoke test
"""

from __future__ import annotations

import argparse
import csv
import json
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
sys.path.insert(0, str(ROOT / "src" / "qlgr_routing"))

import run_sim  # noqa: E402
import qlgr_routing.core as qcore  # noqa: E402
from stats_utils import (mannwhitneyu, kruskal_wallis, holm_bonferroni,  # noqa: E402
                         cohens_d, rank_biserial, mean_ci95)

OUT = ROOT / "data" / "processed"

TRAIN_SEED0 = 900        # 10 training seeds: 900-909 (disjoint from eval)
TRAIN_SEEDS = 10
EVAL_SEED0 = 1000        # identical to the Phase-4 evaluation seeds
ALGO_COLOR = {"astar_dist": "#8c8c8c", "gpsr": "#ed7d31", "aodv": "#4472c4",
              "olsr": "#70ad47", "astar_etx": "#ffc000", "qlgr": "#7030a0",
              "mpcare": "#c00000"}

TRAIN_SPECS = {
    "diamond": dict(duration=90.0, scenario="diamond", injected_loss=0.10),
    "mobility": dict(duration=60.0, scenario="mobility", speed=15.0),
    "patrol": dict(duration=90.0, scenario="patrol", speed=20.0, n_uav=10),
}

CAMPAIGN_SPECS = {
    "diamond": dict(
        conditions=[dict(label=f"loss_{v:.2f}", scenario="diamond",
                         injected_loss=v) for v in (0.0, 0.05, 0.10, 0.20)],
        algorithms=["astar_dist", "gpsr", "aodv", "olsr", "astar_etx",
                    "qlgr", "mpcare"],
        duration=90.0, table="diamond", metric="chain_pdr"),
    "mobility": dict(
        conditions=[dict(label=f"speed_{v:.0f}", scenario="mobility", speed=v)
                    for v in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0)],
        algorithms=["astar_dist", "gpsr", "aodv", "olsr", "qlgr", "mpcare"],
        duration=60.0, table="dynamic", metric="chain_pdr"),
    "patrol": dict(
        conditions=[dict(label="patrol", scenario="patrol", speed=20.0,
                         n_uav=10)],
        algorithms=["astar_dist", "gpsr", "aodv", "olsr", "qlgr", "mpcare"],
        duration=90.0, table="dynamic", metric="chain_pdr"),
}


# ---------------------------------------------------------------- training

ROW_KEYS = ["seed", "trial", "algorithm", "condition", "chain_pdr", "pdr",
            "avg_latency_ms", "last_delivery_s", "reroutes", "scenario",
            "cpu_ms_per_node_tick", "control_pkts_per_node_s",
            "qlgr_route_changes", "qlgr_table_size"]


# ------------------------------------------------------------------ training
def _train_table(scenario: str, seed: int) -> dict[int, dict]:
    qcore.clear_batch()
    run_sim.run(seed, ROOT / "data" / "raw", write_csv=False,
                algorithm="qlgr",
                qlgr_opts={"learn": True, "seed": seed, "eps": 0.15},
                **TRAIN_SPECS[scenario])
    return qcore.live_tables()


def train() -> dict:
    """10 training seeds per scenario; merge state-wise (per-node for the
    role-fixed diamond; global for the symmetric dynamic scenarios)."""
    from qlgr_routing.core import merge_tables
    tables: dict = {}
    for scenario in TRAIN_SPECS:
        per_node: dict[str, list[dict]] = {}
        all_tables: list[dict] = []
        t0 = time.perf_counter()
        for k in range(TRAIN_SEEDS):
            tables_k = _train_table(scenario, TRAIN_SEED0 + k)
            for nid, td in sorted(tables_k.items()):
                per_node.setdefault(str(nid), []).append(td)
            all_tables.extend(tables_k.values())
        if scenario == "diamond":
            merged = {nid: merge_tables(tds)
                      for nid, tds in sorted(per_node.items())}
        else:
            merged = {"global": merge_tables(all_tables)}
        tables[scenario] = merged
        path = OUT / f"phase8_qlgr_{scenario}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"scenario": scenario,
                       "train_seeds": [TRAIN_SEED0 + i
                                       for i in range(TRAIN_SEEDS)],
                       "params": {"eps": 0.15, "alpha": 0.5, "gamma": 0.9},
                       "table": merged}, f, indent=2)
        n_states = sum(len(v) for v in merged.values())
        print(f"[train] {scenario:<8} 10 seeds, {n_states} state entries, "
              f"({time.perf_counter() - t0:.0f}s) -> {path.name}")
    return tables


def merged_dynamic_tables(tables: dict) -> dict:
    """Mobility + patrol -> one global 'dynamic' table (class-space transfer)."""
    from qlgr_routing.core import merge_tables
    parts = [tables[s]["global"] for s in ("mobility", "patrol") if s in tables]
    if not parts:
        return {}
    dyn = merge_tables(parts)
    path = OUT / "phase8_qlgr_dynamic.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"scenario": "dynamic(mobility+patrol)",
                   "table": {"global": dyn}}, f, indent=2)
    return {"dynamic": {"global": dyn}}


def load_saved_tables(saved: Path) -> dict:
    tables = {}
    for scenario in TRAIN_SPECS:
        p = saved / f"phase8_qlgr_{scenario}.json"
        if p.exists():
            tables[scenario] = json.loads(p.read_text(encoding="utf-8"))["table"]
    dyn_p = saved / "phase8_qlgr_dynamic.json"
    if dyn_p.exists():
        tables["dynamic"] = json.loads(
            dyn_p.read_text(encoding="utf-8"))["table"]
    return tables


# ---------------------------------------------------------------- campaigns
def _flatten(r: dict, condition: str, params: dict) -> dict:
    ev = r.get("trigger_events") or {}
    row = {"seed": r["seed"], "algorithm": r["algorithm"],
           "condition": condition,
           "chain_pdr": r.get("chain_pdr"), "pdr": r.get("pdr"),
           "avg_latency_ms": r.get("avg_latency_ms"),
           "last_delivery_s": r.get("last_delivery_s"),
           "reroutes": sum(ev.values()) if ev else 0,
           "cpu_ms_per_node_tick": r.get("cpu_ms_per_node_tick"),
           "control_pkts_per_node_s": r.get("control_pkts_per_node_s"),
           "qlgr_route_changes": r.get("qlgr_route_changes"),
           "qlgr_table_size": r.get("qlgr_table_size")}
    row.update(params)
    return row


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
            posthoc[a] = {"mw_p": (round(mw["p"], 5) if mw["p"] is not None
                                   else None),
                          "paired_margin": round(paired, 4)
                          if paired is not None else None,
                          "rank_biserial": rank_biserial(mv, bv),
                          "cohens_d": (round(cohens_d(mv, bv), 3)
                                       if cohens_d(mv, bv) is not None else None)}
            pvals.append(mw["p"])
            pairs.append(a)
        adjusted = holm_bonferroni(pvals)
        for a, p in zip(pairs, adjusted):
            posthoc[a]["mw_p_holm"] = round(p, 5) if p is not None else None
        out[label] = {"per_algo": per_algo,
                      "kruskal_wallis": {"H": round(kw["H"], 3)
                                         if kw["H"] is not None else None,
                                         "p": round(kw["p"], 6)
                                         if kw["p"] is not None else None,
                                         "df": kw["df"]},
                      "posthoc_vs_mpcare": posthoc}
    return out


def run_campaign(name: str, spec: dict, trials: int, seed0: int,
                 tables: dict, tag: str = "") -> list[dict]:
    tk = spec["table"]
    table = tables.get(tk, {}).get("global") \
        if tk == "dynamic" else tables.get("diamond")
    if table is None:
        sys.exit(f"trained QLGR tables missing for campaign {name} — "
                 "run with --train first")
    rows: list[dict] = []
    t_start = time.perf_counter()
    for cond in spec["conditions"]:
        label = cond["label"]
        cond_kwargs = {k: v for k, v in cond.items() if k != "label"}
        params = {"scenario": cond_kwargs.get("scenario", ""),
                  "speed_mps": cond_kwargs.get("speed"),
                  "loss": cond_kwargs.get("injected_loss") or 0.0}
        for algo in spec["algorithms"]:
            for trial in range(trials):
                seed = seed0 + trial
                r = run_sim.run(seed, ROOT / "data" / "raw",
                                duration=spec["duration"], write_csv=False,
                                algorithm=algo,
                                qlgr_opts={"learn": False, "seed": seed,
                                           "eps": 0.0, "table": table},
                                **cond_kwargs)
                rows.append(_flatten(r, label, params))
            print(f"[{name}] {label:<10} {algo:<11} {trials} trials "
                  f"({time.perf_counter() - t_start:.0f}s elapsed)")
    with open(OUT / f"phase8_{name}{tag}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    labels = [c["label"] for c in spec["conditions"]]
    algos = spec["algorithms"]
    summary = {
        "experiment": f"phase8_{name}", "n_trials": trials,
        "seed0": seed0, "duration_s": spec["duration"],
        "algorithms": algos, "conditions": labels,
        "primary_metric": spec["metric"],
        "primary_stats": condition_stats(rows, labels, algos, spec["metric"]),
        "pdr_stats": condition_stats(rows, labels, algos, "chain_pdr"),
        "cpu_stats": condition_stats(rows, labels, algos,
                                     "cpu_ms_per_node_tick"),
        "benchmarks": run_campaign_benchmarks(name, rows, labels, algos),
    }
    with open(OUT / f"phase8_{name}{tag}_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _plot(name, algos, spec, rows, labels, tag)
    print(f"[{name}] wrote phase8_{name}{tag}.csv/_summary.json + figure")
    return rows


def run_campaign_benchmarks(name: str, rows: list[dict], labels: list[str],
                            algos: list[str]) -> dict:
    """Publishable-result checks evaluated on the measured data (Task 1/3)."""
    b: dict = {}

    def pdr(label):
        return [r["chain_pdr"] for r in rows if r["condition"] == label
                and isinstance(r["chain_pdr"], (int, float))]

    def q_pdr(label):
        return [r["chain_pdr"] for r in rows
                if r["condition"] == label and r["algorithm"] == "qlgr"
                and isinstance(r["chain_pdr"], (int, float))]

    def m_pdr(label):
        return [r["chain_pdr"] for r in rows
                if r["condition"] == label and r["algorithm"] == "mpcare"
                and isinstance(r["chain_pdr"], (int, float))]

    if name == "diamond" and "loss_0.10" in labels:
        mv, qv = m_pdr("loss_0.10"), q_pdr("loss_0.10")
        mw = mannwhitneyu(mv, qv)
        b["qlgr_not_ahead_at_10pct_loss"] = {
            "mpcare_mean": round(statistics.mean(mv), 4),
            "qlgr_mean": round(statistics.mean(qv), 4),
            "mw_p": round(mw["p"], 5) if mw["p"] is not None else None,
            "mpcare_ahead": statistics.mean(mv) >= statistics.mean(qv)}
        cpu_q = [r["cpu_ms_per_node_tick"] for r in rows
                 if r["algorithm"] == "qlgr"
                 and isinstance(r["cpu_ms_per_node_tick"], (int, float))]
        cpu_m = [r["cpu_ms_per_node_tick"] for r in rows
                 if r["algorithm"] == "mpcare"
                 and isinstance(r["cpu_ms_per_node_tick"], (int, float))]
        if cpu_q and cpu_m:
            b["qlgr_cpu_vs_mpcare"] = {
                "qlgr_ms": round(statistics.mean(cpu_q), 4),
                "mpcare_ms": round(statistics.mean(cpu_m), 4)}
    if name == "mobility":
        hi = [l for l in labels if float(l.split("_")[-1]) >= 30.0]
        for h in hi:
            mv, qv = m_pdr(h), q_pdr(h)
            if mv and qv:
                mw = mannwhitneyu(mv, qv)
                b[f"mpcare_vs_qlgr[{h}]"] = {
                    "mpcare_mean": round(statistics.mean(mv), 4),
                    "qlgr_mean": round(statistics.mean(qv), 4),
                    "mw_p": round(mw["p"], 5) if mw["p"] is not None else None,
                    "margin_pp": round(100 * (statistics.mean(mv)
                                              - statistics.mean(qv)), 1)}
    if name == "patrol":
        for a in ("gpsr", "aodv", "olsr", "astar_dist", "qlgr"):
            av = [r["chain_pdr"] for r in rows
                  if r["algorithm"] == a and isinstance(r["chain_pdr"],
                                                        (int, float))]
            mv = m_pdr("patrol")
            if av and mv:
                mw = mannwhitneyu(mv, av)
                b[f"mpcare_vs_{a}_margin_pp"] = round(
                    100 * (statistics.mean(mv) - statistics.mean(av)), 1)
    return b


def _plot(name, algos, spec, rows, labels, tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metric = spec["metric"]
    x_var = "injected loss" if name == "diamond" else "speed"
    if len(labels) > 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for a in algos:
            xs, ms, cs = [], [], []
            for label in labels:
                st = per_algo_stats([r for r in rows
                                     if r["condition"] == label], a, metric)
                if st.get("n"):
                    xs.append(float(label.split("_")[-1]))
                    ms.append(st["mean"])
                    cs.append(st["ci95"])
            if xs:
                ax.plot(xs, ms, marker="o", ms=4, color=ALGO_COLOR.get(a),
                        label=a)
                ax.fill_between(xs, [m - c for m, c in zip(ms, cs)],
                                [m + c for m, c in zip(ms, cs)],
                                color=ALGO_COLOR.get(a), alpha=0.15)
        ax.set_xlabel("injected packet loss" if name == "diamond"
                      else "flight speed (m/s)")
        ax.set_ylabel("PDR (mean ± 95% CI)")
        ax.set_title(f"Phase 8: PDR vs {x_var} — modern learned baseline added",
                     fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / f"phase8_{name}{tag}_pdr.png", dpi=600)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        data, labels_, colors, pos = [], [], [], []
        for i, a in enumerate(algos):
            vals = [r[metric] for r in rows
                    if r["algorithm"] == a and isinstance(r[metric],
                                                          (int, float))]
            if vals:
                data.append(vals)
                labels_.append(a)
                colors.append(ALGO_COLOR.get(a, "#888"))
                pos.append(i)
        bp = ax.boxplot(data, positions=pos, widths=0.55, patch_artist=True,
                        showfliers=False)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_xticks(pos)
        ax.set_xticklabels(labels_, fontsize=8)
        ax.set_ylabel("PDR")
        ax.set_title("Phase 8: 10-UAV patrol lanes (1000 x 1000 m, "
                     "crossing paths, 20 m/s)", fontsize=10)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"phase8_{name}{tag}_pdr.png", dpi=600)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Phase 8 campaigns (Tasks 1+3)")
    ap.add_argument("--train", action="store_true",
                    help="run 10-seed QLGR training campaigns")
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=EVAL_SEED0)
    ap.add_argument("--only", default="",
                    help="comma list: diamond,mobility,patrol")
    ap.add_argument("--tag", default="", help="output filename suffix")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    scenarios = [s.strip() for s in args.only.split(",") if s.strip()]

    tables = load_saved_tables(OUT)
    need_table = any(s in scenarios for s in ("diamond", "mobility", "patrol"))
    if args.train:
        tables.update(train())
        tables.update(merged_dynamic_tables(tables))
    elif need_table and not tables:
        sys.exit("no trained QLGR tables found — run with --train first")
    for name, spec in CAMPAIGN_SPECS.items():
        if scenarios and name not in scenarios:
            continue
        print(f"\n=== phase8_{name}: {len(spec['conditions'])} conditions x "
              f"{len(spec['algorithms'])} algorithms x {args.trials} trials ===")
        run_campaign(name, spec, args.trials, args.seed0, tables, args.tag)
    print("\nDONE — data in", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
