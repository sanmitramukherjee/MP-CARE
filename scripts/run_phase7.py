#!/usr/bin/env python3
"""Phase 7 ablation + weight-sensitivity campaign (plan section 9.1-9.3).

ABLATIONS (section 9.1, exact table):
  A0 full MP-CARE (anchor)                A4 w_q = 0 (no congestion term)
  A1 predict_etx=False (current ETX only) A5 periodic re-route, 1 s
  A2 use_let=False (no LET -> no T2)      A6 periodic re-route, 5 s
  A3 delta=0, t_dwell=0 (no hysteresis)   A7 static route (lower bound)

Condition per plan: 30 trials, 5 UAVs, 10% injected loss (chain semantics).
Reported per section 9.1: the DELTA from full MP-CARE (A0) on PDR, mission
time, and re-route count. "Mission time": chain missions have no per-UAV
waypoint deadline, so the routing-dependent time metric is
delivery_completion_s = last_delivery_s - SEND_FROM (documented in
NOTES/Phase_7.md). The plan-literal chain window is STATIC while packets
live (release at t=30), which measurably zeroes the component ablations'
deltas (prediction/LET/hysteresis/queue have nothing to do) and the
GATING of periodic/static (but NOT the walk/recovery differences); the
supplementary mobility condition (same 10% loss, continuous 10 m/s flight)
is where the switch policies express themselves.

WEIGHT SENSITIVITY (section 9.3): anchor w_d = 1.0 (all other weights at
defaults); sweep each of w_e, w_t, w_q, w_h over {0, 0.25, 0.5, 1.0, 2.0,
4.0} -> 24 configs x 30 trials on the plan condition; Graph 7.2 = PDR vs
delivery-time scatter with Pareto-efficient configs highlighted.

Outputs (data/processed/):
  phase7_ablation_trials.csv / _summary.json / _g71_ablation.png
  phase7_weights_trials.csv  / _summary.json / _g72_pareto.png
Usage:
  py -3.13 scripts/run_phase7.py                     # full campaign
  py -3.13 scripts/run_phase7.py --trials 3 --tag _smoke
  py -3.13 scripts/run_phase7.py --only ablation|weights
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
sys.path.insert(0, str(ROOT / "src" / "aodv_routing"))
sys.path.insert(0, str(ROOT / "src" / "olsr_routing"))
sys.path.insert(0, str(ROOT / "src" / "logger"))

import run_sim  # noqa: E402
from stats_utils import (mannwhitneyu, holm_bonferroni, mean_ci95)  # noqa: E402

OUT = ROOT / "data" / "processed"
TRIALS = 30
SEED0 = 5000
DURATION = 90.0
LOSS = 0.10
N_UAV = 5

# plan section 9.1 table: name -> mpcare_opts (A0 = full MP-CARE)
ABLATIONS: list[tuple[str, dict]] = [
    ("A0_full", {}),
    ("A1_no_etx_pred", {"predict_etx": False}),
    ("A2_no_let", {"use_let": False}),
    ("A3_no_hysteresis", {"trigger_params": {"delta": 0.0, "t_dwell": 0.0}}),
    ("A4_no_congestion", {"weights": {"w_q": 0.0}}),
    ("A5_periodic_1s", {"reroute_mode": "periodic", "period_s": 1.0}),
    ("A6_periodic_5s", {"reroute_mode": "periodic", "period_s": 5.0}),
    ("A7_static", {"reroute_mode": "static"}),
]

# plan section 9.3: anchor w_d = 1.0; sweep the other four weights
SWEEP_VALUES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
WEIGHT_NAMES = ("w_e", "w_t", "w_q", "w_h")

METRICS = ("pdr", "delivery_completion_s", "reroutes_total")

ABL_KEYS = ["condition", "variant", "seed", "trial", "pdr",
            "packets_sent", "packets_delivered", "delivery_completion_s",
            "reroutes_total", "void_recovery_events", "void_walk_drops",
            "run_wall_s"]


def _opts(spec: dict) -> dict:
    """run_sim passes mpcare_opts straight into MpcareRouter(**opts); build
    real TriggerParams objects for the A3 variant."""
    o = dict(spec)
    if "trigger_params" in o:
        from mpcare_agent.event_trigger import TriggerParams
        o["trigger_params"] = TriggerParams(**o["trigger_params"])
    return o


def run_variant(opts: dict, seed: int, mobility: bool) -> dict:
    if mobility:
        return run_sim.run(seed, ROOT / "data" / "raw", duration=DURATION,
                           write_csv=False, algorithm="mpcare",
                           scenario="mobility", speed=10.0,
                           injected_loss=LOSS, mpcare_opts=_opts(opts))
    return run_sim.run(seed, ROOT / "data" / "raw", duration=DURATION,
                       write_csv=False, algorithm="mpcare", scenario="chain",
                       injected_loss=LOSS, mpcare_opts=_opts(opts))


def flatten(r: dict, condition: str, variant: str, seed: int, k: int,
            wall: float) -> dict:
    trig = r.get("trigger_events", {})
    last = r.get("last_delivery_s")
    return {
        "condition": condition, "variant": variant, "seed": seed, "trial": k,
        "pdr": r["pdr"], "packets_sent": r["packets_sent"],
        "packets_delivered": r["packets_delivered"],
        "delivery_completion_s": (round(last - run_sim.SEND_FROM, 2)
                                  if last is not None else None),
        "reroutes_total": sum(trig.values()),
        "void_recovery_events": r.get("void_recovery_events", 0),
        "void_walk_drops": r.get("void_walk_drops", 0),
        "run_wall_s": round(wall, 3),
    }


def _series(rows: list[dict], condition: str, variant: str,
            metric: str) -> list[float]:
    return [float(r[metric]) for r in rows
            if r["condition"] == condition and r["variant"] == variant
            and r[metric] is not None]


# ------------------------------------------------------------------ ablation
def ablation_campaign(trials: int, seed0: int, tag: str) -> None:
    t_start = time.perf_counter()
    rows: list[dict] = []
    for cond, mobility in (("chain_loss10", False), ("mobility_loss10", True)):
        for name, opts in ABLATIONS:
            for k in range(trials):
                seed = seed0 + k
                t0 = time.perf_counter()
                r = run_variant(opts, seed, mobility)
                rows.append(flatten(r, cond, name, seed, k,
                                    time.perf_counter() - t0))
            print(f"[{cond}] {name:<18} {trials} trials "
                  f"pdr={statistics.mean(_series(rows, cond, name, 'pdr')):.3f} "
                  f"({time.perf_counter() - t_start:.0f}s elapsed)")

    name = f"phase7_ablation{tag}"
    _write_rows(rows, f"{name}_trials.csv")

    summary: dict = {"experiment": "phase7_ablations",
                     "trials": trials, "seed0": seed0, "loss": LOSS,
                     "n_uav": N_UAV, "duration_s": DURATION,
                     "variants": {n: o for n, o in ABLATIONS},
                     "mission_time_note": "delivery_completion_s = "
                     "last_delivery_s - SEND_FROM (routing-dependent time "
                     "metric; chain missions have no waypoint deadline)",
                     "stats": {}}
    for cond in ("chain_loss10", "mobility_loss10"):
        st: dict = {"A0_full": _anchor_stats(rows, cond)}
        pvals: list[float | None] = []
        names: list[str] = []
        for vname, _ in ABLATIONS[1:]:
            st[vname] = _variant_stats(rows, cond, vname, st["A0_full"])
            pvals.append(st[vname]["mw_p_vs_full"])
            names.append(vname)
        holm = holm_bonferroni(pvals)
        for vname, hp in zip(names, holm):
            st[vname]["mw_p_holm_vs_full"] = hp
        summary["stats"][cond] = st

    ablation_graph(rows, name)
    with open(OUT / f"{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {name}_trials.csv / _summary.json")
    _print_ablation_verdict(summary)


def _anchor_stats(rows: list[dict], cond: str) -> dict:
    out = {}
    for m in METRICS:
        vals = _series(rows, cond, "A0_full", m)
        mc = mean_ci95(vals)
        out[m] = {"mean": round(statistics.mean(vals), 4) if vals else None,
                  "ci95": round(mc[1], 4) if mc else None, "n": len(vals)}
    return out


def _variant_stats(rows: list[dict], cond: str, vname: str,
                   anchor: dict) -> dict:
    out: dict = {}
    for m in METRICS:
        bv = _series(rows, cond, "A0_full", m)
        vv = _series(rows, cond, vname, m)
        mc = mean_ci95(vv)
        out[m] = {"mean": round(statistics.mean(vv), 4) if vv else None,
                  "ci95": round(mc[1], 4) if mc else None, "n": len(vv)}
        out[f"delta_{m}"] = (round(out[m]["mean"] - anchor[m]["mean"], 4)
                             if out[m]["mean"] is not None
                             and anchor[m]["mean"] is not None else None)
    p = None
    if out["pdr"]["n"] and anchor["pdr"]["n"]:
        p = mannwhitneyu(_series(rows, cond, vname, "pdr"),
                         _series(rows, cond, "A0_full", "pdr"))["p"]
    out["mw_p_vs_full"] = p
    return out


def ablation_graph(rows: list[dict], name: str) -> None:
    """Graph 7.1: horizontal bars, each variant's PDR relative to A0_full,
    sorted by impact (mobility condition carries the gating deltas)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, vals = [], []
    for vname, _ in ABLATIONS[1:]:
        m = statistics.mean(_series(rows, "mobility_loss10", vname, "pdr"))
        b = statistics.mean(_series(rows, "mobility_loss10", "A0_full", "pdr"))
        labels.append(vname)
        vals.append(m - b)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    labels = [labels[i] for i in order]
    vals = [vals[i] for i in order]
    bars = ax.barh(labels, vals,
                   color=["#c0392b" if v < 0 else "#27ae60" for v in vals])
    for b, v in zip(bars, vals):
        if v >= 0:
            ax.text(v + 0.002, b.get_y() + b.get_height() / 2, f"{v:+.3f}", va="center", ha="left", fontsize=8)
        else:
            ax.text(v + 0.004, b.get_y() + b.get_height() / 2, f"{v:+.3f}", va="center", ha="left", color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(left=min(vals) * 1.1)
    ax.set_xlabel("delta PDR vs full MP-CARE (mobility, 10% loss)")
    ax.set_title("Graph 7.1: ablation impact, sorted (plan section 9.2)")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_g71_ablation.png", dpi=600)
    plt.close(fig)
    print(f"chart: {OUT / f'{name}_g71_ablation.png'}")


def _print_ablation_verdict(summary: dict) -> None:
    print("\n== section 9.1 deltas vs full MP-CARE (A0) ==")
    for cond, st in summary["stats"].items():
        print(f"-- {cond}")
        a0 = st["A0_full"]
        print(f"   A0_full           pdr={a0['pdr']['mean']:.3f}"
              f"+-{a0['pdr']['ci95']:.3f} "
              f"t={a0['delivery_completion_s']['mean']}s "
              f"reroutes={a0['reroutes_total']['mean']}")
        for vname, _ in ABLATIONS[1:]:
            s = st[vname]
            hp = s["mw_p_holm_vs_full"]
            print(f"   {vname:<17} dpdr={s['delta_pdr']:+.3f} "
                  f"dt={s['delta_delivery_completion_s']:+.2f}s "
                  f"dreroute={s['delta_reroutes_total']:+.1f} "
                  f"holm_p={hp if hp is None else round(hp, 5)}")


# ------------------------------------------------------------------- weights
def weight_campaign(trials: int, seed0: int, tag: str) -> None:
    t_start = time.perf_counter()
    configs: list[tuple[str, dict]] = []
    for wn in WEIGHT_NAMES:
        for v in SWEEP_VALUES:
            label = f"{wn}={v}"
            configs.append((label, {"weights": {wn: v}}))
    rows: list[dict] = []
    # plan condition (chain) + mobility: the chain window is STATIC while
    # packets live, so all 24 configs tie there BY CONSTRUCTION (measured in
    # the smoke); the mobility condition lets the weights actually re-rank
    # candidates mid-flight, which is what makes Graph 7.2's Pareto a real
    # trade-off surface instead of a single point.
    for cond, mobility in (("chain_loss10", False), ("mobility_loss10", True)):
        for i, (label, cfg) in enumerate(configs):
            for k in range(trials):
                seed = seed0 + k
                t0 = time.perf_counter()
                r = run_variant(cfg, seed, mobility)
                rows.append(flatten(r, cond, label, seed, k,
                                    time.perf_counter() - t0))
            print(f"[{cond}] weights {i + 1}/{len(configs)} {label:<8} "
                  f"pdr={statistics.mean(_series(rows, cond, label, 'pdr')):.3f} "
                  f"({time.perf_counter() - t_start:.0f}s elapsed)")

    name = f"phase7_weights{tag}"
    _write_rows(rows, f"{name}_trials.csv")

    per_config = {}
    for label, cfg in configs:
        per_config[label] = {
            "weights": cfg["weights"],
            "chain_loss10": {"pdr": _mc(rows, "chain_loss10", "pdr", label),
                             "delivery_completion_s":
                             _mc(rows, "chain_loss10",
                                 "delivery_completion_s", label)},
            "mobility_loss10": {"pdr": _mc(rows, "mobility_loss10", "pdr", label),
                                "delivery_completion_s":
                                _mc(rows, "mobility_loss10",
                                    "delivery_completion_s", label)},
        }
    summary = {"experiment": "phase7_weight_sensitivity",
               "trials": trials, "seed0": seed0, "loss": LOSS,
               "anchor": "w_d=1.0; other weights at defaults "
                         "(w_e=1.0, w_t=0.5, w_q=0.5, w_h=0.3); each swept "
                         "weight replaces its default in that config",
               "sweep": {wn: list(SWEEP_VALUES) for wn in WEIGHT_NAMES},
               "per_config": per_config}
    with open(OUT / f"{name}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    pareto_graph(per_config, name)
    print(f"wrote {name}_trials.csv / _summary.json")


def _mc(rows: list[dict], cond: str, metric: str,
        label: str) -> dict | None:
    vals = _series(rows, cond, label, metric)
    mc = mean_ci95(vals)
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 4),
            "ci95": round(mc[1], 4) if mc else None, "n": len(vals)}


def pareto_graph(per_config: dict, name: str) -> None:
    """Graph 7.2: PDR vs delivery time per weight config; Pareto-efficient
    configs (no other config has both higher PDR and lower time) red."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pts = {label: (d["mobility_loss10"]["pdr"]["mean"],
                   d["mobility_loss10"]["delivery_completion_s"]["mean"])
           for label, d in per_config.items()
           if d["mobility_loss10"]["pdr"]
           and d["mobility_loss10"]["delivery_completion_s"]}
    dominated = set()
    for a, (pa, ta) in pts.items():
        for b, (pb, tb) in pts.items():
            if a != b and pb >= pa and tb < ta - 1e-12:
                dominated.add(a)
                break
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    offsets = {
        "w_h=2.0": (-40, 2), "w_h=4.0": (6, 2),
        "w_e=0.0": (6, -8),
        "w_h=1.0": (-38, 4), "w_t=0.0": (6, 4),
        "w_e=0.5": (6, -4),
        "w_q=4.0": (-38, 4), "w_h=0.0": (6, -8),
        "w_t=0.5": (6, 8), "w_e=1.0": (-38, -2), "w_q=0.5": (6, -12),
        "w_t=2.0": (6, 6), "w_q=0.0": (-38, -6), "w_t=1.0": (6, -10),
        "w_h=0.5": (6, 4), "w_e=0.25": (6, 4), "w_t=0.25": (6, -8),
    }
    for label, (p, t) in pts.items():
        on = label not in dominated
        ax.scatter(t, p, s=90 if on else 40,
                   color="#c0392b" if on else "#7f8c8d",
                   zorder=3 if on else 2)
        off = offsets.get(label, (3, 3))
        ax.annotate(label, (t, p), fontsize=6, xytext=off,
                    textcoords="offset points")
    ax.set_xlabel("mean delivery completion (s)  [lower is better]"
                  "\n(mobility + 10% loss condition)")
    ax.set_ylabel("mean PDR")
    ax.set_title("Graph 7.2: weight-sensitivity Pareto (plan section 9.3)\n"
                 "red = Pareto-efficient configs")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}_g72_pareto.png", dpi=600)
    plt.close(fig)
    print(f"chart: {OUT / f'{name}_g72_pareto.png'}")
    print("Pareto-efficient configs:", sorted(set(pts) - dominated))


def _write_rows(rows: list[dict], fname: str) -> None:
    with open(OUT / fname, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ABL_KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Phase 7 ablations + weight sensitivity (plan section 9)")
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--only", choices=("ablation", "weights"), default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    if args.only in (None, "ablation"):
        ablation_campaign(args.trials, args.seed0, args.tag)
    if args.only in (None, "weights"):
        weight_campaign(args.trials, args.seed0, args.tag)
    print(f"\n{args.trials} trials/variant in {time.perf_counter() - t0:.0f}s "
          f"-> data/processed/phase7*")


if __name__ == "__main__":
    main()
