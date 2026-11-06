#!/usr/bin/env python3
"""DCF contention campaign — does the protocol ranking survive shared airtime?

Runs the diamond and shadow-fading conditions with the opt-in 802.11 DCF
model (network_model.dcf) at two offered-load levels:

  bg = 0    : only the protocols' own traffic (beacons + data)
  bg = 600  : 600 aggregate 1 kB frames/s of synthetic same-channel load
              (~55% airtime at 11 Mbps), representing video/telemetry

Conditions (30 seeds, 90 s, all eight protocol models):
  diamond_dcf_bg{0,600}                 loss 10%, sigma 0
  shadow_diamond_s{0,2,4}_bg{0,600}     loss 10%
  shadow_patrol_s{0,2,4}_bg{0,600}      patrol lanes, 10 UAVs, 20 m/s

Outputs:
  data/processed/dcf_campaign.csv
  data/processed/dcf_campaign_summary.json

The summary reports, per condition: mean PDR and 95% CI per protocol, the
MP-CARE margin over GPSR, the full ranking, mean channel utilisation, and a
side-by-side comparison with the stored legacy (no-DCF) campaign means.

Usage:
  py -3.13 scripts/run_dcf.py --trials 3 --tag _smoke
  py -3.13 scripts/run_dcf.py
  py -3.13 scripts/run_dcf.py --replication   # retry-only 2 dB replication
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
from stats_utils import (mannwhitneyu, holm_bonferroni, rank_biserial,  # noqa: E402
                         mean_ci95)

OUT = ROOT / "data" / "processed"
SEED0 = 1000
ALGORITHMS = ["astar_dist", "gpsr", "aodv", "olsr", "letolsr",
              "astar_etx", "qlgr", "mpcare"]
BG_LEVELS = (0.0, 600.0)
SIGMAS = (0.0, 2.0, 4.0)
LOSS = 0.10
DURATION = 90.0

ROW_KEYS = ["seed", "algorithm", "condition", "scenario", "sigma_db", "bg_fps",
            "loss", "chain_pdr", "avg_latency_ms", "reroutes",
            "airtime_s", "utilization", "collisions", "frames_scheduled",
            "bc_frames", "bc_rx_attempts", "bc_rx_blocked", "bc_rx_ok",
            "bc_collided_frames", "deferred", "bg_frames"]


def _tables() -> dict:
    dia = json.loads((OUT / "phase8_qlgr_diamond.json").read_text(encoding="utf-8"))
    dyn = json.loads((OUT / "phase8_qlgr_dynamic.json").read_text(encoding="utf-8"))
    return {"diamond": dia["table"], "dynamic": dyn["table"]}


def _conditions(cs_range: float = 600.0) -> list[dict]:
    conds: list[dict] = []
    for bg in BG_LEVELS:
        conds.append(dict(label=f"diamond_dcf_bg{int(bg)}", scenario="diamond",
                          injected_loss=LOSS, shadow_sigma_db=0.0,
                          dcf_bg_frames_per_s=bg, dcf_cs_range_m=cs_range,
                          sigma_db=0.0, bg_fps=bg))
    for bg in BG_LEVELS:
        for s in SIGMAS:
            conds.append(dict(label=f"shadow_diamond_s{int(s)}_bg{int(bg)}",
                              scenario="diamond", injected_loss=LOSS,
                              shadow_sigma_db=s, dcf_bg_frames_per_s=bg,
                              dcf_cs_range_m=cs_range, sigma_db=s, bg_fps=bg))
    for bg in BG_LEVELS:
        for s in SIGMAS:
            conds.append(dict(label=f"shadow_patrol_s{int(s)}_bg{int(bg)}",
                              scenario="patrol", speed=20.0, n_uav=10,
                              shadow_sigma_db=s, dcf_bg_frames_per_s=bg,
                              dcf_cs_range_m=cs_range, sigma_db=s, bg_fps=bg))
    return conds


def _run(conds: list[dict], trials: int, seed0: int, tag: str) -> list[dict]:
    tables = _tables()
    rows: list[dict] = []
    t0 = time.perf_counter()
    for cond in conds:
        label = cond["label"]
        kwargs = {k: v for k, v in cond.items()
                  if k not in ("label", "sigma_db", "bg_fps")}
        table = tables["dynamic" if kwargs.get("scenario") == "patrol"
                       else "diamond"]
        for algo in ALGORITHMS:
            for trial in range(trials):
                seed = seed0 + trial
                r = run_sim.run(
                    seed, ROOT / "data" / "raw", duration=DURATION,
                    write_csv=False, algorithm=algo, dcf=True,
                    qlgr_opts={"learn": False, "seed": seed, "eps": 0.0,
                               "table": table},
                    **kwargs)
                st = r.get("dcf") or {}
                rows.append({
                    "seed": seed, "algorithm": algo, "condition": label,
                    "scenario": cond["scenario"], "sigma_db": cond["sigma_db"],
                    "bg_fps": cond["bg_fps"], "loss": LOSS,
                    "chain_pdr": r["chain_pdr"],
                    "avg_latency_ms": r.get("avg_latency_ms"),
                    "reroutes": sum((r.get("trigger_events") or {}).values()),
                    "airtime_s": st.get("airtime_s"),
                    "utilization": (round(st.get("airtime_s", 0.0) / DURATION, 4)
                                    if st else None),
                    "collisions": st.get("collisions"),
                    "frames_scheduled": st.get("frames_scheduled"),
                    "bc_frames": st.get("broadcast_frames"),
                    "bc_rx_attempts": st.get("broadcast_rx_attempts"),
                    "bc_rx_blocked": st.get("broadcast_rx_blocked"),
                    "bc_rx_ok": st.get("broadcast_rx_ok"),
                    "bc_collided_frames": st.get("broadcast_collided_frames"),
                    "deferred": st.get("deferred"),
                    "bg_frames": st.get("bg_frames"),
                })
            print(f"[dcf] {label:<28} {algo:<11} {trials} trials "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return rows


def _legacy_means() -> dict:
    """Stored no-DCF means for the same conditions (phase 9)."""
    out: dict = {}
    for name, prefix in (("shadow_diamond", "s"), ("shadow_patrol", "s")):
        p = OUT / f"phase9_{name}.csv"
        if not p.exists():
            continue
        for r in csv.DictReader(p.open(encoding="utf-8")):
            cond = r["condition"]
            if name == "shadow_diamond":
                if cond not in ("s0_loss_0.10", "s2_loss_0.10", "s4_loss_0.10"):
                    continue
                sigma = int(cond[1])
                key = f"shadow_diamond_s{sigma}_bg0"
            else:
                if not cond.endswith("_patrol"):
                    continue
                sigma = int(cond[1])
                key = f"shadow_patrol_s{sigma}_bg0"
            out.setdefault(key, {}).setdefault(r["algorithm"], []).append(
                float(r["chain_pdr"]))
    return {k: {a: round(statistics.mean(v), 4) for a, v in d.items()}
            for k, d in out.items()}


def _replication(trials: int = 30) -> dict:
    """Retry-only 2 dB diamond replication used in the manuscript (seeds 7100+).

    The original 2 dB sample produced a borderline significant MP-CARE lead
    over the ETX oracle. This disjoint sample re-tests the same contrast with
    the legacy MAC, so the DCF sample is not the only evidence against it.
    """
    table = _tables()["diamond"]
    rows: list[dict] = []
    seed0 = 7100
    for algo in ALGORITHMS:
        for i in range(trials):
            seed = seed0 + i
            r = run_sim.run(
                seed, ROOT / "data" / "raw", duration=DURATION,
                write_csv=False, algorithm=algo, dcf=False,
                scenario="diamond", injected_loss=LOSS, shadow_sigma_db=2.0,
                qlgr_opts={"learn": False, "seed": seed, "eps": 0.0,
                           "table": table})
            rows.append({"seed": seed, "algorithm": algo,
                         "condition": "diamond_s2_retry",
                         "scenario": "diamond", "sigma_db": 2.0,
                         "bg_fps": 0.0, "loss": LOSS,
                         "chain_pdr": r["chain_pdr"],
                         "avg_latency_ms": r.get("avg_latency_ms"),
                         "reroutes": sum((r.get("trigger_events") or {}).values())})
    per: dict = {}
    for algo in ALGORITHMS:
        vals = [x["chain_pdr"] for x in rows if x["algorithm"] == algo]
        ci = mean_ci95(vals)
        per[algo] = {"n": len(vals), "mean": round(statistics.mean(vals), 4),
                     "ci95": round(ci[1], 4) if ci else None}
    mw = mannwhitneyu([x["chain_pdr"] for x in rows if x["algorithm"] == "mpcare"],
                      [x["chain_pdr"] for x in rows if x["algorithm"] == "astar_etx"])
    summary = {
        "experiment": "dcf_legacy_replication",
        "n_trials": trials, "seed0": seed0, "duration_s": DURATION,
        "algorithms": ALGORITHMS,
        "conditions": {"diamond_s2_retry": {
            "per_algorithm": per,
            "mpcare_minus_oracle_pp": round(
                100 * (per["mpcare"]["mean"] - per["astar_etx"]["mean"]), 2),
        }},
        "oracle_test": {"u": mw["U"], "p_two_sided": mw["p"]},
    }
    with open(OUT / "dcf_legacy_replication_s2.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "dcf_legacy_replication_s2.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("replication:", summary["conditions"]["diamond_s2_retry"],
          summary["oracle_test"], flush=True)
    return summary


def _summarise(rows: list[dict], trials: int, seed0: int, tag: str,
               cs_range: float = 600.0) -> dict:
    labels = sorted({r["condition"] for r in rows})
    summary: dict = {
        "experiment": "dcf_campaign", "n_trials": trials, "seed0": seed0,
        "duration_s": DURATION, "algorithms": ALGORITHMS,
        "dcf": {"slot_s": 9e-6, "difs_s": 28e-6, "cw_min": 32, "cw_max": 1024,
                "retries": 4, "rate_mbps": 11.0, "cs_range_m": cs_range,
                "bg_frame_bytes": 1000},
        "conditions": {},
    }
    legacy = _legacy_means()
    for lab in labels:
        sub = [r for r in rows if r["condition"] == lab]
        per: dict = {}
        for algo in ALGORITHMS:
            vals = [r["chain_pdr"] for r in sub if r["algorithm"] == algo]
            if not vals:
                continue
            ci = mean_ci95(vals)
            util = [r["utilization"] for r in sub
                    if r["algorithm"] == algo and r["utilization"] is not None]
            per[algo] = {
                "n": len(vals), "mean": round(statistics.mean(vals), 4),
                "ci95": round(ci[1], 4) if ci else None,
                "utilization": round(statistics.mean(util), 4) if util else None,
                "collisions": int(statistics.mean(
                    [r["collisions"] or 0 for r in sub
                     if r["algorithm"] == algo])),
            }
            bcf = [r.get("bc_frames") or 0 for r in sub
                   if r["algorithm"] == algo]
            bca = [r.get("bc_rx_attempts") or 0 for r in sub
                   if r["algorithm"] == algo]
            bcb = [r.get("bc_rx_blocked") or 0 for r in sub
                   if r["algorithm"] == algo]
            if bca:
                per[algo]["bc_frames"] = round(statistics.mean(bcf), 1)
                per[algo]["bc_rx_blocked"] = round(statistics.mean(bcb), 1)
                per[algo]["bc_block_rate"] = round(
                    sum(bcb) / sum(bca), 4) if sum(bca) else 0.0
        # MP-CARE vs each baseline, Holm within the condition
        pvals, pairs = [], []
        for algo in ALGORITHMS:
            if algo == "mpcare" or algo not in per:
                continue
            mp = [r["chain_pdr"] for r in sub if r["algorithm"] == "mpcare"]
            bl = [r["chain_pdr"] for r in sub if r["algorithm"] == algo]
            mw = mannwhitneyu(mp, bl)
            per[algo]["mw_p_vs_mpcare"] = round(mw["p"], 6) if mw["p"] is not None else None
            per[algo]["rank_biserial_vs_mpcare"] = rank_biserial(mp, bl)
            pvals.append(mw["p"])
            pairs.append(algo)
        adj = holm_bonferroni([p if p is not None else 1.0 for p in pvals])
        for algo, p in zip(pairs, adj):
            per[algo]["holm_p_vs_mpcare"] = round(p, 6) if p is not None else None
        ranking = sorted(per, key=lambda a: per[a]["mean"], reverse=True)
        summary["conditions"][lab] = {
            "per_algorithm": per,
            "ranking": ranking,
            "mpcare_minus_gpsr_pp": round(
                100 * (per["mpcare"]["mean"] - per["gpsr"]["mean"]), 2)
            if "mpcare" in per and "gpsr" in per else None,
            "legacy_no_dcf": legacy.get(lab),
        }
    with open(OUT / f"dcf_campaign{tag}.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW_KEYS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    with open(OUT / f"dcf_campaign{tag}_summary.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=SEED0)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--cs-range", type=float, default=600.0,
                    help="physical carrier-sense range in metres (decode=300)")
    ap.add_argument("--replication", action="store_true",
                    help="run the retry-only 2 dB oracle replication (seeds 7100+)")
    args = ap.parse_args()
    if args.replication:
        _replication(args.trials)
        return
    conds = _conditions(args.cs_range)
    if args.only:
        keep = {x.strip() for x in args.only.split(",") if x.strip()}
        conds = [c for c in conds if c["label"] in keep]
    t0 = time.perf_counter()
    rows = _run(conds, args.trials, args.seed0, args.tag)
    summary = _summarise(rows, args.trials, args.seed0, args.tag,
                         args.cs_range)
    print(f"dcf campaign done in {time.perf_counter() - t0:.0f}s", flush=True)
    for lab, s in summary["conditions"].items():
        print(f"{lab:<28} margin_vs_gpsr={s['mpcare_minus_gpsr_pp']:+.1f}pp "
              f"rank={s['ranking'][:3]}", flush=True)


if __name__ == "__main__":
    main()
