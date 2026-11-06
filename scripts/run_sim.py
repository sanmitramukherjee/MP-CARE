#!/usr/bin/env python3
"""MP-CARE simulation driver — Phase 1 benchmark + Phase 2 baselines.

Algorithms (implementation_plan.md sections 3.5, 4.2 and 5):
  astar_dist : B1  distance-only A* (navigation lower bound)
  gpsr       : B2  GPSR greedy + perimeter forwarding (FANET standard)
  aodv       : B3  AODV (RFC 3561 subset, reactive routing)
  olsr       : B4  OLSR (RFC 3626 subset, proactive routing, MPRs)
  astar_etx  : B5  AT-PP style: A* with ETX edge cost (throughput-aware cousin)
  qlgr       : B6  modern learned baseline (improvement.md Task 1): tabular
               Q-learning geographic routing (QLGR family, adaptation of
               Qiu et al. 2021 KSII TIIS 15(11)) — per-node state->
               preference table over (progress, ETX, receding-speed) bins,
               epsilon-greedy training runs, inference-only evaluation
  mpcare     : MP-CARE (Phase 3: predictive composite cost + event-triggered
               re-routing with hysteresis, decentralized beacons)

Scenarios:
  chain   : the Phase 1 strict 3-hop+ line (160 m spacing, 50 packets)
  diamond : two near-equal-length routes (upper 2-hop with marginal links vs
            rim 3-hop with solid links, total distance within ~15%) — the
            section 5.4 head-to-head topology that can actually discriminate
            routing decisions (a bare line offers no alternatives at all)
  mobility: random-waypoint flight in a 900 x 600 m box at `speed` m/s
            (Phase 4 experiment 5: link churn, LET accuracy, PDR vs velocity);
            packets are generated for the whole run (no hover benchmark window)
  scale    : Phase 6 (plan section 8.2): N UAVs on a static random geometric
            graph at ~constant density (~57.8 km^2/UAV -> bounded 1-hop degree,
            which is what makes the O(deg) claims testable), 50 benchmark
            packets UAV0 -> UAV(N-1) during the static window t=[5,30) s
            (chain semantics), then a pre-drawn random-waypoint mission of
            SCALE_POINTS[N] legs at 8 m/s in the per-N box

Knobs (Phase 4): `n_uav` (swarm size — 10 for the node-failure experiment),
`speed` (mobility m/s), `injected_latency_ms` (per-hop one-way latency).
All algorithms run the SAME scenario, the SAME physical network model, and
write the SAME section 3.4 logging schema — fair comparison by construction.

Extra knobs:
  --loss X      : global injected per-attempt packet loss (0..1)
  --badlink     : inject 0.85 per-attempt loss on the pair {0,1} during
                  t in [15, 25] s (chain scenario) — drives the T1 trigger

All algorithms run the SAME scenario, the SAME physical network model, and
write the SAME section 3.4 logging schema — fair comparison by construction.

Scenario:
  t=0-30 s  : 5 UAVs hover on a 160 m-spaced line (strict 3-hop+ chain:
              two-hop distance 320 m > 300 m range). UAV0 -> UAV4 every 0.5 s
              -> 50 benchmark packets.
  t=30-50 s : swarm compresses to 140 m spacing, flies to new waypoints.
  t=50-90 s : hover at the new positions.

Checks (any algorithm):
  1. spawn/stability   3. chain PDR (>90% for GPSR per §4.4)
  2. A* waypoints      4. ETX moves / inf out of range
  5. CSV schema >= 10 Hz            6. same seed -> identical run

Usage:
  py -3.13 scripts/run_sim.py --algorithm gpsr
  py -3.13 scripts/run_sim.py --algorithm aodv
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
import time
from collections import deque
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "network_model"))
sys.path.insert(0, str(ROOT / "src" / "mpcare_agent"))
sys.path.insert(0, str(ROOT / "src" / "logger"))
sys.path.insert(0, str(ROOT / "src" / "aodv_routing"))
sys.path.insert(0, str(ROOT / "src" / "olsr_routing"))
sys.path.insert(0, str(ROOT / "src" / "qlgr_routing"))

from network_model.model import NetworkParams, LinkModel, NetworkSnapshot  # noqa: E402
from mpcare_agent.agent import UavAgent, WaypointMission, Packet, astar_path  # noqa: E402
from mpcare_agent.messaging import MessageBus  # noqa: E402
from mpcare_agent.router import MpcareRouter, SimHooks  # noqa: E402
from aodv_routing.core import AodvAgent  # noqa: E402
from olsr_routing.core import OlsrAgent  # noqa: E402
from olsr_routing.let_olsr import LetOlsrAgent  # noqa: E402
from qlgr_routing.core import QlgrAgent  # noqa: E402
from logger_pkg.core import TrialLogger, verify_schema  # noqa: E402

N_UAV = 5
SPACING_START = 160.0     # strict chain: 2-hop distance 320 m > 300 m range
SPACING_GOAL = 140.0
ALT = 20.0
HOVER_UNTIL = 30.0        # strict-chain phase = the PDR benchmark window
FLY_UNTIL = 50.0
SEND_FROM = 5.0
SEND_EVERY = 0.5
SEND_BUFFER_S = 5.0       # dynamic protocols: packets queue max 5 s (real buffer)

ALGORITHMS = ("astar_dist", "gpsr", "aodv", "olsr", "letolsr", "astar_etx",
              "qlgr", "mpcare")

MOBILITY_BOX = (700.0, 500.0)      # random-waypoint arena (experiment 5): sized
                                   # so 5 UAVs stay mostly connected with episodic
                                   # partitions — the regime where routing matters
MOBILITY_SPEEDS = (5.0, 10.0, 15.0, 20.0, 25.0)

# Continuous-traffic scenarios (improvement.md Task 3b): packets for the whole
# run, aircraft stay airborne (no hover/waypoint-completion benchmark).
CONTINUOUS_SCENARIOS = ("mobility", "patrol")

PATROL_N = 10                     # 10-UAV patrol lanes (Task 3b): 5 east-west
PATROL_SPEED = 20.0               # shuttles x 5 north-south shuttles in a
PATROL_BOX = (1000.0, 1000.0)     # 1000 x 1000 m box, crossing paths
def _patrol_wp(uav: int, leg: int) -> tuple[float, float, float]:
    """Deterministic patrol loop: UAVs 0-4 shuttle east-west along lanes
    y = 150 + 140*i (x alternates 120 <-> 880); UAVs 5-9 shuttle north-south
    along lanes x = 150 + 140*(i-5) (y alternates 120 <-> 880). Crossing
    paths churn the topology periodically; no hover, continuous traffic."""
    lo, hi = 120.0, 880.0
    if uav < 5:
        y = 150.0 + 140.0 * uav
        x = lo if leg % 2 == 0 else hi
        return (x, y, 40.0)
    x = 150.0 + 140.0 * (uav - 5)
    y = lo if leg % 2 == 0 else hi
    return (x, y, 40.0)

CONVERGENCE_DEFS = {   # phase 6 metric definitions (plan section 8.3 metric 3)
    # All measured INSIDE THE STATIC WINDOW t in [0, 30) s: afterwards the
    # mission churns the topology and "convergence" is ill-defined. Measured
    # as SET COVERAGE (never table size): first tick the source's protocol
    # state covers its end-of-window neighborhood / component.
    "mpcare": "src 1-hop table covers its static-window neighborhood "
              "(set coverage in t<30 s; 1-hop state = full state)",
    "olsr": "src topology table covers its static-window connected component "
            "(set coverage in t<30 s)",
    "aodv": "first route discovery to the far destination within the static "
            "window (None if discovery completes only after t=30 s)",
    "gpsr": "stateless: n/a",
    "qlgr": "stateless: n/a (learned tabular Q-forwarding, no topology state)",
}

# Phase 6 scale points (plan section 8.2): N -> (box_w, box_h, mission legs).
# Box area grows linearly in N (density held ~constant): the plan's own logic —
# 500x500 for 5 UAVs is 50 km^2/UAV; 1700x1700 for 50 is 57.8 km^2/UAV.
SCALE_POINTS: dict[int, tuple[float, float, int]] = {
    5: (500.0, 500.0, 10),
    10: (700.0, 700.0, 20),
    20: (1000.0, 1000.0, 40),
    30: (1200.0, 1200.0, 60),
    40: (1500.0, 1500.0, 80),
    50: (1700.0, 1700.0, 100),
}
MPCARE_WALK_CAP = 30  # max consecutive progress-less void-walk hops per
                      # packet (phase 6 remediation): the face walk may
                      # REVISIT nodes (circumnavigating the void is the
                      # point), so loop safety is this cap, not a visited
                      # set. 30 hops ~ 3 s of forwarding at 10 hops/s, well
                      # inside the 5 s send buffer; exceeding it drops the
                      # packet honestly (void_walk_drops), never a fake loop.
SCALE_SPEED = 8.0     # m/s: the Phase 1 mission speed
SCALE_INSET = 40.0    # waypoint margin = the spawn margin (40 m): the walk box
                      # EQUALS the spawn box, which is what makes the leg bound
                      # provable (clamping never lengthens a leg when the walk
                      # START is inside the clamp box)

# Diamond scenario (section 5.4 head-to-head). Distances under the section 3.3
# radio model: upper links 0-1 (285 m, PRR~0.62, ETX~2.6) and 1-4 (275 m,
# PRR~0.70, ETX~2.1) vs rim links 0-2 (227 m, ETX~1.1), 2-3 (195 m, ETX~1.0),
# 3-4 (215 m, ETX~1.1). Upper: 2 hops / 560 m / ETX-sum ~4.7; rim: 3 hops /
# 637 m / ETX-sum ~3.3. Cross links 1-2, 1-3 are strong (~1.0). All other
# pairs are out of range. distance-A* takes the upper route; quality-aware
# routers take the rim.
DIAMOND_POS = {0: (0.0, 0.0, 40.0), 1: (295.0, 0.0, 40.0), 4: (585.0, 0.0, 40.0),
               2: (195.0, -125.0, 40.0), 3: (395.0, -125.0, 40.0)}


def build_swarm(seed: int, scenario: str = "chain", speed: float | None = None,
                n_uav: int = N_UAV, dcf: bool = False,
                dcf_bg_frames_per_s: float = 0.0,
                dcf_cs_range_m: float = 600.0):
    params = NetworkParams(rng_seed=seed, dcf_enabled=dcf,
                          dcf_bg_frames_per_s=dcf_bg_frames_per_s,
                          dcf_cs_range_m=dcf_cs_range_m)
    link = LinkModel(params)
    if scenario == "patrol":
        # Task 3b (improvement.md): non-diamond dynamic geometry — 10 UAVs on
        # crossing patrol lanes at PATROL_SPEED, continuous traffic, no hover.
        if n_uav != PATROL_N:
            raise ValueError(f"patrol scenario is defined for {PATROL_N} UAVs only")
        v = PATROL_SPEED if speed is None else speed
        start = [_patrol_wp(i, 0) for i in range(PATROL_N)]
        goal = [_patrol_wp(i, 1) for i in range(PATROL_N)]
        agents = [UavAgent(i, start[i], WaypointMission([goal[i]]),
                           speed_mps=v, seed=seed + i) for i in range(PATROL_N)]
        return params, link, agents, start, goal
    if scenario == "diamond":
        if n_uav != N_UAV:
            raise ValueError("diamond scenario is defined for 5 UAVs only")
        start = [DIAMOND_POS[i] for i in range(N_UAV)]
        # compress gently toward the center (topology change for trigger tests)
        goal = [(x * 0.55 + 100.0, y * 0.5, z) for (x, y, z) in start]
        agents = [UavAgent(i, start[i], WaypointMission([goal[i]]),
                           speed_mps=8.0, seed=seed + i) for i in range(N_UAV)]
        return params, link, agents, start, goal
    if scenario == "mobility":
        # Phase 4 experiment 5: random-waypoint in a box, continuous flight at
        # `speed` m/s (deterministic per seed); legs are refilled in the loop.
        v = MOBILITY_SPEEDS[2] if speed is None else speed
        box_w, box_h = MOBILITY_BOX
        m0 = random.Random(seed)
        start = [(m0.uniform(50.0, box_w - 50.0), m0.uniform(50.0, box_h - 50.0), 40.0)
                 for _ in range(n_uav)]
        goals = [(m0.uniform(50.0, box_w - 50.0), m0.uniform(50.0, box_h - 50.0), 40.0)
                 for _ in range(n_uav)]
        agents = [UavAgent(i, start[i], WaypointMission([goals[i]]),
                           speed_mps=v, seed=seed + 100 + i) for i in range(n_uav)]
        return params, link, agents, start, goals
    if scenario == "scale":
        if n_uav not in SCALE_POINTS:
            raise ValueError(f"scale scenario supports N in {sorted(SCALE_POINTS)}")
        box_w, box_h, n_wps = SCALE_POINTS[n_uav]
        m0 = random.Random(seed)
        start = [(m0.uniform(40.0, box_w - 40.0), m0.uniform(40.0, box_h - 40.0), 40.0)
                 for _ in range(n_uav)]
        wps = [_scale_waypoints(seed, i, n_wps, (box_w, box_h), start[i][:2])
               for i in range(n_uav)]
        agents = [UavAgent(i, start[i], WaypointMission(list(wps[i])),
                           speed_mps=SCALE_SPEED, seed=seed + 100 + i)
                  for i in range(n_uav)]
        # goal = the FIRST mission leg per UAV (the post-hover release target);
        # later legs are already inside the mission and fly in order.
        goal = [wps[i][0] for i in range(n_uav)]
        return params, link, agents, start, goal
    if scenario == "grid":
        # Phase 4 experiment 6: 10 UAVs in two rows (redundant paths); killing
        # mid-chain nodes forces re-route onto the parallel row/diagonals.
        if n_uav != 10:
            raise ValueError("grid scenario is defined for exactly 10 UAVs")
        sp = GRID_SPACING
        start = ([(i * sp, 0.0, 40.0) for i in range(5)]
                 + [(i * sp, sp, 40.0) for i in range(5)])
        goal = [(x + 40.0, y, z) for (x, y, z) in start]   # gentle drift at t=30
        agents = [UavAgent(i, start[i], WaypointMission([goal[i]]),
                           speed_mps=8.0, seed=seed + i) for i in range(n_uav)]
        return params, link, agents, start, goal
    start = [(i * SPACING_START, 0.0, ALT) for i in range(n_uav)]
    goal = [(i * SPACING_GOAL + 150.0, 60.0, ALT) for i in range(n_uav)]
    agents = [UavAgent(i, start[i], WaypointMission([goal[i]]),
                       speed_mps=8.0, seed=seed + i) for i in range(n_uav)]
    return params, link, agents, start, goal


GRID_SPACING = 140.0     # double-row grid (experiment 6): rows at y=0 / y=140,
                         # columns at x = i*140 — cross + diagonal links give
                         # 2-disjoint src-dst paths so node kills are recoverable


def _random_wp(seed: int, uav: int, leg: int) -> tuple[float, float, float]:
    """Deterministic random-waypoint target (leg = number of finished legs)."""
    m = random.Random((seed * 1000 + uav) * 97 + leg)
    w, h = MOBILITY_BOX
    return (m.uniform(50.0, w - 50.0), m.uniform(50.0, h - 50.0), 40.0)


def _scale_waypoints(seed: int, uav: int, n_wps: int,
                     box: tuple[float, float],
                     start_xy: tuple[float, float]) -> list[tuple[float, float, float]]:
    """Phase 6: per-UAV BOUNDED-LEG random-waypoint walk (deterministic per
    (seed, uav)). Every leg is drawn 0.25-0.5 x (box - 2*inset) long, then the
    target is CLAMPED into the walk box (= the spawn box). Per-axis clamping
    never lengthens a leg when the walk START is inside the clamp box (each
    |axis displacement| weakly decreases), so every leg is <= 0.5 x span: the
    longest possible leg at the LARGEST scale point (810 m at S6) finishes in
    ~101 s after the t=30 release — inside the 150 s run, i.e. NO duration
    censoring of first-leg arrivals (waypoint_ok / mission time well-defined).
    (Draft-note: a MIRROR-back reflection was tried first and is WRONG here —
    reflecting a far-outside target across the boundary can LENGTHEN the leg
    from a deep-inside start; caught by test_phase6.py before any campaign.)"""
    m = random.Random((seed * 7919 + uav * 97) ^ 0x5CA1E)
    w, h = box
    lo = SCALE_INSET
    hi_x, hi_y = w - SCALE_INSET, h - SCALE_INSET
    span = min(w, h) - 2.0 * SCALE_INSET
    pts: list[tuple[float, float, float]] = []
    cx, cy = float(start_xy[0]), float(start_xy[1])
    for _ in range(n_wps):
        leg = m.uniform(0.25 * span, 0.5 * span)
        th = m.uniform(0.0, 2.0 * math.pi)
        tx = min(max(cx + leg * math.cos(th), lo), hi_x)   # clamp (see docstring)
        ty = min(max(cy + leg * math.sin(th), lo), hi_y)
        pts.append((tx, ty, 40.0))
        cx, cy = tx, ty
    return pts


def _graph(positions, snap):
    links = {uid: set(snap.neighbors(uid)) for uid in positions}
    return links


def run(seed: int, out_dir: Path, duration: float = 90.0, dt: float = 0.1,
        write_csv: bool = True, algorithm: str = "gpsr", scenario: str = "chain",
        injected_loss: float | None = None,
        loss_schedule: list | None = None,
        badlink: tuple | None = None,
        kill_node: tuple | list | None = None,
        n_uav: int = N_UAV,
        speed: float | None = None,
        injected_latency_ms: float = 0.0,
        shadow_sigma_db: float | None = None,
        dcf: bool = False,
        dcf_bg_frames_per_s: float = 0.0,
        dcf_cs_range_m: float = 600.0,
        mpcare_opts: dict | None = None,
        qlgr_opts: dict | None = None) -> dict:
    """One deterministic trial of `algorithm`. Returns per-check evidence.

    badlink: (t0, t1, loss, a, b) — inject per-PAIR packet loss on {a, b}
    during [t0, t1] (used by the T1 trigger test; measurements must react to
    impairments the physics snapshot cannot see). The impairment is cleared
    after t1.
    kill_node: (t_kill, node) — the UAV disappears at t_kill (T4 test); or a
    LIST of such tuples for the multi-kill conditions of experiment 6.
    n_uav: swarm size (10 for the Phase 4 node-failure experiment; chain
    spacing unchanged, so a longer chain with the same 160 m hop length).
    speed: mobility-scenario flight speed in m/s (experiment 5).
    injected_latency_ms: per-hop one-way latency added on top of airtime +
    propagation (experiment 3); end-to-end delay accumulates over the path.
    qlgr_opts: {'learn': bool, 'seed': int, 'eps': float, 'table': {"(p,e,s)": v}}
    — training vs inference mode of the QLGR baseline (improvement.md Task 1);
    'table' is a merged state-> preference map loaded before the trial; when
    'learn' is False the agent runs inference-only (eps = 0, no updates).
    """
    assert algorithm in ALGORITHMS, algorithm
    params, link, agents, start, goal = build_swarm(
        seed, scenario, speed, n_uav, dcf=dcf,
        dcf_bg_frames_per_s=dcf_bg_frames_per_s,
        dcf_cs_range_m=dcf_cs_range_m)
    n = n_uav
    if injected_loss is not None:
        params.injected_loss = float(injected_loss)
    # loss_schedule: list of (t_from, t_to, loss) step windows — time-varying
    # global injected loss (loss-shift experiment). Evaluated each tick inside
    # the sim loop below; a window's loss applies on its inclusive interval.
    _loss_sched = sorted(loss_schedule or [], key=lambda w: w[0])

    def _loss_at(t: float) -> float:
        base = params.injected_loss
        for t0, t1, v in _loss_sched:
            if t0 <= t < t1:
                return float(v)
        return base
    if injected_latency_ms:
        params.injected_latency_ms = float(injected_latency_ms)
    if shadow_sigma_db is not None:
        params.shadow_sigma_db = float(shadow_sigma_db)
    trial = TrialLogger(out_dir, trial_id=f"{algorithm}_seed{seed}") if write_csv else None

    bus = MessageBus(link) if algorithm in ("aodv", "olsr", "letolsr") else None
    protos = None
    if algorithm == "aodv":
        protos = [AodvAgent(i) for i in range(n)]
    elif algorithm == "olsr":
        protos = [OlsrAgent(i, n) for i in range(n)]
    elif algorithm == "letolsr":
        protos = [LetOlsrAgent(i, n, max_range_m=params.max_range_m)
                  for i in range(n)]
    routers = None
    if algorithm == "mpcare":
        routers = [MpcareRouter(i, n, max_range_m=params.max_range_m,
                                **(mpcare_opts or {}))
                   for i in range(n)]
    qlgr_agents = None
    if algorithm == "qlgr":
        opts = dict(qlgr_opts or {})
        table = opts.pop("table", None)
        learn = bool(opts.pop("learn", True))
        qlgr_agents = [QlgrAgent(i, n, learn=learn,
                                 seed=int(opts.pop("seed", seed)),
                                 eps=float(opts.pop("eps", 0.15)))
                       for i in range(n)]
        assert not opts, f"unknown qlgr_opts: {opts}"
        for qa in qlgr_agents:
            if table:
                qa.load_table(table.get(str(qa.node_id)) or table)
    assert not (mpcare_opts and algorithm != "mpcare"), "mpcare_opts ignored"

    # data packets in flight: dicts {pkt, cur, route(list), next_idx, created}
    inflight: list[dict] = []
    stats = {"sent": 0, "delivered": 0, "chain_sent": 0, "chain_delivered": 0}
    chain_hops: list[int] = []
    speed_rows: list[float] = []         # mobility: per-UAV distance travelled
    events: dict[str, int] = {}          # trigger fires by name (all algorithms)
    first_fire: dict[str, float] = {}    # first fire time per trigger
    route_choice = {"upper": 0, "rim": 0}   # diamond: which route delivered
    hop_counts: list[int] = []
    latencies: list[float] = []
    next_pid = 0
    deliveries: list[tuple[float, float]] = []   # (delivered_t, created_t)
    win: list[deque] = [deque() for _ in range(n)]
    win_lat: list[deque] = [deque() for _ in range(n)]
    etx_t0 = etx_t1 = None
    conv_t: float | None = None
    arrival_t: list[float | None] = [None] * n
    kill_applied = False
    prev_links: dict | None = None
    let_audit: list[tuple[float, float, float]] = []   # (radial v, LET_pred, actual)
    let_audit_noinf = 0        # breaks while approaching: no finite forecast
    past_pos: dict[int, deque] = {}   # short own-position history (LET audit): a
                                      # real node knows where IT was at beacon time
    pos_history: list[list[tuple]] = []
    t0_pos = [tuple(a.pos) for a in agents]
    # ---- phase 6 instrumentation (plan section 8.3) ----
    src_view_sets: list[frozenset] = []  # per-tick SET of the SOURCE's routing
                                         # knowledge (mpcare 1-hop table / olsr
                                         # topology dests) for the convergence
                                         # metric (set coverage, NOT size)
    static_end_links: dict | None = None  # links at the last static tick
                                          # (t = 29.9 s) = the coverage target
    beacon_tx = [0] * n              # mpcare beacon sends per node (overhead)
    void_drops = 0                   # mpcare packets dropped by the walk cap
    deg_acc = 0                      # degree accumulation -> mean degree over run
    deg_n = 0
    cpu_acc = 0.0                    # phase 6: perf_counter seconds spent in
                                     # PROTOCOL work (all real nodes, all ticks):
                                     # mpcare = router ticks + beacon exchange;
                                     # aodv/olsr = ticks + bus delivery; the
                                     # mobility/physics loop is NOT protocol
                                     # compute and is excluded

    t = 0.0
    for tick in range(int(duration / dt)):
        t = tick * dt
        # time-varying injected loss (loss-shift experiment): re-apply each tick
        if _loss_sched:
            params.injected_loss = _loss_at(t)

        # ---- generate benchmark packets (UAV0 -> UAV4) during the chain phase
        if t >= SEND_FROM and t < (duration if scenario in CONTINUOUS_SCENARIOS
                                   else HOVER_UNTIL) \
                and (tick % int(SEND_EVERY / dt) == 0):
            pkt = Packet(pid=next_pid, src=0, dst=n - 1,
                         dst_pos=(agents[n - 1].pos[0], agents[n - 1].pos[1]),
                         created_t=t, path=[0])
            next_pid += 1
            entry = {"pkt": pkt, "cur": 0, "route": [], "next_idx": 1, "chain": True}
            if algorithm in ("astar_dist", "astar_etx"):
                # route computed ONCE at send time over the then-current graph
                # (baseline semantics: no re-routing; Phase 3 adds that)
                entry["_needs_route"] = True
            elif algorithm in ("aodv", "olsr", "mpcare"):
                pass   # route found dynamically; packet queues at the src node
            inflight.append(entry)
            stats["sent"] += 1
            stats["chain_sent"] += 1

        # ---- mobility
        old_pos = [a.pos for a in agents]
        if scenario in CONTINUOUS_SCENARIOS:
            for a in agents:
                wp = a.mission.current_wp()
                if wp and a.move_toward(wp, dt):
                    # random-waypoint: keep flying new legs for the whole run
                    # (mobility); patrol: replay the same loop forever
                    a.mission.advance()
                    if scenario == "mobility":
                        a.mission.waypoints.append(
                            _random_wp(seed, a.uav_id, len(a.mission.waypoints)))
                    else:
                        a.mission.waypoints.append(
                            _patrol_wp(a.uav_id, len(a.mission.waypoints)))
                while len(speed_rows) <= a.uav_id:
                    speed_rows.append(0.0)
                speed_rows[a.uav_id] += math.dist(a.pos[:2], old_pos[a.uav_id][:2])
        else:
            for a in agents:
                if t >= HOVER_UNTIL:
                    wp = a.mission.current_wp()
                    if wp and a.move_toward(wp, dt):
                        a.mission.advance()
                        if arrival_t[a.uav_id] is None:
                            arrival_t[a.uav_id] = t

        # ---- network snapshot + protocol ticks
        # slow log-normal shadow state advances before the snapshot so every
        # unicast this tick (data, beacons, control traffic) sees one coherent
        # per-link offset; the distance-derived snapshot ETX stays blind to it
        # (heterogeneity the physics delivers but the snapshot cannot predict).
        positions = {a.uav_id: a.pos for a in agents}
        if params.shadow_sigma_db > 0.0:
            _ids = list(positions)
            link.step_shadow(
                [(a, b) for i, a in enumerate(_ids) for b in _ids[i + 1:]
                 if math.dist(positions[a], positions[b]) <= params.max_range_m],
                dt)
        snap = NetworkSnapshot(positions, link, t)
        links = _graph(positions, snap)
        # per-pair impairment window (T1 trigger test); clear outside the window
        if badlink:
            if badlink[0] <= t < badlink[1]:
                link.set_link_loss(badlink[3], badlink[4], badlink[2])
            else:
                link.clear_link_losses()
        # node-kill scenario (T4 / experiment 6): UAVs vanish from the network
        if kill_node:
            kills = kill_node if isinstance(kill_node[0], tuple) else [kill_node]
            removed = False
            kill_applied = False
            for tk, nid in kills:
                if t >= tk and nid in positions:
                    positions.pop(nid)
                    removed = True
            if removed:
                kill_applied = True
                snap = NetworkSnapshot(positions, link, t)
                links = _graph(positions, snap)
        # DCF scheduler: one frame-level channel state per protocol tick
        link.begin_tick(t, positions)
        deg_acc += sum(len(v) for v in links.values())   # phase 6 avg degree
        deg_n += len(links)
        if (static_end_links is None and algorithm != "gpsr"
                and t >= HOVER_UNTIL - dt - 1e-9):
            static_end_links = links   # last static-window tick = coverage target
        # RQ3 post-hoc audit (experiment 5): for every link that just broke in
        # the physics while both endpoints are alive, ask MP-CARE's neighbor
        # table + predictor what it would have forecast, and compare with the
        # actual remaining link life (break_t - last_beacon_t).
        if algorithm == "mpcare" and prev_links is not None:
            for a in positions:
                ra = routers[a]
                for b in prev_links.get(a, ()):
                    if a > b or b not in positions or b in links.get(a, ()):
                        continue
                    info = ra.table.neighbors.get(b)
                    if info is None:
                        continue
                    v = info.radial_vel
                    if v > 0.0:
                        # d as it was at the last beacon: use OUR position at
                        # that moment (a real node retains its own position
                        # history), not the break-time position.
                        pa2 = None
                        for tt, xx, yy in reversed(past_pos.get(a, ())):
                            if tt <= info.last_seen_t + 1e-9:
                                pa2 = (xx, yy)
                                break
                        if pa2 is None:
                            continue
                        d_lb = math.dist(pa2, info.last_pos[:2])
                        act = t - info.last_seen_t
                        if act > 0.0:
                            let_audit.append((v, ra.predictor.let_s(d_lb, v), act))
                    else:
                        let_audit_noinf += 1   # approaching at break: LET = inf
        if algorithm == "aodv":
            for pr in protos:
                pr.set_context(bus, t, positions, links)
        if tick == 0:
            # farthest pair at spawn: the "2 hops away" sanity pair (chain and
            # diamond reduce to the historical (0, 4) choice)
            far = max(((i, j) for i in positions for j in positions if i < j),
                      key=lambda p: math.dist(positions[p[0]][:2], positions[p[1]][:2]))
            etx_t0 = (snap.etx_of(0, 1), snap.etx_of(*far))
        if tick == int(duration / dt) - 1:
            etx_t1 = (snap.etx_of(0, 1), snap.etx_of(*far))

        if algorithm in ("aodv", "olsr", "letolsr"):
            _t00 = time.perf_counter()   # phase 6 CPU instrumentation
            for pr in protos:
                if pr.node_id not in positions:
                    continue    # killed UAV: protocol goes silent
                if algorithm == "letolsr":   # kinematic LET from the position service
                    pr.observe_state(t, positions)
                pr.tick(t, dt, links)
                pr.flush(bus, positions, links)
            handlers = {i: pr for i, pr in enumerate(protos)}
            bus.deliver(handlers, t)
            for pr in protos:                      # replies generated on delivery
                if pr.node_id not in positions:
                    continue
                pr.flush(bus, positions, links)
            bus.deliver(handlers, t)
            cpu_acc += time.perf_counter() - _t00   # phase 6 CPU instrumentation
            # practical convergence evidence: can the source route to the far
            # destination yet? (first time = convergence/discovery time)
            src_pr = protos[0]
            # phase 6 fix: the far destination is n-1 (N_UAV hardcoding was
            # correct only for the default 5-UAV runs)
            has_route = (src_pr.next_hop_for(n - 1, t) is not None
                         if algorithm == "aodv"
                         else src_pr.next_hop_for(n - 1) is not None)
            if conv_t is None and has_route:
                conv_t = t
            if algorithm in ("olsr", "letolsr"):   # phase 6 convergence instrumentation
                src_view_sets.append(frozenset(d for (_lh, d) in protos[0].topology)
                                     if protos[0].node_id in positions
                                     else frozenset())

        if algorithm == "qlgr":
            _t00 = time.perf_counter()   # phase 6 CPU instrumentation
            for qa in qlgr_agents:
                if qa.node_id not in positions:
                    continue          # killed UAV: protocol goes silent
                qa.observe(t, positions, links, snap)
                qa.reroute_event = False
            cpu_acc += time.perf_counter() - _t00

        if algorithm == "mpcare":
            _t00 = time.perf_counter()   # phase 6 CPU instrumentation
            hooks = SimHooks(positions, link, snap)
            for r in routers:
                if r.node_id not in positions:
                    continue          # killed UAV: no beacons, no ticks
                r.bind(hooks)
                r._own_queue_ratio = 0.0
                r.tick(t)
            # beacon exchange (plan section 5.2.4): one hop per tick over the
            # physical channel, then a second round for freshly heard beacons
            for r in routers:
                if r.node_id not in positions:
                    continue
                payload = r.beacon_payload(positions[r.node_id], r._own_queue_ratio, t)
                beacon_tx[r.node_id] += 1   # phase 6 overhead instrumentation
                nbs = [nb for nb in sorted(links.get(r.node_id, ()))
                       if nb in positions]
                if link.dcf is not None and nbs:
                    # one broadcast frame for all neighbours (802.11 broadcast:
                    # no ACK, no retry), received independently per neighbour
                    ds = {nb: math.dist(positions[r.node_id][:2], positions[nb][:2])
                          for nb in nbs}
                    for nb, ok in link.broadcast_success(
                            ds, r.node_id, size_bytes=64).items():
                        if ok:
                            routers[nb].receive_beacon(
                                payload, r.node_id,
                                snap.etx_of(r.node_id, nb), t)
                else:
                    for nb in nbs:
                        d = math.dist(positions[r.node_id][:2], positions[nb][:2])
                        if link.unicast_success(d, r.node_id, nb, size_bytes=64):
                            routers[nb].receive_beacon(payload, r.node_id, snap.etx_of(r.node_id, nb), t)
            for r in routers:
                if r.node_id in positions:
                    r.table.set_my_position(positions[r.node_id])
            cpu_acc += time.perf_counter() - _t00   # phase 6 CPU instrumentation
            src_has = (routers[0].next_hop.get(n - 1) is not None
                       if n - 1 in positions else False)
            if conv_t is None and (src_has or routers[0].table.ids()):
                conv_t = t
            src_view_sets.append(frozenset(routers[0].table.ids())
                                 if 0 in positions else frozenset())

        # ---- advance data packets one hop per tick
        still: list[dict] = []
        for p in inflight:
            # dynamic-protocol packets queue at nodes while discovery/convergence
            # runs; give up after SEND_BUFFER_S (a real network's send buffer).
            if algorithm in ("aodv", "olsr", "mpcare") and t - p["pkt"].created_t > SEND_BUFFER_S:
                continue   # dropped: sat too long in the queue
            if p["cur"] not in positions:
                continue   # holder UAV died in flight -> packet lost with it
            pkt: Packet = p["pkt"]
            if pkt.dst in positions:
                # position service (geocast assumption): every forwarder knows the
                # destination's CURRENT location, so greedy mode stays honest when
                # the destination itself moves (mobility scenario)
                pkt.dst_pos = (positions[pkt.dst][0], positions[pkt.dst][1])
            cur = p["cur"]
            if cur == pkt.dst:
                still.append(p)
                continue

            if algorithm in ("astar_dist", "astar_etx"):
                if p.get("_needs_route"):
                    if algorithm == "astar_dist":
                        route = astar_path(0, pkt.dst, links, positions)
                    else:  # astar_etx: prefer high-throughput multi-hop routes
                        route = astar_path(
                            0, pkt.dst, links, positions,
                            edge_cost=lambda a, b: snap.etx_of(a, b),
                            heuristic=lambda n: math.dist(positions[n][:2], pkt.dst_pos)
                            / params.max_range_m)
                    p["route"] = route or []
                    p["next_idx"] = 1
                    p["_needs_route"] = False
                    if not p["route"]:
                        continue                    # unreachable at send time -> drop
                route = p["route"]
                if p["next_idx"] >= len(route):
                    still.append(p)
                    continue
                nxt = route[p["next_idx"]]
                if nxt not in links.get(cur, ()):     # route broke: baseline drops
                    continue
                d = snap.dist[(cur, nxt)] if (cur, nxt) in snap.dist else math.dist(
                    positions[cur][:2], positions[nxt][:2])
                if link.unicast_success(d, cur, nxt):
                    p["cur"] = nxt
                    p["latency_ms"] = p.get("latency_ms", 0.0) + link.latency_ms(d)
                    if p.get("first_fwd_t") is None:
                        p["first_fwd_t"] = t   # source queue wait = first fwd - creation
                    p["next_idx"] += 1
                    pkt.hops += 1
                    pkt.path.append(nxt)
                    if nxt == pkt.dst:
                        _deliver(p, t, stats, hop_counts, latencies, chain_hops, win,
                                 win_lat, route_choice, deliveries)
                    else:
                        still.append(p)

            elif algorithm == "gpsr":
                nxt = agents[cur].forward(pkt, positions, links)
                if nxt is None:
                    continue
                d = math.dist(positions[cur][:2], positions[nxt][:2])
                if link.unicast_success(d, cur, nxt):
                    p["cur"] = nxt
                    p["latency_ms"] = p.get("latency_ms", 0.0) + link.latency_ms(d)
                    if p.get("first_fwd_t") is None:
                        p["first_fwd_t"] = t   # source queue wait = first fwd - creation
                    if nxt == pkt.dst:
                        _deliver(p, t, stats, hop_counts, latencies, chain_hops, win,
                                 win_lat, route_choice, deliveries)
                    else:
                        still.append(p)

            elif algorithm == "qlgr":
                qa = qlgr_agents[cur]
                nh = qa.forward(pkt, positions, links, snap)
                if nh is None or nh not in links.get(cur, ()):
                    continue           # isolated: drop (GPSR baseline parity)
                d = math.dist(positions[cur][:2], positions[nh][:2])
                ok = link.unicast_success(d, cur, nh)
                delivered = ok and nh == pkt.dst
                qa.observe_attempt(pkt, ok, delivered)
                if not ok:
                    continue           # failed unicast: packet lost here
                p["cur"] = nh
                p["latency_ms"] = p.get("latency_ms", 0.0) + link.latency_ms(d)
                if p.get("first_fwd_t") is None:
                    p["first_fwd_t"] = t
                # walk accounting mirrors the MP-CARE hop cap: a learned
                # router may loop its face (progress-less hops) — drop past
                # a bounded cap so loops cannot stall the packet queue.
                prev_d = math.dist(positions[cur][:2], pkt.dst_pos)
                new_d = math.dist(positions[nh][:2], pkt.dst_pos)
                d0 = p.get("walk_d0")
                if d0 is None:
                    if new_d >= prev_d:
                        p["walk_d0"] = prev_d
                        p["walk_steps"] = 1
                elif new_d < d0:
                    p["walk_d0"] = None
                    p["walk_steps"] = 0
                elif new_d < prev_d:
                    p["walk_steps"] = 0
                else:
                    p["walk_steps"] += 1
                    if p["walk_steps"] > MPCARE_WALK_CAP:
                        void_drops += 1
                        continue
                pkt.hops += 1
                pkt.path.append(nh)
                if nh == pkt.dst:
                    _deliver(p, t, stats, hop_counts, latencies, chain_hops, win,
                             win_lat, route_choice, deliveries)
                else:
                    still.append(p)

            elif algorithm == "mpcare":
                r = routers[cur]
                r.mp_tx_count = getattr(r, "mp_tx_count", 0) + 1   # data-plane tx
                walking = p.get("walk_steps", 0) > 0   # in a void walk?
                nh = r.next_hop_for(pkt.dst, t, visited=frozenset(pkt.path),
                                    prev=pkt.path[-2] if len(pkt.path) >= 2 else None,
                                    walking=walking)
                if nh is None or nh not in links.get(cur, ()):
                    still.append(p)      # isolated (no usable walk edge): queued
                    continue
                d = math.dist(positions[cur][:2], positions[nh][:2])
                ok = link.unicast_success(d, cur, nh)
                r.observe_delivery(nh, ok)   # measured evidence -> T1 / EWMA ETX
                if ok:
                    p["cur"] = nh
                    p["latency_ms"] = p.get("latency_ms", 0.0) + link.latency_ms(d)
                    if p.get("first_fwd_t") is None:
                        p["first_fwd_t"] = t   # source queue wait = first fwd - creation
                    # void-walk accounting (phase 6 remediation), GPSR
                    # resumption semantics: greedy has recovered only when
                    # the packet beats its VOID-ENTRY distance (walk_d0);
                    # anything else counts toward the per-packet walk cap,
                    # which drops the packet HONESTLY when exceeded (the
                    # walk is circling a face with no greedy exit).
                    prev_d = math.dist(positions[cur][:2], pkt.dst_pos)
                    new_d = math.dist(positions[nh][:2], pkt.dst_pos)
                    d0 = p.get("walk_d0")
                    if d0 is None:
                        if new_d >= prev_d:      # greedy stuck: walk begins
                            p["walk_d0"] = prev_d
                            p["walk_steps"] = 1
                    elif new_d < d0:             # beat the entry point: exit
                        p["walk_d0"] = None
                        p["walk_steps"] = 0
                    elif new_d < prev_d:
                        p["walk_steps"] = 0      # productive hop: new face soon
                    else:
                        p["walk_steps"] += 1     # progress-less hop on the face
                        if p["walk_steps"] > MPCARE_WALK_CAP:
                            void_drops += 1
                            continue
                    pkt.hops += 1
                    pkt.path.append(nh)
                    if nh == pkt.dst:
                        _deliver(p, t, stats, hop_counts, latencies, chain_hops, win,
                                 win_lat, route_choice, deliveries)
                    else:
                        still.append(p)

            else:  # aodv / olsr: dynamic next hop from the protocol
                pr = protos[cur]
                nh = (pr.next_hop_for(pkt.dst, t) if algorithm == "aodv"
                      else pr.next_hop_for(pkt.dst))
                if nh is None or nh not in links.get(cur, ()):
                    # no route yet, or a stale route through a node that just
                    # died (RERR/HELLO timeout has not caught up): the
                    # transmission fails and the packet stays queued at the
                    # node until the protocol re-routes (buffer cap applies)
                    still.append(p)
                    continue
                d = math.dist(positions[cur][:2], positions[nh][:2])
                if link.unicast_success(d, cur, nh):
                    p["cur"] = nh
                    p["latency_ms"] = p.get("latency_ms", 0.0) + link.latency_ms(d)
                    if p.get("first_fwd_t") is None:
                        p["first_fwd_t"] = t   # source queue wait = first fwd - creation
                    pkt.hops += 1
                    pkt.path.append(nh)
                    if nh == pkt.dst:
                        _deliver(p, t, stats, hop_counts, latencies, chain_hops, win,
                                 win_lat, route_choice, deliveries)
                    else:
                        still.append(p)
        inflight = still

        # trigger-event evidence (Phase 3) + route-change counts for all
        # algorithms (Phase 4 graph 4.4: event-triggered vs periodic)
        if algorithm == "mpcare":
            for r in routers:
                if r.reroute_event:
                    events[r.reroute_trigger] = events.get(r.reroute_trigger, 0) + 1
                    if r.reroute_trigger not in first_fire:
                        first_fire[r.reroute_trigger] = t
        elif algorithm == "qlgr":
            for qa in qlgr_agents:
                if qa.node_id in positions and qa.reroute_event:
                    events["route-change"] = events.get("route-change", 0) + 1
        elif algorithm in ("aodv", "olsr", "letolsr"):
            for pr in protos:
                if pr.next_hop_change:
                    events["route-change"] = events.get("route-change", 0) + 1
        else:
            for a in agents:
                if a.uav_id in positions and a.reroute_event:
                    events["route-change"] = events.get("route-change", 0) + 1

        # ---- logging rows (section 3.4)
        if trial:
            for a in agents:
                if a.uav_id not in positions:
                    continue    # killed UAV logs nothing
                pr = protos[a.uav_id] if protos else None
                vel = tuple((a.pos[i] - old_pos[a.uav_id][i]) / dt for i in range(3))
                min_sep = min((math.dist(a.pos[:2], agents[o].pos[:2])
                               for o in range(n) if o != a.uav_id), default=0.0)
                nbs = sorted(links.get(a.uav_id, set()))
                while win[a.uav_id] and win[a.uav_id][0][0] < t - 2.0:
                    win[a.uav_id].popleft()
                while win_lat[a.uav_id] and len(win_lat[a.uav_id]) > 50:
                    win_lat[a.uav_id].popleft()
                n_del = sum(1 for _, dflag in win[a.uav_id] if dflag)
                pdr = n_del / max(len(win[a.uav_id]), 1) if win[a.uav_id] else 0.0
                lat_avg = (sum(win_lat[a.uav_id]) / len(win_lat[a.uav_id])
                           if win_lat[a.uav_id] else 0.0)
                tput = n_del * 8.0 / 2.0
                # route telemetry
                if algorithm in ("astar_dist", "astar_etx") and a.uav_id == 0:
                    live = next((p for p in inflight if p["pkt"].src == 0), None)
                    route, hops = (live["route"], max(0, len(live["route"]) - 1)) if live else ([], 0)
                    cost = sum(snap.etx_of(x, y) if algorithm == "astar_etx"
                               else math.dist(positions[x][:2], positions[y][:2])
                               for x, y in zip(route, route[1:]))
                elif pr is not None:
                    route, hops, cost = [], (len(pr.routes) if algorithm == "aodv"
                                             else len(pr.topology) + len(pr.neigh)), 0.0
                else:
                    route, hops, cost = [], 0, 0.0
                wp3 = a.mission.current_wp() or (0.0, 0.0, 0.0)
                trial.log(
                    timestamp_ms=int(t * 1000), uav_id=a.uav_id,
                    position_x=a.pos[0], position_y=a.pos[1], position_z=a.pos[2],
                    velocity_x=vel[0], velocity_y=vel[1], velocity_z=vel[2],
                    target_waypoint_x=wp3[0], target_waypoint_y=wp3[1], target_waypoint_z=wp3[2],
                    current_route_cost=round(float(cost), 2), route_hop_count=hops,
                    neighbors=nbs,
                    neighbor_etx=[snap.etx_of(a.uav_id, nb) for nb in nbs],
                    neighbor_let=[-1.0] * len(nbs),
                    pdr_instantaneous=round(pdr, 3), latency_avg_ms=round(lat_avg, 3),
                    throughput_kbps=round(tput, 3), queue_depth=0,
                    reroute_event=(pr.next_hop_change if pr else a.reroute_event),
                    reroute_trigger=("route-changed" if (pr.next_hop_change if pr
                                                         else a.reroute_event) else ""),
                    min_separation_m=round(min_sep, 2),
                    cpu_time_us=(getattr(a, "cpu_time_us", 0) or 0),
                    algorithm=algorithm, experiment="phase2_baselines",
                )
        pos_history.append([tuple(a.pos) for a in agents])
        prev_links = links
        for a in agents:
            past_pos.setdefault(a.uav_id, deque(maxlen=60)).append(
                (t, a.pos[0], a.pos[1]))

    if trial:
        trial.close()

    # ---------------- checks ----------------
    results = {"seed": seed, "algorithm": algorithm}
    if link.dcf is not None:
        results["dcf"] = link.dcf.summary()
    if scenario in CONTINUOUS_SCENARIOS and speed is not None and speed_rows:
        v_avg = sum(speed_rows) / len(speed_rows) / duration
        results["speed_mps"] = round(v_avg, 2)
        results["speed_ok"] = 0.9 * speed <= v_avg <= 1.1 * speed
    idx10 = int(10.0 / dt)
    results["stability_ok"] = (scenario in CONTINUOUS_SCENARIOS or all(
        math.dist(pos_history[idx10][i][:2], start[i][:2]) < 2.0 for i in range(n)))
    if scenario in CONTINUOUS_SCENARIOS:
        results["waypoint_ok"] = True       # continuous flight: no terminal wp
        results["max_arrival_s"] = None
    else:
        arrivals = [x for x in arrival_t if x is not None]
        # scale missions fly box-scaled legs: the deadline is "every UAV
        # finished its FIRST leg inside the run", not the Phase 1 120 s bar
        deadline = duration if scenario == "scale" else 120.0
        results["waypoint_ok"] = (len(arrivals) == n and max(arrivals) < deadline)
        results["max_arrival_s"] = round(max(arrivals), 1) if arrivals else None
        if scenario == "scale":
            results["mission_time_s"] = (round(results["max_arrival_s"]
                                                - HOVER_UNTIL, 1)
                                         if arrivals else None)
    chain_pdr = stats["chain_delivered"] / stats["chain_sent"] if stats["chain_sent"] else 0.0
    results["chain_pdr"] = round(chain_pdr, 3)
    results["chain_sent"] = stats["chain_sent"]
    results["chain_delivered"] = stats["chain_delivered"]
    results["min_chain_hops"] = min(chain_hops) if chain_hops else None
    results["pdr"] = round(stats["delivered"] / stats["sent"] if stats["sent"] else 0.0, 3)
    results["packets_sent"] = stats["sent"]
    results["packets_delivered"] = stats["delivered"]
    results["last_delivery_s"] = round(max(dt for dt, _ in deliveries), 1) if deliveries else None
    if kill_node:
        # section 6.3 experiment 6 metric: route recovery = time from the first
        # kill to the first end-to-end delivery of a packet CREATED after that
        # kill (i.e. how long until the network carries fresh traffic again).
        kt = kill_node[0][0] if isinstance(kill_node[0], tuple) else kill_node[0]
        post = [dt for dt, ct in deliveries if ct >= kt]
        results["recovery_s"] = round(post[0] - kt, 2) if post else None
    results["avg_latency_ms"] = round(sum(latencies) / len(latencies), 1) if latencies else None
    # §4.4 per-algorithm expectations (NOMINAL conditions only — the absolute
    # bars are sanity checks for the no-stress benchmark; under injected
    # loss/impairments the section 5.4 criterion is the COMPARATIVE margin,
    # computed across algorithms by run_experiment.py, so check 3 is n/a).
    stressed = bool(injected_loss or badlink or kill_node or injected_latency_ms)
    results["stressed"] = stressed
    if algorithm == "gpsr":
        results["routing_ok"] = None if stressed else chain_pdr > 0.85
    elif algorithm == "aodv":
        disc = (protos[0].discovery_times if protos else [])
        results["max_discovery_s"] = round(max(disc), 2) if disc else None
        results["first_route_s"] = conv_t
        results["routing_ok"] = (None if stressed else
                                 chain_pdr > 0.8 and bool(disc) and max(disc) < 2.0)
    elif algorithm in ("olsr", "letolsr"):
        results["converged_s"] = conv_t
        results["routing_ok"] = (None if stressed else
                                 chain_pdr > 0.85 and conv_t is not None and conv_t < 10.0)
    else:
        results["routing_ok"] = None if stressed else chain_pdr > 0.8
    results["etx_changed"] = (etx_t0 is not None and etx_t1 is not None
                              and etx_t0[0] != etx_t1[0])
    results["etx_inf_oor"] = math.isinf(etx_t0[1]) and math.isinf(etx_t1[1])
    if scenario in CONTINUOUS_SCENARIOS:
        # check 4 is a static-topology sanity check (ETX must move as the swarm
        # compresses); the patrol/mobility equivalent is the speed audit: the
        # average flown speed must match the commanded speed within +-10%.
        results["etx_changed"] = results["etx_inf_oor"] = True
        results["network_ok"] = (True if speed is None
                                 else bool(results.get("speed_ok", False)))
    elif scenario == "scale":
        # at constant density the (0,1) and farthest pairs are typically out of
        # range for the WHOLE run (and the graph is static for the first 30 s),
        # so the Phase 1 "ETX moves / inf OOR" check is inapplicable by
        # construction (same argument as grid's rigid drift); the meaningful
        # checks here are the section 8.3 scale metrics themselves.
        results["etx_changed"] = results["etx_inf_oor"] = True
        results["network_ok"] = True
    elif scenario == "grid":
        # the grid drifts rigidly (+40 m on every node), so pairwise distances
        # never change and the ETX-moves check is inapplicable by construction;
        # the meaningful check is that requested kills were actually applied.
        results["etx_changed"] = results["etx_inf_oor"] = True
        results["network_ok"] = True if kill_node is None else bool(kill_applied)
    else:
        results["network_ok"] = bool(results["etx_changed"] and results["etx_inf_oor"])
    if write_csv:
        v = verify_schema(trial.path, min_rows=int(duration / dt) * n)
        results["logger_ok"] = v["schema_ok"] and v["rows_ok"]
        results["csv_rows"] = v["rows"]
        results["csv_path"] = str(trial.path)
        results["log_rate_hz"] = round(v["rows"] / n / duration, 1)
    else:
        results["logger_ok"] = True
        results["log_rate_hz"] = None
    results["final_positions"] = [tuple(round(c, 3) for c in a.pos) for a in agents]
    if algorithm == "mpcare" and let_audit:
        from stats_utils import spearman_rho  # local import: scripts/ on path
        errs = [abs(p - act) / act for _, p, act in let_audit]
        errs15 = [abs(p - act) / act for v, p, act in let_audit if v <= 15.0]
        errs15b = [abs(p - act) / act for v, p, act in let_audit
                   if v <= 15.0 and act >= 2.0]   # exclude quantization-dominated
        results["let_audit_n"] = len(let_audit)
        results["let_median_rel_err"] = round(statistics.median(errs), 3)
        results["let_median_rel_err_vle15"] = (round(statistics.median(errs15), 3)
                                               if errs15 else None)
        results["let_median_rel_err_vle15_act2"] = (round(statistics.median(errs15b), 3)
                                                    if errs15b else None)
        _rho = (spearman_rho([p for _, p, _ in let_audit],
                             [act for _, _, act in let_audit])
                if len(let_audit) >= 5 else None)
        results["let_spearman"] = round(_rho, 3) if _rho is not None else None
        results["let_noinf_breaks"] = let_audit_noinf
    results["route_choice"] = dict(route_choice)
    results["trigger_events"] = dict(events)
    results["first_fire"] = {k: round(v, 2) for k, v in first_fire.items()}
    if algorithm == "qlgr":
        # learned-baseline telemetry: table size (per agent), end-to-end
        # route churn, and the overhead reality: zero control packets
        results["qlgr_route_changes"] = sum(
            qa.n_route_changes for qa in (qlgr_agents or []))
        results["qlgr_table_size"] = (max((len(qa.q) for qa in qlgr_agents),
                                          default=0) if qlgr_agents else 0)
        results["qlgr_explore_rate"] = round(
            sum(qa.n_explore for qa in (qlgr_agents or []))
            / max(1, sum(qa.n_attempts for qa in (qlgr_agents or []))), 3)
    # ---- phase 6 scalability metrics (plan section 8.3) ----
    # metric 1: per-UAV protocol compute per tick (perf_counter around every
    # real node's protocol work; includes GIL-serialized global work -> honest
    # UPPER bound on per-node compute). Baselines without a control plane
    # (gpsr, astar_*) measure 0.0 by construction: stateless forwarding.
    results["cpu_ms_per_node_tick"] = round(
        cpu_acc / max(1, int(duration / dt)) / n * 1000.0, 4)
    if deg_n:
        results["avg_degree"] = round(deg_acc / deg_n, 2)
    if algorithm == "mpcare":
        results["beacons_per_node_s"] = round(
            statistics.mean(beacon_tx) / duration, 3)
        results["void_recovery_events"] = sum(
            getattr(r, "recovery_events", 0) for r in routers)
        results["void_walk_drops"] = void_drops
        results["data_tx_per_node_s"] = round(
            statistics.mean(getattr(r, "mp_tx_count", 0) for r in routers)
            / duration, 3)
        results["convergence_def"] = CONVERGENCE_DEFS["mpcare"]
        target = set(static_end_links.get(0, ())) if static_end_links else set()
        ft = next((i * dt for i, v in enumerate(src_view_sets)
                   if i * dt < HOVER_UNTIL and target and target <= set(v)), None)
        results["full_conv_s"] = round(ft, 2) if ft is not None else None
        results["conv_scope"] = "static window [0, 30) s"
    elif algorithm in ("olsr", "letolsr"):
        results["convergence_def"] = CONVERGENCE_DEFS["olsr"]
        target: set = set()
        if static_end_links and 0 in static_end_links:
            seen, stack = {0}, [0]                     # static-window component
            while stack:
                c = stack.pop()
                for nb in static_end_links.get(c, ()):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            target = seen - {0}
        ft = next((i * dt for i, v in enumerate(src_view_sets)
                   if i * dt < HOVER_UNTIL and target and target <= set(v)), None)
        results["full_conv_s"] = round(ft, 2) if ft is not None else None
        results["conv_scope"] = "static window [0, 30) s"
    elif algorithm == "aodv":
        results["convergence_def"] = CONVERGENCE_DEFS["aodv"]
        results["full_conv_s"] = (round(conv_t, 2)
                                  if conv_t is not None and conv_t < HOVER_UNTIL
                                  else None)
        results["conv_scope"] = "static window [0, 30) s"
    else:
        results["convergence_def"] = CONVERGENCE_DEFS["gpsr"]
        results["full_conv_s"] = None
        results["conv_scope"] = "n/a"
    # control-plane overhead (AODV/OLSR)
    if bus is not None:
        results["control_pkts_per_node_s"] = round(
            bus.control_overhead_per_node(n, duration), 3)
    if protos and algorithm in ("olsr", "letolsr"):
        results["avg_mpr_size"] = round(sum(len(p.mpr_set) for p in protos) / n, 2)
    return results


def _deliver(p, t, stats, hop_counts, latencies, chain_hops, win, win_lat,
             route_choice=None, deliveries=None):
    pkt = p["pkt"]
    stats["delivered"] += 1
    if deliveries is not None:
        deliveries.append((t, pkt.created_t))
    hop_counts.append(pkt.hops)
    if p.get("latency_ms") is not None and p.get("first_fwd_t") is not None:
        # physical delay model (Phase 4): per-hop airtime + propagation +
        # injected latency (link.latency_ms) over the actual path, plus the
        # source queue wait. Reduces to the old wall-time semantics only when
        # hop pacing dominates; see NOTES/Phase_4.md metric definition note.
        latencies.append(p["latency_ms"] + (p["first_fwd_t"] - pkt.created_t))
    else:
        latencies.append(t - pkt.created_t)
    if p.get("chain"):
        stats["chain_delivered"] += 1
        chain_hops.append(pkt.hops)
    if route_choice is not None and 1 in pkt.path:
        route_choice["upper"] += 1
    elif route_choice is not None:
        route_choice["rim"] += 1
    for uid in set(pkt.path):
        win[uid].append((t, True))
        win_lat[uid].append(t - pkt.created_t)


def determinism_check(seed: int, out_dir: Path, algorithm: str) -> bool:
    a = run(seed, out_dir, write_csv=False, algorithm=algorithm)
    b = run(seed, out_dir, write_csv=False, algorithm=algorithm)
    keys = ("final_positions", "packets_delivered", "pdr", "chain_pdr", "min_chain_hops")
    return all(a[k] == b[k] for k in keys)


def main():
    ap = argparse.ArgumentParser(description="MP-CARE sim driver (Phases 1-2)")
    ap.add_argument("--algorithm", default="gpsr", choices=ALGORITHMS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--dur", type=float, default=90.0)
    ap.add_argument("--scenario", default="chain",
                    choices=("chain", "diamond", "mobility", "patrol", "grid",
                             "scale"))
    ap.add_argument("--loss", type=float, default=None,
                    help="global injected per-attempt packet loss 0..1")
    ap.add_argument("--badlink", action="store_true",
                    help="inject 0.85 packet loss on pair {0,1} during t=[15,25] s "
                         "(chain scenario; exercises the T1 trigger)")
    ap.add_argument("--kill", type=int, default=None, metavar="UAV_ID",
                    help="kill this UAV at t=20 s (diamond scenario; exercises T4)")
    ap.add_argument("--n-uav", type=int, default=None,
                    help="swarm size (default 5; node-failure experiment uses 10)")
    ap.add_argument("--speed", type=float, default=None,
                    help="mobility-scenario flight speed m/s (experiment 5)")
    ap.add_argument("--latency-ms", type=float, default=0.0,
                    help="injected per-hop one-way latency ms (experiment 3)")
    ap.add_argument("--no-csv", action="store_true")
    ap.add_argument("--skip-determinism", action="store_true")
    args = ap.parse_args()

    print(f"MP-CARE run: algorithm={args.algorithm} seed={args.seed} "
          f"duration={args.dur}s N={args.n_uav or N_UAV}")
    t0 = time.perf_counter()
    badlink = (15.0, 25.0, 0.85, 0, 1) if args.badlink else None
    kill_node = (20.0, args.kill) if args.kill is not None else None
    r = run(args.seed, Path(args.out), duration=args.dur, write_csv=not args.no_csv,
            algorithm=args.algorithm, scenario=args.scenario,
            injected_loss=args.loss, badlink=badlink, kill_node=kill_node,
            n_uav=args.n_uav or 5, speed=args.speed,
            injected_latency_ms=args.latency_ms)
    wall = time.perf_counter() - t0

    print(f"\nwall-clock {wall:.1f}s | chain PDR {r['chain_pdr']} "
          f"({r['chain_delivered']}/{r['chain_sent']} packets, min hops "
          f"{r['min_chain_hops']}) | arrival {r['max_arrival_s']}s | "
          f"latency avg {r['avg_latency_ms']} ms | log {r['log_rate_hz']} Hz/UAV")
    if "control_pkts_per_node_s" in r:
        print(f"control overhead: {r['control_pkts_per_node_s']} pkts/node/s"
              + (f" | avg MPR set size {r['avg_mpr_size']}" if "avg_mpr_size" in r else ""))
    if r.get("trigger_events"):
        print(f"trigger events: {r['trigger_events']} | first fires: {r['first_fire']}")
    if sum(r.get("route_choice", {}).values()):
        print(f"route choice (deliveries): {r['route_choice']}")
    checks = [
        ("1 spawn/stable t=10s", r["stability_ok"]),
        ("2 A* waypoints < 120s", r["waypoint_ok"]),
        (f"3 chain PDR ({args.algorithm})", r["routing_ok"]),
        ("4 ETX moves / inf OOR", r["network_ok"]),
        ("5 CSV schema >=10Hz", r["logger_ok"]),
    ]
    if r.get("stressed"):
        checks[2] = ("3 chain PDR (n/a under injected stress — "
                     "comparative margin is the criterion)", True)
    ok = True
    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        ok &= bool(passed)
    if not args.skip_determinism:
        det = determinism_check(args.seed, Path(args.out), args.algorithm)
        print(f"  [{'PASS' if det else 'FAIL'}] 6 same seed -> identical run")
        ok &= det
    if "csv_path" in r:
        print(f"  csv: {r['csv_path']} ({r['csv_rows']} rows)")
    print("\nRESULT:", "ALL CHECKS PASS" if ok else "FAILURES PRESENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
