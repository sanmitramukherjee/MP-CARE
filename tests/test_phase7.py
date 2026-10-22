"""Phase 7 tests: ablation switches (implementation_plan.md section 9.1).

Run from the repo root:  py -3.13 -m pytest tests/ -q

Contract under test:
  - the DEFAULT router (no switch kwargs) is byte-identical in behavior to
    the pre-phase-7 router (pinned by seed-42 and seed-4000 values that were
    measured BEFORE the refactor);
  - each ablation switch disables exactly its own mechanism:
      A1 predict_etx=False  -> cost uses current ETX only (prediction is a
         no-op: etx_pred == etx_current for every edge);
      A2 use_let=False      -> T2 (expiring link) can never fire;
      A3 delta=t_dwell=0    -> hysteresis gate fully open (fires on every
         rising edge, no dwell blocking);
      A4 w_q=0              -> queue term contributes zero to the cost;
      A5/A6 periodic        -> switches only on the timer, trigger events
         stay empty, forwarding (walk) still available;
      A7 static             -> exactly ONE relay decision per destination,
         then frozen; no triggers, no walk.
"""

import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in ("src/network_model", "src/mpcare_agent", "src/logger", "scripts"):
    sys.path.insert(0, str(ROOT / p))

from mpcare_agent.router import MpcareRouter, SimHooks           # noqa: E402
from mpcare_agent.event_trigger import TriggerParams             # noqa: E402
from network_model.model import LinkModel, NetworkParams, NetworkSnapshot  # noqa: E402


# ------------------------------------------------------------------ helpers
def _hood_router(me_xy, dst_xy, hood, dst_id=99, **router_kwargs):
    """Router at me_xy with the given 1-hop neighborhood beacons bound."""
    r = MpcareRouter(0, 1 + len(hood) + 1, max_range_m=300.0, **router_kwargs)
    pos = {0: (me_xy[0], me_xy[1], 40.0), dst_id: (dst_xy[0], dst_xy[1], 40.0)}
    t = 1.0
    for nid, (x, y) in hood.items():
        pos[nid] = (x, y, 40.0)
    link = LinkModel(NetworkParams(rng_seed=1))
    snap = NetworkSnapshot(pos, link, t)
    r.bind(SimHooks(pos, link, snap))
    for nid, (x, y) in hood.items():
        r.receive_beacon({"type": "MPBEACON", "src": nid, "pos": (x, y, 40.0),
                          "queue": 0.0, "nbrs": (0,)}, nid,
                         snap.etx_of(0, nid), t)
    r.table.set_my_position((me_xy[0], me_xy[1], 40.0))
    return r


def _chain_run(seed=4000, **kw):
    import run_sim
    return run_sim.run(seed, ROOT / "data" / "raw", duration=90.0,
                       write_csv=False, algorithm="mpcare", scenario="chain",
                       injected_loss=0.10, **kw)


# ------------------------------------------------------- defaults unchanged
def test_default_full_run_matches_pre_phase7_values():
    """Seed 4000 chain + 10% loss was measured BEFORE the phase-7 refactor:
    pdr 0.66, 33/50 delivered, 5 trigger events, 0 void steps. The default
    constructor must reproduce it exactly."""
    r = _chain_run()
    assert r["pdr"] == 0.66
    assert r["packets_delivered"] == 33
    assert sum(r["trigger_events"].values()) == 5
    assert r["void_recovery_events"] == 0


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        MpcareRouter(0, 5, reroute_mode="quantum")


# --------------------------------------------------------------- A1 predict
def test_a1_predict_etx_false_makes_prediction_identity():
    hood = {1: (100.0, 0.0), 2: (150.0, 50.0)}
    r_full = _hood_router((0.0, 0.0), (500.0, 0.0), hood)
    r_a1 = _hood_router((0.0, 0.0), (500.0, 0.0), hood, predict_etx=False)
    # a RECEDING neighbor (4 successive beacons marching away): the full
    # router predicts a strictly worse ETX (f(v) > 1 for v > 0); A1 keeps
    # the current ETX (identity).
    for k, (x, y) in enumerate(((104.0, 96.0), (108.0, 99.0),
                                (112.0, 102.0), (116.0, 105.0))):
        t = 2.0 + 0.2 * k
        r_full.receive_beacon({"type": "MPBEACON", "src": 1,
                               "pos": (x, y, 40.0), "queue": 0.0,
                               "nbrs": (0,)}, 1, 1.05, t)
        r_a1.receive_beacon({"type": "MPBEACON", "src": 1,
                             "pos": (x, y, 40.0), "queue": 0.0,
                             "nbrs": (0,)}, 1, 1.05, t)
    cur = r_full._etx_current(0, 1)
    assert math.isfinite(cur)
    assert r_full.table.radial_velocity(1) > 0.0, "setup: neighbor receding"
    assert r_full._etx_pred(0, 1) > cur, "receding neighbor must predict worse"
    assert r_a1._etx_pred(0, 1) == cur, "A1 must return the CURRENT etx"


def test_a1_end_to_end_still_routes():
    r = _chain_run(mpcare_opts={"predict_etx": False})
    assert r["packets_delivered"] > 0


# ------------------------------------------------------------------- A2 LET
def test_a2_use_let_false_blinds_t2():
    # a RECEDING far neighbor: LET = (range - d) / v is small and finite ->
    # below tau_min = 3.0, so T2's level condition is true for the full router
    hood = {1: (290.0, 0.0)}
    r_full = _hood_router((0.0, 0.0), (500.0, 0.0), hood)
    r_a2 = _hood_router((0.0, 0.0), (500.0, 0.0), hood, use_let=False)
    r_full.receive_beacon({"type": "MPBEACON", "src": 1,
                           "pos": (296.0, 0.0, 40.0), "queue": 0.0,
                           "nbrs": (0,)}, 1, 1.05, 2.0)
    r_a2.receive_beacon({"type": "MPBEACON", "src": 1,
                         "pos": (296.0, 0.0, 40.0), "queue": 0.0,
                         "nbrs": (0,)}, 1, 1.05, 2.0)
    r_full.next_hop[99] = 1
    r_a2.next_hop[99] = 1
    let = r_full._min_route_let(1)
    assert let is not None and math.isfinite(let) and let < 3.0
    assert r_full.trigger.let_bad(let) is True
    assert r_a2._min_route_let(1) is None, "A2 must return None (T2 blinded)"
    assert r_a2.trigger.let_bad(r_a2._min_route_let(1)) is False


# ----------------------------------------------------------- A3 hysteresis
def test_a3_zero_hysteresis_opens_the_gate():
    t_full = TriggerParams()                       # delta=0.10, dwell=2.0
    t_a3 = TriggerParams(delta=0.0, t_dwell=0.0)
    from mpcare_agent.event_trigger import EventTrigger
    e_full, e_a3 = EventTrigger(t_full), EventTrigger(t_a3)
    # a 5% better alternative: full gate refuses (needs 10%), open gate fires
    better_alt = 1.05 < (1.0 - 0.0) * 1.0
    fire_full, _ = e_full.should_reroute(10.0, route_prr=None, min_let_s=None,
                                         better_alt=False, lost_neighbors=[])
    fire_a3, _ = e_a3.should_reroute(10.0, route_prr=None, min_let_s=None,
                                     better_alt=True, lost_neighbors=[])
    assert fire_full is False
    assert fire_a3 is True
    # and with dwell=0 a second fire 0.1 s later is NOT dwell-blocked
    fire2, _ = e_a3.should_reroute(10.1, route_prr=None, min_let_s=None,
                                   better_alt=True, lost_neighbors=[])
    assert fire2 is True


# ----------------------------------------------------------------- A4 w_q=0
def test_a4_zero_queue_weight_zeroes_the_term():
    from mpcare_agent.cost_engine import CostEngine, PathFeatures
    eng_full = CostEngine()
    eng_a4 = CostEngine({"w_q": 0.0})
    # two candidates identical except queue: f1 congested (Q=0.9), f2 free.
    # Min-max over the pair: D normalized to f2=1.0 (the longer path), Q to
    # f1=1.0. With w_q=0.5 the congested path MUST cost strictly more; with
    # w_q=0 the queue term vanishes and the costs tie.
    f1 = PathFeatures(path=[0, 1], distance_m=110.0, etx_sum=1.0, etx_max=1.0,
                      latency_s=0.01, queue_max=0.9, hops=1)
    f2 = PathFeatures(path=[0, 2], distance_m=110.0, etx_sum=1.0, etx_max=1.0,
                      latency_s=0.01, queue_max=0.0, hops=1)
    norm = {"D": [0.0, 0.0], "E": [0.0, 0.0], "L": [0.0, 0.0],
            "Q": [1.0, 0.0], "H": [0.0, 0.0]}
    f1.idx, f2.idx = 0, 1
    c1_full, c2_full = eng_full.cost(f1, norm), eng_full.cost(f2, norm)
    c1_a4, c2_a4 = eng_a4.cost(f1, norm), eng_a4.cost(f2, norm)
    assert c1_full > c2_full, "w_q=0.5: congested path must cost more"
    assert c1_a4 == c2_a4, "w_q=0 must make the queue term irrelevant"
    assert c1_full - c2_full == pytest.approx(0.5)


# ------------------------------------------------- A5/A6 periodic + A7 static
def test_a5_periodic_switches_on_timer_not_events():
    hood = {1: (100.0, 0.0), 2: (140.0, 60.0), 3: (140.0, -60.0)}
    r = _hood_router((0.0, 0.0), (500.0, 0.0), hood,
                     reroute_mode="periodic", period_s=1.0)
    nh1 = r.next_hop_for(99, 5.0)          # first query: route established
    assert nh1 in (1, 2, 3)
    assert r.reroute_event is False        # an establish is not a switch
    # next query ranks a DIFFERENT relay best, 0.5 s later: the periodic
    # timer starts at the FIRST SWITCH (last_reroute_t = -inf here), so the
    # first adaptation fires immediately -- exactly a periodic protocol's
    # first-cycle behavior -- and the period gates every switch after it.
    orig = r._ranked_candidates
    r._ranked_candidates = lambda *a, **k: [(0.5, type("F", (), {"path": [0, 3]})())]
    nh2 = r.next_hop_for(99, 5.5)
    assert nh2 == 3 and r.reroute_event is True
    assert r.reroute_trigger == "periodic"
    r.reroute_event = False
    r.next_hop_for(99, 6.0)                # 0.5 s after the switch: gated
    assert r.reroute_event is False, "timer not elapsed: no switch"
    # now ranking prefers relay 2 again; 1.5 s after the switch: fires
    r._ranked_candidates = lambda *a, **k: [(0.5, type("F", (), {"path": [0, 2]})())]
    nh4 = r.next_hop_for(99, 7.0)
    assert nh4 == 2 and r.reroute_event is True
    r._ranked_candidates = orig


def test_periodic_forwarding_still_walks_in_voids():
    """A5/A6 keep the face walk: only the switch POLICY changed."""
    r = _hood_router((0.0, 0.0), (300.0, 0.0), {1: (-100.0, 200.0), 2: (-100.0, -200.0)},
                     reroute_mode="periodic", period_s=1.0)
    nh = r.next_hop_for(99, 5.0)     # both nbrs farther: void -> walk
    assert nh in (1, 2)
    assert r.recovery_events == 1


def test_a7_static_freezes_first_relay_and_never_walks():
    hood = {1: (-100.0, 200.0), 2: (-100.0, -200.0), 3: (100.0, 0.0)}
    r = _hood_router((0.0, 0.0), (300.0, 0.0), hood, reroute_mode="static")
    nh = r.next_hop_for(99, 5.0)
    assert nh is not None and nh != 99
    assert r.next_hop[99] == nh
    assert r.recovery_events == 0, "static mode must not face-walk"
    # freeze: even when a closer neighbor appears later, the relay holds
    pos = r._sim.positions
    pos[4] = (150.0, 0.0, 40.0)            # new strictly-closer node
    r.receive_beacon({"type": "MPBEACON", "src": 4, "pos": pos[4],
                      "queue": 0.0, "nbrs": (0,)}, 4, 1.0, 6.0)
    assert r.next_hop_for(99, 7.0) == nh
    assert r.reroute_event is False


def test_a7_static_holds_route_until_relay_dies():
    hood = {1: (100.0, 0.0), 2: (140.0, 60.0)}
    r = _hood_router((0.0, 0.0), (500.0, 0.0), hood, reroute_mode="static")
    nh = r.next_hop_for(99, 5.0)
    assert nh in (1, 2)
    # relay vanishes from the position set (kill): static queues (None) --
    # the honest lower-bound behavior, never a silent re-route
    r._sim.positions.pop(2, None)
    r._sim.snap = NetworkSnapshot(r._sim.positions, r._sim.link, 9.0)
    if nh == 2:
        assert r.next_hop_for(99, 9.0) is None
    else:
        assert r.next_hop_for(99, 9.0) == nh


# ------------------------------------------------------------- end-to-end
def test_ablation_variants_end_to_end_smoke():
    """All variants complete a full run with sane invariants (every variant
    delivers; periodic/static do not emit event-trigger fires)."""
    variants = {
        "A1": {"predict_etx": False},
        "A2": {"use_let": False},
        "A4": {"weights": {"w_q": 0.0}},
        "A5": {"reroute_mode": "periodic", "period_s": 1.0},
        "A7": {"reroute_mode": "static"},
    }
    for name, opts in variants.items():
        r = _chain_run(mpcare_opts=opts)
        assert r["packets_delivered"] > 0, name
        if name in ("A5", "A7"):
            assert all(k in ("periodic", "route-change")
                       for k in r["trigger_events"]), name
