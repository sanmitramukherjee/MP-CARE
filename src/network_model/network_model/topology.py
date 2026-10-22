"""Pure topology-blob helpers (ROS-side plumbing without rclpy).

The `/swarm/topology` blob (published by network_model_node) is JSON:

    {"t": <sim seconds>, "links": [[a, b, dist_m, etx], ...]}

with etx = None meaning "out of range" (infinite ETX). The blob carries no
absolute coordinates — positions come from the per-UAV `/uav_i/odom`
subscriptions. These helpers turn the blob into the dict shapes the
simulation cores expect:

    links: {uid: set(neighbor_uids)}     # only finite-ETX edges
    etx:   {(a, b): float}               # finite entries only, symmetric

Keeping them pure means the exact code path is unit-testable on Windows
(no ROS install required) — see tests/test_phase4_ros.py.

Staleness: a UAV whose odometry stopped updating (killed / crashed node) is
pruned from positions once it has been silent for `stale_s` — mirroring what
`scripts/run_sim.py` does when a node is killed (the UAV vanishes from the
network; beacons stop; packets in flight die with it).
"""

from __future__ import annotations

import json


def parse_blob(data: str | bytes | dict) -> tuple[float, dict, dict]:
    """blob -> (t_s, links, etx)."""
    blob = data if isinstance(data, dict) else json.loads(data)
    t = float(blob.get("t", 0.0))
    links: dict[int, set[int]] = {}
    etx: dict[tuple[int, int], float] = {}
    for a, b, _d, e in blob.get("links", ()):
        a, b = int(a), int(b)
        if e is not None:
            links.setdefault(a, set()).add(b)
            links.setdefault(b, set()).add(a)
            etx[(a, b)] = float(e)
            etx[(b, a)] = float(e)
    return t, links, etx


def prune_stale(odom: dict[int, tuple[float, float, float]],
                last_seen: dict[int, float], t_now: float,
                stale_s: float = 3.0) -> dict[int, tuple[float, float, float]]:
    """Drop UAVs whose odometry went silent (kill/failure emulation)."""
    return {uid: p for uid, p in odom.items()
            if t_now - last_seen.get(uid, t_now) <= stale_s}


def snapshot_etx(links: dict, etx: dict, positions: dict,
                 a: int, b: int, max_range_m: float = 300.0) -> float:
    """ETX for (a, b) consistent with the sim's NetworkSnapshot.etx_of:
    inf when out of range, either node missing, or no link in the blob."""
    import math
    if a not in positions or b not in positions:
        return math.inf
    if b not in links.get(a, ()):
        return math.inf
    return float(etx.get((a, b), math.inf))
