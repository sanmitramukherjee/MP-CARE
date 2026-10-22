"""Phase 1 swarm core: distance-only A* planner, GPSR forwarding, UAV agent.

Pure Python, deterministic, ROS-free. The ROS 2 nodes (planner_node.py) wrap
these classes; scripts/run_sim.py drives them headless for the Phase 1 MVP.

References:
- A*: Hart, Nilsson & Raphael (1968) — distance-only cost in Phase 1.
- GPSR: Karp & Kung, MOBICOM 2000 — greedy mode with perimeter fallback over a
  Gabriel-graph planarization, right-hand-rule face traversal.
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass, field

INF = math.inf


# ------------------------------------------------------------------ A* planner

def astar_path(start: int, goal: int,
               links: dict[int, set[int]],
               positions: dict[int, tuple[float, float, float]],
               max_cost: float = INF,
               edge_cost=None,
               heuristic=None) -> list[int] | None:
    """A* over the current connectivity graph.

    Default (edge_cost=None): distance-only cost — the Phase 1 baseline B1.
    edge_cost(a, b)  : optional custom edge cost (B5 "AT-PP style": pass the
                       ETX of the link so the planner prefers higher-throughput
                       routes even when they are geometrically longer).
    heuristic(n)     : must be admissible for the chosen cost. For ETX costs the
                       Euclidean distance is NOT admissible (one hop can cost ~1
                       while covering 300 m), so pass dist/max_range.
    Returns the node-id path INCLUDING both endpoints, or None if unreachable.
    Deterministic: neighbors explored in id order.
    """

    def dist2d(a: int, b: int) -> float:
        return math.dist(positions[a][:2], positions[b][:2])

    if edge_cost is None:
        edge_cost = dist2d
    if heuristic is None:
        heuristic = lambda n: dist2d(n, goal)  # noqa: E731

    openq: list[tuple[float, float, int]] = [(heuristic(start), 0.0, start)]
    g = {start: 0.0}
    parent: dict[int, int] = {}
    closed: set[int] = set()
    while openq:
        f, gc, cur = heapq.heappop(openq)
        if cur in closed:
            continue
        closed.add(cur)
        if cur == goal:
            return _reconstruct(parent, cur)
        for nb in sorted(links.get(cur, ())):
            if nb in closed or nb == cur:
                continue
            ng = gc + edge_cost(cur, nb)
            if ng < g.get(nb, INF):
                g[nb] = ng
                parent[nb] = cur
                heapq.heappush(openq, (ng + heuristic(nb), ng, nb))
    return None


def _reconstruct(parent: dict[int, int], end: int) -> list[int]:
    path = [end]
    while end in parent:
        end = parent[end]
        path.append(end)
    path.reverse()
    return path


# ------------------------------------------------------------------ GPSR

@dataclass
class Packet:
    pid: int
    src: int
    dst: int
    dst_pos: tuple[float, float]
    created_t: float
    hops: int = 0
    mode: str = "greedy"            # greedy | perimeter
    entry_edge: tuple[int, int] | None = None  # (prev, cur) edge the packet arrived on
    prev_node: int | None = None
    path: list[int] = field(default_factory=list)


class GpsrEngine:
    """Per-node GPSR forwarding decisions (greedy + perimeter, Karp & Kung 2000)."""

    def __init__(self, node_id: int):
        self.node_id = node_id

    def next_hop(self, pkt: Packet,
                 positions: dict[int, tuple[float, float, float]],
                 links: dict[int, set[int]]) -> int | None:
        """Next-hop node id, or None if the node cannot forward (drop)."""
        if self.node_id == pkt.dst:
            return None
        if pkt.mode == "greedy":
            return self._greedy(pkt, positions, links)
        return self._perimeter(pkt, positions, links)

    def _greedy(self, pkt, positions, links):
        me = positions[self.node_id][:2]
        cur_d = math.dist(me, pkt.dst_pos)
        best, best_d = None, cur_d
        for nb in sorted(links.get(self.node_id, ())):
            if nb == pkt.prev_node:
                continue  # never bounce straight back in greedy mode
            d = math.dist(positions[nb][:2], pkt.dst_pos)
            if d < best_d:
                best, best_d = nb, d
        if best is None:
            # stuck: fall back to perimeter mode at this node
            pkt.mode = "perimeter"
            pkt.entry_edge = (pkt.prev_node, self.node_id) if pkt.prev_node is not None else None
            return self._perimeter(pkt, positions, links)
        return best

    def _perimeter(self, pkt, positions, links):
        me = positions[self.node_id][:2]
        nbs = [nb for nb in sorted(links.get(self.node_id, ()))
               if self._gg_edge(self.node_id, nb, positions)]
        if not nbs:
            return None
        entry_dir = self._entry_dir(pkt, positions, me)
        entry_ang = math.atan2(entry_dir[1], entry_dir[0])
        best, best_rel = None, None
        for nb in nbs:
            v = (positions[nb][0] - me[0], positions[nb][1] - me[1])
            rel = (math.atan2(v[1], v[0]) - entry_ang) % (2 * math.pi)
            if best_rel is None or rel < best_rel:
                best, best_rel = nb, rel
        if best == pkt.prev_node and len(nbs) > 1:
            # right-hand rule picked the edge we came in on; rotate once more
            rels = sorted(((math.atan2(positions[nb][1] - me[1],
                                       positions[nb][0] - me[0]) - entry_ang) % (2 * math.pi), nb)
                          for nb in nbs)
            best = rels[1][1]
        return best

    def _entry_dir(self, pkt, positions, me) -> tuple[float, float]:
        if pkt.entry_edge:
            prev, cur = pkt.entry_edge
            if cur == self.node_id and prev in positions:
                pv = positions[prev][:2]
                d = (me[0] - pv[0], me[1] - pv[1])
                if d != (0.0, 0.0):
                    return d
        if pkt.prev_node is not None and pkt.prev_node in positions:
            pv = positions[pkt.prev_node][:2]
            d = (me[0] - pv[0], me[1] - pv[1])
            if d != (0.0, 0.0):
                return d
        return (1.0, 0.0)

    def _gg_edge(self, a: int, b: int, positions) -> bool:
        """Gabriel-graph test: no third node inside the diametral circle of (a, b)."""
        pa, pb = positions[a][:2], positions[b][:2]
        r2 = math.dist(pa, pb) ** 2
        cx, cy = (pa[0] + pb[0]) / 2, (pa[1] + pb[1]) / 2
        for c, pc3 in positions.items():
            if c in (a, b):
                continue
            if math.dist(pc3[:2], (cx, cy)) ** 2 < r2 / 4.0:
                return False
        return True


# ------------------------------------------------------------------ agent

@dataclass
class WaypointMission:
    """Waypoints to visit in order (Phase 1: simple list; Phase 2+ = tasks)."""
    waypoints: list[tuple[float, float, float]]
    current: int = 0

    def current_wp(self):
        return self.waypoints[self.current] if self.current < len(self.waypoints) else None

    def advance(self):
        self.current += 1


class UavAgent:
    """One UAV: A* route to its waypoint + GPSR forwarding engine + telemetry."""

    def __init__(self, uav_id: int, start: tuple[float, float, float],
                 mission: WaypointMission, speed_mps: float = 8.0,
                 seed: int = 0):
        self.uav_id = uav_id
        self.pos = tuple(start)
        self.mission = mission
        self.speed = speed_mps
        self.route: list[int] = []          # multi-hop route used by GPSR for delivery
        self.route_cost: float = 0.0
        self.route_hops: int = 0
        self.reroute_event = False
        self.reroute_trigger = ""
        self.rng = random.Random(seed)
        self.gpsr = GpsrEngine(uav_id)
        # telemetry (filled by the sim loop each tick)
        self.neighbors: list[int] = []
        self.etx: dict[int, float] = {}
        self.pdr_window: list[int] = []
        self.queue_depth: int = 0

    # ---- mobility ------------------------------------------------------
    def move_toward(self, target3: tuple[float, float, float], dt: float):
        d = math.dist(self.pos, target3)
        step = self.speed * dt
        if d <= step:
            self.pos = tuple(target3)
            return True
        f = step / d
        self.pos = tuple(self.pos[i] + (target3[i] - self.pos[i]) * f for i in range(3))
        return False

    # ---- routing (Phase 1: distance-only A*) ---------------------------
    def plan_route(self, links: dict[int, set[int]],
                   positions: dict[int, tuple[float, float, float]]):
        """Route this UAV to the next UAV in the ring (Phase 1 waypoint mission:
        deliver a packet to the next-ring neighbor via A* over the swarm graph)."""
        self.reroute_event = False
        wp = self.mission.current_wp()
        if wp is None:
            return
        target_id = wp[3] if len(wp) > 3 else None
        if target_id is None or target_id == self.uav_id:
            return
        path = astar_path(self.uav_id, target_id, links, positions)
        if path != self.route:
            self.reroute_event = True
            self.reroute_trigger = "route-changed"
        self.route = path or []
        self.route_cost = sum(
            math.dist(positions[a][:2], positions[b][:2])
            for a, b in zip(self.route, self.route[1:])) if self.route else 0.0
        self.route_hops = max(0, len(self.route) - 1)

    # ---- packet forwarding (GPSR) --------------------------------------
    def forward(self, pkt: Packet, positions, links) -> int | None:
        nxt = self.gpsr.next_hop(pkt, positions, links)
        if nxt is not None:
            pkt.prev_node = self.uav_id
            pkt.hops += 1
            pkt.path.append(nxt)
        return nxt
