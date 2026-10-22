"""ROS 2 network-model node (plan section 3.2/3.3).

Subscribes to every /uav_i/odom (ground truth in simulation), evaluates the
explicit network model at the beacon interval, and publishes /swarm/topology
as a JSON blob: {t, links: [[a, b, dist_m, etx], ...]}. Odometry silent for
longer than `stale_s` is pruned from the blob (same helper the agents use),
so a killed UAV's links vanish within one beacon interval.

Requires rclpy (target distro: ROS 2 Lyrical on Ubuntu 26.04 — see
NOTES/testROS2.md). For the headless MVP use scripts/run_sim.py, which drives
the same NetworkParams/LinkModel/NetworkSnapshot core directly.
"""

import json
import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from network_model.model import NetworkParams, LinkModel, NetworkSnapshot
from network_model.topology import prune_stale


class NetworkModelNode(Node):

    def __init__(self):
        super().__init__("network_model")
        self.declare_parameters("", [
            ("alpha", 2.5), ("pl0_db", 46.4), ("tx_power_dbm", 20.0),
            ("noise_floor_dbm", -90.0), ("bandwidth_mbps", 11.0),
            ("snr50_db", 1.7), ("k_per_db", 0.9), ("max_range_m", 300.0),
            ("beacon_interval_s", 1.0), ("injected_loss", 0.0),
            ("injected_latency_ms", 0.0),            ("queue_size", 64), ("rng_seed", 1),
            ("stale_s", 3.0), ("n_uav", 5),
        ])
        gp = self.get_parameters_by_prefix("")
        self.params = NetworkParams(
            alpha=gp["alpha"].value, pl0_db=gp["pl0_db"].value,
            tx_power_dbm=gp["tx_power_dbm"].value,
            noise_floor_dbm=gp["noise_floor_dbm"].value,
            bandwidth_mbps=gp["bandwidth_mbps"].value, snr50_db=gp["snr50_db"].value,
            k_per_db=gp["k_per_db"].value, max_range_m=gp["max_range_m"].value,
            beacon_interval_s=gp["beacon_interval_s"].value,
            injected_loss=gp["injected_loss"].value,
            injected_latency_ms=gp["injected_latency_ms"].value,
            queue_size=gp["queue_size"].value, rng_seed=gp["rng_seed"].value)
        self.link = LinkModel(self.params)
        self.n_uav = int(gp["n_uav"].value)
        self.stale_s = float(gp["stale_s"].value)
        self.positions: dict[int, tuple[float, float, float]] = {}
        self.last_seen: dict[int, float] = {}     # node-clock time of last odom
        self._stale_logged: set[int] = set()
        self.subs = []
        for i in range(self.n_uav):
            self.subs.append(self.create_subscription(
                Odometry, f"/uav_{i}/odom",
                (lambda i_: lambda m: self._odom(i_, m))(i), 10))
        self.pub = self.create_publisher(String, "/swarm/topology", 10)
        period = max(0.05, self.params.beacon_interval_s)
        self.timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"network model up: alpha={self.params.alpha} range={self.params.max_range_m} m, "
            f"beacon {period} s")

    def _odom(self, uav_id: int, msg: Odometry):
        p = msg.pose.pose.position
        self.positions[uav_id] = (p.x, p.y, p.z)
        self.last_seen[uav_id] = self.get_clock().now().nanoseconds * 1e-9

    def _tick(self):
        if len(self.positions) < self.n_uav:
            return  # not all odometry streams up yet
        t_now = self.get_clock().now().nanoseconds * 1e-9
        # Same staleness semantics as the agents (prune_stale, stale_s):
        # a killed/silent UAV's links leave the blob within one beacon
        # interval instead of persisting forever (bug found in the native
        # dual-boot Tier-3 run: post-kill echo still showed node-2 links).
        alive = prune_stale(self.positions, self.last_seen, t_now, self.stale_s)
        for uid in sorted(set(self.positions) - set(alive)):
            if uid not in self._stale_logged:
                self.get_logger().warning(
                    f"uav {uid} odom silent > {self.stale_s} s — pruned from topology")
                self._stale_logged.add(uid)
        for uid in set(alive) & self._stale_logged:
            self._stale_logged.discard(uid)   # odom resumed; re-arm the warn
        snap = NetworkSnapshot(alive, self.link, t_now)
        links = [[a, b, round(snap.dist[(a, b)], 2),
                  (None if math.isinf(snap.etx[(a, b)]) else round(snap.etx[(a, b)], 3))]
                 for (a, b) in snap.etx if a < b]
        msg = String()
        msg.data = json.dumps({"t": snap.t_s, "links": links})
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = NetworkModelNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
