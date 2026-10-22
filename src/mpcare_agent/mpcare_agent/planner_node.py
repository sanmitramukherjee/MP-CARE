"""ROS 2 wrapper: one MP-CARE agent node per UAV — the Task B wiring.

Each agent process subscribes ALL odometry topics (the swarm context; the
/swarm/topology blob carries links but no absolute positions) and advances
the SAME pure swarm adapter (``MpcareSwarmAdapter`` — identical per-tick
sequence as scripts/run_sim.py) with every received topology blob. What this
process publishes for ITS uav_id:

    /uav_<i>/route      std_msgs/String   {"route": [...]} on change, and
                        {"route": [...], "trigger": <T-name>} on an
                        event-triggered re-route (T1–T4)
    /uav_<i>/cmd_vel    geometry_msgs/Twist  toward /swarm/assignment
    /swarm/telemetry    std_msgs/String   one flat §3.4-schema JSON row per
                        tick (the logger node writes these to CSV)

Honest scope (see NOTES/testROS2.md): this validates the MP-CARE *control
plane* (neighbor knowledge, predictive costs, trigger-gated re-routing,
telemetry/logging) inside ROS 2. A hop-by-hop data plane over ROS topics is
not implemented; pdr/throughput fields are therefore 0.0 placeholders — the
PDR evidence comes from the headless lab (NOTES/Phase_4.md).

Requires rclpy (target distro: ROS 2 Lyrical on Ubuntu 26.04); the headless
equivalent is scripts/run_sim.py.
"""

import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String

from mpcare_agent.ros_adapter import MpcareSwarmAdapter


class MpcareAgentNode(Node):

    def __init__(self):
        super().__init__("mpcare_agent")
        self.declare_parameters("", [
            ("uav_id", 0), ("n_uav", 5), ("speed_mps", 8.0),
            ("wp_tolerance_m", 1.0), ("max_range_m", 300.0), ("stale_s", 3.0),
            ("rng_seed", 1), ("telemetry_rate_hz", 10.0),
        ])
        gp = self.get_parameters_by_prefix("")
        self.uav_id = int(gp["uav_id"].value)
        self.n_uav = int(gp["n_uav"].value)
        self.speed = float(gp["speed_mps"].value)
        self.tol = float(gp["wp_tolerance_m"].value)

        self.adapter = MpcareSwarmAdapter(
            self.n_uav, max_range_m=float(gp["max_range_m"].value),
            stale_s=float(gp["stale_s"].value),
            rng_seed=int(gp["rng_seed"].value))
        self.goal = None
        self.pos = None
        self.vel = (0.0, 0.0, 0.0)
        self._last_route: list[int] | None = None
        self._last_trigger = ""
        self._hb = 0

        # swarm context: every odom topic feeds the adapter
        for j in range(self.n_uav):
            self.create_subscription(
                Odometry, f"/uav_{j}/odom",
                (lambda j_: lambda m: self._odom(j_, m))(j), 10)
        self.create_subscription(String, "/swarm/topology", self._topology, 10)
        self.create_subscription(PoseStamped, "/swarm/assignment", self._assignment, 10)
        self.pub_cmd = self.create_publisher(Twist, f"/uav_{self.uav_id}/cmd_vel", 10)
        self.pub_route = self.create_publisher(String, f"/uav_{self.uav_id}/route", 10)
        self.pub_tel = self.create_publisher(String, "/swarm/telemetry", 10)
        self.create_timer(1.0 / max(1.0, float(gp["telemetry_rate_hz"].value)),
                          self._tick)
        self.get_logger().info(
            f"mpcare agent {self.uav_id} up (MP-CARE router wired: predictive "
            f"cost + event triggers)")

    # ------------------------------------------------------------ inputs
    def _odom(self, uav: int, msg: Odometry):
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        self.adapter.on_odom(uav, (p.x, p.y, p.z),
                             self.get_clock().now().nanoseconds * 1e-9)
        if uav == self.uav_id:
            self.pos = (p.x, p.y, p.z)
            self.vel = (v.x, v.y, v.z)

    def _topology(self, msg: String):
        try:
            self.adapter.on_topology(msg.data)
        except (ValueError, KeyError) as exc:
            self.get_logger().warn(f"bad topology blob: {exc}",
                                   throttle_duration_sec=5.0)
            return
        route = self.adapter.route_of(self.uav_id)
        if route is not None and route != self._last_route:
            self._last_route = route
            self._last_trigger = next(
                (ev["trigger"] for ev in self.adapter.route_events
                 if ev["uav_id"] == self.uav_id), "new-route")
            self._publish_route(route, self._last_trigger)
        elif route is None and self._last_route is not None:
            # honest no-route across a partition (kill / staleness): one
            # explicit route-lost event, then the topic goes silent
            self._last_route = None
            self._publish_route([], "route-lost")

    def _assignment(self, msg: PoseStamped):
        p = msg.pose.position
        self.goal = (p.x, p.y, p.z)

    # ----------------------------------------------------------- outputs
    def _publish_route(self, route: list[int], trigger: str):
        out = String()
        out.data = json.dumps({"route": route, "trigger": trigger})
        self.pub_route.publish(out)

    def _tick(self):
        now_ms = int(self.get_clock().now().nanoseconds / 1e6)
        pos = self.pos

        # route heartbeat at ~1 Hz so late-joining observers (volatile QoS)
        # still see the current route; a lost route publishes nothing here.
        self._hb += 1
        if self._last_route and self._hb % 10 == 0:
            self._publish_route(self._last_route, self._last_trigger or "steady")

        cmd = Twist()
        if pos is not None and self.goal is not None:
            d = math.dist(pos, self.goal)
            if d > self.tol:
                cmd.linear.x = min(self.speed, d)  # heading: offboard controller
            else:
                self.goal = None
        self.pub_cmd.publish(cmd)

        row = self.adapter.telemetry_row(
            self.uav_id,
            timestamp_ms=now_ms,
            position_x=pos[0] if pos else 0.0,
            position_y=pos[1] if pos else 0.0,
            position_z=pos[2] if pos else 0.0,
            velocity_x=self.vel[0], velocity_y=self.vel[1], velocity_z=self.vel[2],
            target_waypoint_x=self.goal[0] if self.goal else 0.0,
            target_waypoint_y=self.goal[1] if self.goal else 0.0,
            target_waypoint_z=self.goal[2] if self.goal else 0.0,
            pdr_instantaneous=0.0,          # no ROS data plane yet (see docstring)
            latency_avg_ms=0.0, throughput_kbps=0.0, queue_depth=0,
            cpu_time_us=0)
        if row is not None:
            out = String()
            out.data = json.dumps(row)
            self.pub_tel.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = MpcareAgentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
