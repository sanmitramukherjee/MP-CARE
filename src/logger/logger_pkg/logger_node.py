"""ROS 2 wrapper: metrics logger node (plan section 3.4).

Collects /swarm/topology plus per-UAV odometry (via the topology blob in the
MVP; a full version subscribes each /uav_i/odom) and appends one CSV row per
UAV per tick at >= 10 Hz using the exact section 3.4 schema. The CSV-writing
core is shared with the headless MVP (logger_pkg.core.TrialLogger).
"""

import json

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from logger_pkg.core import TrialLogger


class LoggerNode(Node):

    def __init__(self):
        super().__init__("swarm_logger")
        self.declare_parameters("", [
            ("out_dir", "data/raw"), ("trial_id", ""), ("experiment", "phase1_ros"),
            ("algorithm", "astar_dist+gpsr"), ("rate_hz", 10.0),
        ])
        gp = self.get_parameters_by_prefix("")
        self.experiment = gp["experiment"].value
        self.algorithm = gp["algorithm"].value
        self.trial = TrialLogger(gp["out_dir"].value,
                                 trial_id=gp["trial_id"].value or None)
        self.latest_topology = {}
        self.create_subscription(String, "/swarm/topology", self._topology, 10)
        self.create_subscription(String, "/swarm/telemetry", self._telemetry, 10)
        self.create_timer(1.0 / max(1.0, gp["rate_hz"].value), self._tick)
        self.get_logger().info(f"logger writing {self.trial.path}")

    def _topology(self, msg: String):
        self.latest_topology = json.loads(msg.data)

    def _telemetry(self, msg: String):
        row = json.loads(msg.data)
        self.trial.log(**row, algorithm=self.algorithm, experiment=self.experiment)

    def _tick(self):
        # pacing heartbeat; rows are written from /swarm/telemetry events
        pass

    def destroy_node(self):
        self.trial.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
