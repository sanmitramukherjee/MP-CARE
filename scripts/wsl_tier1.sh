#!/usr/bin/env bash
# SOP NOTES/testROS2.md Tier 1 (WSL run, 2026-09): pipeline validation with
# hand-injected odometry, no Gazebo. Run from anywhere; it cds to the repo.
# NOTE: no `set -u` — the ROS setup scripts reference unset vars.
source /opt/ros/lyrical/setup.bash
source "$HOME/RoutingComm/install/setup.bash"
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd "$HOME/RoutingComm"

ros2 launch launch/mpcare_sim.launch.py trial_id:=ubuntu_tier1_wsl \
    > /tmp/tier1_launch.log 2>&1 &
LPID=$!
sleep 12

for i in 0 1 2 3 4; do
  ros2 topic pub -r 10 "/uav_$i/odom" nav_msgs/msg/Odometry \
    "{pose: {pose: {position: {x: $((i*200)), y: 0.0, z: 40.0}}}}" \
    > /dev/null 2>&1 &
done
sleep 18   # beacons + route settle

echo '=== /swarm/topology (one message) ==='
timeout 8 ros2 topic echo /swarm/topology --once 2>/dev/null | head -c 700; echo
echo '=== /uav_0/route (one message) ==='
timeout 8 ros2 topic echo /uav_0/route --once 2>/dev/null | head -c 300; echo
echo '=== /swarm/telemetry (one message) ==='
timeout 8 ros2 topic echo /swarm/telemetry --once 2>/dev/null | head -c 500; echo
echo '=== rates ==='
timeout 6 ros2 topic hz /swarm/topology 2>/dev/null | head -1
timeout 6 ros2 topic hz /swarm/telemetry 2>/dev/null | head -1

echo '=== logger CSV ==='
CSV=$(ls -t data/raw/ubuntu_tier1_wsl*.csv 2>/dev/null | head -1)
echo "file: $CSV"
if [ -n "$CSV" ]; then wc -l "$CSV"; head -1 "$CSV" | cut -c1-160; fi

echo '=== launch log tail ==='
tail -4 /tmp/tier1_launch.log

jobs -p | xargs -r kill 2>/dev/null
sleep 1
pkill -f "ros2" 2>/dev/null
exit 0
