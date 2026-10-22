#!/usr/bin/env bash
# Debug: why does /uav_0/route stay silent in Tier 1?
source /opt/ros/lyrical/setup.bash
source "$HOME/RoutingComm/install/setup.bash"
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd "$HOME/RoutingComm"

ros2 launch launch/mpcare_sim.launch.py trial_id:=dbg > /tmp/dbg_launch.log 2>&1 &
sleep 12
for i in 0 1 2 3 4; do
  ros2 topic pub -r 10 "/uav_$i/odom" nav_msgs/msg/Odometry \
    "{pose: {pose: {position: {x: $((i*200)), y: 0.0, z: 40.0}}}}" \
    > /dev/null 2>&1 &
done
sleep 15

echo '=== UAV-0 telemetry rows (neighbors / hops / etx) ==='
timeout 8 ros2 topic echo /swarm/telemetry 2>/dev/null | grep -o '"uav_id": 0[^}]*"route_hop_count": [0-9]*' | head -2
timeout 8 ros2 topic echo /swarm/telemetry 2>/dev/null | grep -o '"uav_id": 1[^}]*"neighbors": \[[^]]*\]' | head -2

echo '=== route topics that exist ==='
ros2 topic list 2>/dev/null | grep route

echo '=== each route topic, 4 s listen ==='
for i in 0 1 2 3 4; do
  printf "uav_%s: " "$i"
  timeout 4 ros2 topic echo "/uav_$i/route" --once 2>/dev/null | tr -d '\n' | head -c 120
  echo
done

echo '=== launch log: warnings/errors from agents ==='
grep -E "WARN|ERROR" /tmp/dbg_launch.log | head -6

jobs -p | xargs -r kill 2>/dev/null; sleep 1; pkill -f ros2 2>/dev/null
exit 0
