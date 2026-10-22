#!/usr/bin/env bash
# Server-side proof of probe existence + positions (no GUI needed).
# Spawns the five probes into a headless gz sim, then asks the SIM what
# exists: odometry topic count + one odometry message per probe (x coordinate)
# + the SceneBroadcaster's scene/info (exactly what a GUI would render).
# NOTE: no `set -u` — ROS setup scripts reference unset vars.
source /opt/ros/lyrical/setup.bash
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd "$HOME/RoutingComm"

gz sim -s -r worlds/open_field.sdf > /tmp/probe_vis_gz.log 2>&1 &
GZPID=$!
sleep 6

for i in 0 1 2 3 4; do
  ros2 run ros_gz_sim create \
    -file worlds/models/uav_probe/model.sdf -name uav_probe_$i \
    -x "$(python3 -c "print($i*200.0)")" -y 0.0 -z 0.5 > /dev/null 2>&1
done
sleep 3

echo '=== 1) gz odometry topics (expect 5, one per probe) ==='
gz topic -l | grep odometry | sort
echo "count: $(gz topic -l | grep -c odometry)"

echo
echo '=== 2) one odom message per probe -> x position (expect 0/200/400/600/800) ==='
for i in 0 1 2 3 4; do
  x=$(timeout 5 gz topic -e -t "/model/uav_probe_$i/odometry" -n 1 2>/dev/null \
      | grep -A2 'position' | grep -m1 'x:' | head -1)
  echo "uav_probe_$i at $x"
done

echo
echo '=== 3) scene/info (what a GUI would render) -> probe models + poses ==='
timeout 8 gz topic -e -t /world/open_field/scene/info -n 1 2>/dev/null \
  | grep -B1 -A4 'name: "uav_probe' | grep -E 'name:|x:' | head -20
echo "probe models in scene: $(timeout 8 gz topic -e -t /world/open_field/scene/info -n 1 2>/dev/null | grep -c 'name: "uav_probe')"

kill $GZPID 2>/dev/null; sleep 1
pkill -f '[g]z sim' 2>/dev/null
exit 0
