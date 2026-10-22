#!/usr/bin/env bash
# Debug: make the entity-remove work. Spawn one probe, then try remove
# variants; print model listing between steps.
source /opt/ros/lyrical/setup.bash
source "$HOME/RoutingComm/install/setup.bash"
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd "$HOME/RoutingComm"
export GZ_SIM_RESOURCE_PATH="$HOME/RoutingComm/worlds/models:$GZ_SIM_RESOURCE_PATH"

gz sim -r -s worlds/open_field.sdf > /tmp/dbg2_gz.log 2>&1 &
sleep 14

echo '=== spawn probe_2 via ros_gz_sim create ==='
timeout 20 ros2 run ros_gz_sim create \
  -file worlds/models/uav_probe/model.sdf \
  -name uav_probe_2 -x 400.0 -y 0.0 -z 0.5 2>&1 | head -2
sleep 3

echo '=== gz topics mentioning probe_2 ==='
timeout 8 gz topic -l 2>/dev/null | grep probe_2 | head -3

echo '=== remove variant A: name + type 1 ==='
timeout 10 gz service -s /world/open_field/remove \
  --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 3000 \
  --req 'name: "uav_probe_2", type: 1' 2>&1 | head -2
echo '=== probe_2 topics after A ==='
timeout 6 gz topic -l 2>/dev/null | grep -c probe_2

echo '=== respawn ==='
timeout 20 ros2 run ros_gz_sim create \
  -file worlds/models/uav_probe/model.sdf \
  -name uav_probe_2 -x 400.0 -y 0.0 -z 0.5 2>&1 | head -1
sleep 3
echo '=== remove variant B: name + type MODEL (enum name) ==='
timeout 10 gz service -s /world/open_field/remove \
  --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 3000 \
  --req 'name: "uav_probe_2", type: MODEL' 2>&1 | head -2
echo '=== probe_2 topics after B ==='
timeout 6 gz topic -l 2>/dev/null | grep -c probe_2

echo '=== respawn again ==='
timeout 20 ros2 run ros_gz_sim create \
  -file worlds/models/uav_probe/model.sdf \
  -name uav_probe_2 -x 400.0 -y 0.0 -z 0.5 2>&1 | head -1
sleep 3
echo '=== remove variant C: name only ==='
timeout 10 gz service -s /world/open_field/remove \
  --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 3000 \
  --req 'name: "uav_probe_2"' 2>&1 | head -2
echo '=== probe_2 topics after C ==='
timeout 6 gz topic -l 2>/dev/null | grep -c probe_2

echo '=== gz log errors ==='
grep -iE "error|not found" /tmp/dbg2_gz.log | tail -3

pkill -f "gz sim" 2>/dev/null
exit 0
