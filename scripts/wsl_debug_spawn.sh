#!/usr/bin/env bash
# Debug: why did the uav_probe spawns fail in Tier 3?
source /opt/ros/lyrical/setup.bash
source "$HOME/RoutingComm/install/setup.bash"
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd "$HOME/RoutingComm"
export GZ_SIM_RESOURCE_PATH="$HOME/RoutingComm/worlds/models:$GZ_SIM_RESOURCE_PATH"

gz sim -r -s worlds/open_field.sdf > /tmp/dbg_gz.log 2>&1 &
GZPID=$!
sleep 15

echo '=== gz sim log ==='
head -6 /tmp/dbg_gz.log

echo '=== world services (create/remove present?) ==='
timeout 8 gz service -l 2>/dev/null | grep -E "create|remove" | head -4

echo '=== models currently in the world ==='
timeout 8 gz model -l 2>/dev/null | head -8

echo '=== manual spawn replay (raw output) ==='
timeout 15 gz service -s /world/open_field/create \
  --reqtype gz.msgs.EntityFactory --reptype gz.msgs.Boolean --timeout 5000 \
  --req "name: 'uav_probe_0', allow_renaming: false, pose: {position: {x: 0.0, y: 0.0, z: 0.5}}, file: 'uav_probe'" 2>&1 | head -3

echo '=== models after spawn attempt ==='
timeout 8 gz model -l 2>/dev/null | head -8

echo '=== gz log after spawn attempt (errors?) ==='
grep -iE "error|fail|uav_probe" /tmp/dbg_gz.log | head -6

echo '=== gz topics (odometry?) ==='
timeout 8 gz topic -l 2>/dev/null | grep -i odom | head -4

kill $GZPID 2>/dev/null
sleep 1
pkill -f "gz sim" 2>/dev/null
exit 0
