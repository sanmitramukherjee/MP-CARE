#!/usr/bin/env bash
# SOP NOTES/testROS2.md Tier 3 (WSL run, 2026-09): full stack — gz sim +
# five static uav_probe spawns + bridges + MP-CARE side, then the kill
# perturbation. Gate G5: links drop <= 1 s, routes 0/1 go silent.
# NOTE: no `set -u` — the ROS setup scripts reference unset vars.
source /opt/ros/lyrical/setup.bash
source "$HOME/RoutingComm/install/setup.bash"
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd "$HOME/RoutingComm"

ros2 launch launch/mpcare_sim.launch.py use_gazebo:=true \
    trial_id:=ubuntu_tier3_wsl > /tmp/tier3_launch.log 2>&1 &
LPID=$!
sleep 20

echo '=== odom topics alive (5 probes bridged) ==='
for i in 0 1 2 3 4; do
  timeout 5 ros2 topic hz "/uav_$i/odom" 2>/dev/null | head -1
done

echo '=== /uav_0/route before kill ==='
timeout 8 ros2 topic echo /uav_0/route --once 2>/dev/null | head -c 200; echo

echo '=== /swarm/topology before kill (first 300 chars) ==='
timeout 8 ros2 topic echo /swarm/topology --once 2>/dev/null | head -c 300; echo

echo '=== KILL uav_probe_2 at t=now ==='
# NOTE: gz.msgs.Entity.type is an ENUM — use the NAME (MODEL), not a number.
# type: 1 = WORLD (silently fails with 'not found'); verified empirically.
timeout 10 gz service -s /world/open_field/remove \
  --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 3000 \
  --req 'name: "uav_probe_2", type: MODEL' 2>&1 | head -2
echo '=== models still present (expect probes 0,1,3,4) ==='
timeout 8 gz model -l 2>/dev/null | head -8
sleep 6

echo '=== /swarm/topology after kill: UAV-2 links must be gone ==='
timeout 8 ros2 topic echo /swarm/topology --once 2>/dev/null \
  | grep -cE '\[[0-9]+, 2,|\[2, [0-9]+,' || true
echo '(count of link rows mentioning node 2 above; expect 0)'

echo '=== /uav_0/route after kill: must be SILENT (timeout = correct) ==='
timeout 6 ros2 topic echo /uav_0/route --once 2>/dev/null | head -c 200
echo "(empty above = no route across the partition: CORRECT)"
echo '=== /uav_3/route after kill (3-4 island still routes) ==='
timeout 6 ros2 topic echo /uav_3/route --once 2>/dev/null | head -c 200; echo

echo '=== launch log: errors? ==='
grep -icE "traceback|unhandled" /tmp/tier3_launch.log || true
tail -3 /tmp/tier3_launch.log

CSV=$(ls -t data/raw/ubuntu_tier3_wsl*.csv 2>/dev/null | head -1)
echo "=== logger CSV: $CSV ==="
[ -n "$CSV" ] && wc -l "$CSV"

jobs -p | xargs -r kill 2>/dev/null
sleep 1
pkill -f "ros2" 2>/dev/null
pkill -f "gz sim" 2>/dev/null
exit 0
