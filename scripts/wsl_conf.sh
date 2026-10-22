#!/usr/bin/env bash
# WSL2 confirmation run (default ROS test environment decision, 2026-09-29).
# Same gates as scripts/wsl_tier{1,2,3}.sh but with fresh trial ids
# (*_conf) so nothing overwrites archived evidence, and CSVs are copied
# straight back to the Windows repo at the end.
# Usage: bash scripts/wsl_conf.sh tier1|tier2|tier3
# NOTE: no `set -u` — ROS setup scripts reference unset vars.
source /opt/ros/lyrical/setup.bash
source "$HOME/RoutingComm/install/setup.bash"
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd "$HOME/RoutingComm"
STAGE="${1:?usage: wsl_conf.sh tier1|tier2|tier3}"

# Windows repo, for artifact copy-back
WIN="/mnt/c/Users/Lenovo/Desktop/Personal/RoutingComm/data/raw"

case "$STAGE" in
tier1)
  ros2 launch launch/mpcare_sim.launch.py trial_id:=ubuntu_tier1_conf \
      > /tmp/conf_tier1.log 2>&1 &
  sleep 12
  for i in 0 1 2 3 4; do
    ros2 topic pub -r 10 "/uav_$i/odom" nav_msgs/msg/Odometry \
      "{pose: {pose: {position: {x: $((i*200)), y: 0.0, z: 40.0}}}}" \
      > /dev/null 2>&1 &
  done
  sleep 18   # beacons + route settle
  echo '=== /uav_0/route ==='
  timeout 8 ros2 topic echo /uav_0/route --once 2>/dev/null | head -c 200; echo
  echo '=== /swarm/topology (first 300 chars) ==='
  timeout 8 ros2 topic echo /swarm/topology --once 2>/dev/null | head -c 300; echo
  echo '=== rate of /swarm/topology (expect ~1 Hz) ==='
  timeout 6 ros2 topic hz /swarm/topology 2>/dev/null | head -1
  echo '=== launch log error scan ==='
  grep -icE "traceback|process has died" /tmp/conf_tier1.log || true
  echo '=== CSV ==='
  wc -l data/raw/ubuntu_tier1_conf.csv 2>/dev/null || echo "CSV MISSING"
  jobs -p | xargs -r kill 2>/dev/null; sleep 1; pkill -f "ros2 topic pub" 2>/dev/null
  pkill -f "ros2 launch" 2>/dev/null
  ;;

tier2)
  echo '=== gz sim server-only, 15 s, scan for plugin errors ==='
  timeout 15 gz sim -s -r worlds/open_field.sdf > /tmp/conf_tier2.log 2>&1
  grep -m3 "Server running" /tmp/conf_tier2.log || echo "no 'Server running' line"
  grep -icE "error|failed to load" /tmp/conf_tier2.log || true
  ;;

tier3)
  ros2 launch launch/mpcare_sim.launch.py use_gazebo:=true \
      trial_id:=ubuntu_tier3_conf > /tmp/conf_tier3.log 2>&1 &
  sleep 40   # honest settle: gz boot + 5 spawns + bridges + beacons
  echo '=== odom rates (5 probes bridged) ==='
  for i in 0 1 2 3 4; do
    timeout 5 ros2 topic hz "/uav_$i/odom" 2>/dev/null | head -1
  done
  echo '=== /uav_0/route before kill ==='
  timeout 8 ros2 topic echo /uav_0/route --once 2>/dev/null | head -c 200; echo
  echo '=== KILL uav_probe_2 (type MODEL — enum name, not number) ==='
  timeout 10 gz service -s /world/open_field/remove \
    --reqtype gz.msgs.Entity --reptype gz.msgs.Boolean --timeout 3000 \
    --req 'name: "uav_probe_2", type: MODEL' 2>&1 | head -2
  sleep 6
  echo '=== /swarm/topology: link rows touching node 2 (expect 0) ==='
  timeout 8 ros2 topic echo /swarm/topology --once 2>/dev/null \
    | grep -cE '\[[0-9]+, 2,|\[2, [0-9]+,' || true
  echo '=== /uav_0/route after kill: empty = honest no-route (CORRECT) ==='
  timeout 6 ros2 topic echo /uav_0/route --once 2>/dev/null | head -c 200
  echo '(end)'
  echo '=== /uav_3/route after kill (3-4 island) ==='
  timeout 6 ros2 topic echo /uav_3/route --once 2>/dev/null | head -c 200; echo
  sleep 45   # post-kill recording window so the CSV carries the full story
  echo '=== launch log error scan ==='
  grep -icE "traceback|process has died" /tmp/conf_tier3.log || true
  echo '=== CSV ==='
  wc -l data/raw/ubuntu_tier3_conf.csv 2>/dev/null || echo "CSV MISSING"
  jobs -p | xargs -r kill 2>/dev/null; sleep 1
  pkill -f "ros2 launch" 2>/dev/null
  pkill -f "gz sim" 2>/dev/null
  echo '=== copy artifacts to Windows repo ==='
  for f in ubuntu_tier1_conf.csv ubuntu_tier3_conf.csv; do
    [ -f "data/raw/$f" ] && cp "data/raw/$f" "$WIN/$f" && echo "  -> $WIN/$f"
  done
  ;;
esac
exit 0
