#!/usr/bin/env bash
# Start the BARE headless Gazebo server first (N=50 gate, server-first flow):
#   bash scripts/gz50_server.sh
# Evidence (run1 vs runs 3-5): `ros_gz_sim create` STALLS against a cold
# server (27+ spawns hang, zero telemetry); against an already-serving
# world it completes and telemetry streams. The launch file skips its own
# server when gz_external:=true.
cd "$HOME/RoutingComm" || exit 1
source /opt/ros/lyrical/setup.bash
pkill -9 -f 'gz-sim-[m]ain' 2>/dev/null
pkill -9 -f 'gz [s]im' 2>/dev/null
sleep 1
LOG=data/raw/phase6_dev/gz50_server.log
setsid nohup gz sim -r -s -v 1 worlds/scale_field.sdf > "$LOG" 2>&1 < /dev/null &
echo $! > /tmp/gz50_server.pid
echo "gz server pid $(cat /tmp/gz50_server.pid); log: $LOG"
echo "wait for '[dbg] [Setup] Server configured' in the log before launching"
