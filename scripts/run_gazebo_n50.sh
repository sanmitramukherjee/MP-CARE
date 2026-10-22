#!/usr/bin/env bash
# Gazebo N=50 gate launcher (WSL side):  bash scripts/run_gazebo_n50.sh
# SERVER-FIRST flow (evidence: `ros_gz_sim create` stalls against a cold
# server — 27+ spawns hang, zero telemetry; run1 against an already-serving
# world streamed 13.6k rows). 1) clean leftovers 2) start the bare server
# 3) wait for it to serve 4) launch the ROS stack with gz_external:=true.
# Writes the launcher PID to /tmp/gz50.pid for stop_gazebo_n50.sh.
# (no `set -u`: ROS setup scripts reference unset vars — see wsl_sync_build.sh)
cd "$HOME/RoutingComm" || exit 1

pkill -9 -f 'ros2 [l]aunch' 2>/dev/null
pkill -9 -f 'planner_[n]ode' 2>/dev/null
pkill -9 -f 'parameter_[b]ridge' 2>/dev/null
pkill -9 -f 'gz-sim-[m]ain' 2>/dev/null
pkill -9 -f 'gz [s]im' 2>/dev/null
sleep 2

# ---- 1) the bare headless server ----
source /opt/ros/lyrical/setup.bash
LOG=data/raw/phase6_dev/gz50_server.log
setsid nohup gz sim -r -s -v 1 worlds/scale_field.sdf > "$LOG" 2>&1 < /dev/null &
echo $! > /tmp/gz50_server.pid

# ---- 2) wait until the world is SERVING (functional test: the gz CLI
# answers a service-list request; the wrapper's own log stays empty for a
# while — measured) ----
serving=0
for i in $(seq 1 30); do
  if timeout 5 gz service --list >/dev/null 2>&1; then serving=1; break; fi
  sleep 1
done
[ "$serving" = 1 ] || { echo "SERVER DID NOT COME UP"; exit 1; }
echo "gz server serving (pid $(cat /tmp/gz50_server.pid))"

# ---- 3) the ROS stack (spawns + bridges + agents + logger) ----
source install/setup.bash
export RCUTILS_LOGGING_BUFFERED_STREAM=0
LLOG=data/raw/phase6_dev/gazebo_n50_launch.log
rm -f data/raw/gazebo_n50.csv "$LLOG"
setsid nohup ros2 launch launch/mpcare_sim.launch.py \
  use_gazebo:=true gz_external:=true n_uav:=50 world:=scale_field \
  spawn_csv:=data/raw/phase6_dev/gazebo_n50_positions.csv \
  trial_id:=gazebo_n50 out_dir:=data/raw \
  > "$LLOG" 2>&1 < /dev/null &
echo $! > /tmp/gz50.pid
echo "launcher pid $(cat /tmp/gz50.pid); log: $LLOG"
