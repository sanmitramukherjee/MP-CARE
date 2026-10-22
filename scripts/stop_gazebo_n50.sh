#!/usr/bin/env bash
# Graceful shutdown for the Gazebo N=50 gate:  bash scripts/stop_gazebo_n50.sh
# SIGINT to the launch process group (the logger flushes its CSV in
# destroy_node), wait, then sweep any survivors.
# (no `set -u`: ROS setup scripts reference unset vars)
PID=$(cat /tmp/gz50.pid 2>/dev/null)
if [ -n "${PID:-}" ] && kill -0 "$PID" 2>/dev/null; then
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  kill -INT -"$PGID" 2>/dev/null
  echo "SIGINT -> process group $PGID (pid $PID)"
  sleep 15
  kill -INT -"$PGID" 2>/dev/null
  sleep 8
fi
pkill -9 -f 'planner_[n]ode' 2>/dev/null
pkill -9 -f 'parameter_[b]ridge' 2>/dev/null
pkill -9 -f 'gz-sim-[m]ain' 2>/dev/null
pkill -9 -f 'gz [s]im' 2>/dev/null
echo stopped
