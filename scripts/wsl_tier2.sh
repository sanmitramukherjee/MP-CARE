#!/usr/bin/env bash
# SOP NOTES/testROS2.md Tier 2 (WSL run, 2026-09): Gazebo Jetty loads the
# world headlessly. Server-only (-s) so no GUI is needed in WSL.
# NOTE: no `set -u` — the ROS setup scripts reference unset vars.
source /opt/ros/lyrical/setup.bash
source "$HOME/RoutingComm/install/setup.bash"
export ROS_DOMAIN_ID=77
export ROS_LOCALHOST_ONLY=1
cd "$HOME/RoutingComm"

echo "=== gz version ==="
gz sim --version 2>&1 | head -2

echo "=== headless world load (25 s window) ==="
timeout 25 gz sim -s -r worlds/open_field.sdf > /tmp/tier2_gz.log 2>&1
echo "exit=$? (124 = ran until timeout, i.e. stayed up: good)"
echo "--- log: errors/warnings ---"
grep -iE "error|failed|cannot" /tmp/tier2_gz.log | grep -viE "no error|0 error" | head -5
echo "--- log: key lines ---"
grep -iE "server running|loaded|world" /tmp/tier2_gz.log | head -4
exit 0
