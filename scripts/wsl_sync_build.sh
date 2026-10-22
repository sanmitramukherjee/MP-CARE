#!/usr/bin/env bash
# Sync the Windows checkout (visible in WSL at /mnt/c/...) into the WSL
# worktree ~/RoutingComm, then rebuild. Safe to re-run any time the Windows
# side changed. NOTE: no `set -u` — ROS setup scripts reference unset vars.
SRC="/mnt/c/Users/Lenovo/Desktop/Personal/RoutingComm"
DST="$HOME/RoutingComm"

[ -d "$SRC" ] || { echo "FATAL: $SRC not visible from WSL"; exit 1; }
source /opt/ros/lyrical/setup.bash || exit 1

mkdir -p "$DST"
echo "== syncing source trees (src launch config worlds scripts) =="
for d in src launch config worlds scripts; do
  mkdir -p "$DST/$d"
  cp -a "$SRC/$d/." "$DST/$d/"
done

cd "$DST"
echo "== colcon build =="
colcon build --symlink-install 2>&1 | tail -5
echo "== done; worktree at $DST =="
