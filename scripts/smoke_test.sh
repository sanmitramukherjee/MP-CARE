#!/usr/bin/env bash
# smoke_test.sh — Phase 1 "it works" checks (implementation_plan.md section 3.5).
#
# Headless MVP (runs anywhere with Python 3.11+, deterministic, no ROS needed):
#   ./scripts/smoke_test.sh
#
# Full ROS 2 stack (Ubuntu 22.04 + Humble + Gazebo Fortress):
#   colcon build --symlink-install && source install/setup.bash
#   ros2 launch launch/mpcare_sim.launch.py
set -euo pipefail
cd "$(dirname "$0")/.."

run_py() { python3 "$@"; }
if ! command -v python3 >/dev/null 2>&1; then
  if command -v py >/dev/null 2>&1; then
    run_py() { py -3.13 "$@"; }     # Windows py launcher
  else
    echo "smoke_test: no python3 / py launcher found" >&2
    exit 1
  fi
fi

echo "[smoke_test] headless Phase 1 benchmark (section 3.5, all six checks)"
run_py scripts/run_sim.py --seed 42
