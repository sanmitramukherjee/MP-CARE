# launch/ — ROS 2 Launch Files

Launch files for the full ROS 2 + Gazebo simulation stack. For headless
experiments (no ROS required), use `scripts/run_sim.py` instead.

---

## Files

### `mpcare_sim.launch.py`
Brings up the complete 5-UAV simulation stack:
- `network_model` node — channel emulation with configurable path-loss and MAC
  impairment.
- One `mpcare_agent` node per UAV (`uav_id` 0–4) — the MP-CARE routing agent.
- `metrics_logger` node — records per-trial CSV output tagged with `trial_id`.

Expects Gazebo to publish `/uav_i/odom` topics (one per UAV). Target platform
is Ubuntu 26.04 + ROS 2 Lyrical + Gazebo Jetty.

**Usage** (run from the repository root; parameter paths are cwd-relative):
```bash
ros2 launch launch/mpcare_sim.launch.py trial_id:=trial_001
```

---

> **Note:** The headless equivalent — requiring only Python and no ROS
> installation — is `scripts/run_sim.py`. All paper results were produced
> with the headless engine for reproducibility across platforms.
