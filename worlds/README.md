# worlds/ — Gazebo Simulation Worlds

Gazebo SDF world files used for the Gazebo-stack validation phase of MP-CARE.
The headless simulation engine (`scripts/run_sim.py`) does not require these
files; they are used only when running the full ROS 2 + Gazebo stack.

---

## World Files

### `open_field.sdf`
Obstacle-free 1000 m × 1000 m flat field. Used for the initial 5-UAV scaffold
validation (hover, waypoint tracking, 3-hop chain tests). Targets Gazebo Jetty
on Ubuntu 26.04 + ROS 2 Lyrical; SDF format version 1.8.

### `urban_canyon.sdf`
Obstacle-laden urban canyon world for stress-test experiments (Phase 2+).
Introduces RF shadowing and forced topology changes to validate the
event-triggered re-planning component of MP-CARE.

---

## UAV Probe Model — `models/uav_probe/`

A static instrumented UAV model for pipeline validation. Five probes arranged
in a 200 m-spaced row reproduce the 4-hop chain geometry with fully controlled
positioning, allowing end-to-end validation of the
spawn → bridge → odometry → network-model → MP-CARE → logger pipeline
without requiring a flying vehicle.

**Design notes:**
- Physics body: 0.2 m sphere.
- Visual shell: 4 m translucent marker, added so individual probes are
  distinguishable in the Gazebo GUI at 200 m separation.
- Publishes ground-truth odometry via `OdometryPublisher`; does not simulate
  flight dynamics.
