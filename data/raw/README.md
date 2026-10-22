# data/raw/ — Raw Empirical Datasets

This directory contains per-trial telemetry CSVs and raw execution logs for the MP-CARE simulation campaigns and live ROS 2 / Gazebo middleware runs.

---

## File Registry & Schema

Per-trial CSV files follow the naming convention:
`<experiment>__<algorithm>__seed<seed>__trial<trial>.csv`

Each row follows a standardized 27-field telemetry schema recording per-tick platform state, neighbor tables, routing choices, link metrics, and event trigger outputs.

### Live ROS 2 & Gazebo Validation Datasets

| File | Rows | Description |
| :--- | :--- | :--- |
| `ubuntu_tier1_wsl.csv` | 872 | WSL2 ROS 2 validation, Tier 1 (odometry bridge) |
| `ubuntu_tier3_wsl.csv` | 3,565 | WSL2 ROS 2 validation, Tier 3 (probe kill window) |
| `ubuntu_tier1.csv` | 7,597 | Native dual-boot ROS 2 validation, Tier 1 |
| `ubuntu_tier3_run1.csv` | 10,747 | Native dual-boot ROS 2 validation, Tier 3 (relay kill window) |
| `ubuntu_tier1_conf.csv` | 874 | Post-fix verification run, Tier 1 |
| `ubuntu_tier3_conf.csv` | 6,223 | Post-fix verification run, Tier 3 |

All native datasets are validated by `scripts/validate_native_csvs.py`.
