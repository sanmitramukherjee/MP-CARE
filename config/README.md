# config/ — ROS 2 Parameter Files

Runtime configuration for the ROS 2 stack. The same default values are
mirrored in `src/network_model/network_model/model.py` (`NetworkParams`) so
that the headless simulation engine and the full ROS 2 stack remain in lockstep.

---

## Files

### `network_model.yaml`
Channel and MAC layer parameters:

| Parameter | Description |
|---|---|
| Path-loss exponent α | Controls signal attenuation with distance. |
| PL₀ | Reference path loss at 1 m. |
| P_tx | Transmit power (dBm). |
| Noise floor | Receiver noise floor (dBm). |
| PRR sigmoid (snr50, k) | Packet reception rate curve: 50%-PRR SNR and slope. |
| MAC retries | Maximum link-layer retransmission attempts. |
| Injected loss / latency | Artificial impairment for controlled experiment conditions. |
| Beacon interval | MP-CARE neighbour discovery beacon period (s). |
| Queue size | Per-link TX queue depth. |

### `mpcare_agent.yaml`
Per-agent parameters. `uav_id` is overridden per UAV in the launch file.

| Parameter | Description |
|---|---|
| `uav_id` | Agent identifier (0-indexed). |
| `speed` | Nominal UAV cruise speed (m/s). |
| `waypoint_tolerance` | Arrival radius for waypoint completion (m). |

### `logger.yaml`
Data collection parameters.

| Parameter | Description |
|---|---|
| `output_dir` | Directory for per-trial CSV output. |
| `trial_id` | Unique identifier appended to output filenames. |
| `logging_rate` | Minimum logging frequency (≥ 10 Hz per UAV). |
