"""Explicit swarm network model (implementation_plan.md section 3.3).

Log-distance path loss (alpha=2.5, PL0=46.4 dB), PRR sigmoid from SNR,
per-link ETX = 1/(PRR_fwd * PRR_rev), drop-tail queue model, and injected
packet loss / latency. Parameters live in config/network_model.yaml.
"""
