"""Swarm metrics logger.

Aggregates per-UAV telemetry (odom, route cost, neighbors, network metrics)
and writes one CSV per trial using the section 3.4 logging schema, at
>= 10 Hz per UAV.
"""
