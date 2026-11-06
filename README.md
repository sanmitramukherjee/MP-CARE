# MP-CARE: Simulation Modeling, Verification & Mobility-Predictive Routing for Aerial Swarms

Official reference implementation, simulation framework, and experimental artifact repository for **MP-CARE** (*Simulation Modelling Practice and Theory* submission, Elsevier).

**Author:** Sanmitra Mukherjee (Independent Researcher)  
**Email:** [sanmitra.mukherjee15@gmail.com](mailto:sanmitra.mukherjee15@gmail.com)  
**Repository:** [https://github.com/sanmitramukherjee/MP-CARE](https://github.com/sanmitramukherjee/MP-CARE)  
**Preprint (SSRN):** [https://doi.org/10.2139/ssrn.7647778](https://doi.org/10.2139/ssrn.7647778)

---

## Overview

Simulation is the primary evidence base for Flying Ad-Hoc Network (FANET) routing, yet protocol rankings are traditionally reported under single channel and topology abstractions whose structural influence remains unquantified.

This repository provides a verified, tick-level simulation and modeling framework designed for high-speed multi-UAV swarms:
- **Kinematic & Aerial Motion:** 3D kinematics operating across 5–50 m/s speeds in diamond, crossing-patrol, grid, and random geometric topologies.
- **Physical Channel Realism:** Distance-dependent air-to-air link budgets with temporally correlated log-normal shadow fading ($\sigma \in \{0, 2, 4\}\text{ dB}$, $\tau = 5\text{ s}$) and logistic packet reception curves.
- **MAC Contention Modeling:** IEEE 802.11-style retry MAC and a frame-level CSMA/CA Distributed Coordination Function (DCF) scheduler evaluating carrier-sense ranges from 600 m down to 150 m.
- **Routing Baselines:** Evaluated against 8 distinct protocol models, including classic benchmarks (AODV, OLSR, GPSR, Distance-A*), predictive proactive extensions (LET-OLSR), modern reinforcement learning (QLGR: Q-Learning Geographic Routing), and complete-information ETX oracles.

```
       +--------------------+      +------------------+      +------------------+      +------------------+
       |   Neighbor Table   | ---> |  Link Predictor  | ---> |   Cost Engine    | ---> |  Event Trigger   |
       |  (10 Hz beacons)   |      | (ETX_pred, LET)  |      | (5-term min-max) |      | (T1-T5 triggers) |
       +--------------------+      +------------------+      +------------------+      +------------------+
                 ^                                                                               |
                 |               Re-route feedback (delivery outcomes -> T1 evidence)            |
                 +-------------------------------------------------------------------------------+
```

---

## Empirical Benchmark & Model Sensitivity Highlights

All reported metrics are measured across 30 independent seeds per condition with non-parametric Mann–Whitney $U$ tests, Holm–Bonferroni corrections, and rank-biserial effect sizes.

| Evaluation Regime | Baseline(s) | MP-CARE Performance | Margin / Key Scientific Finding |
| :--- | :--- | :--- | :--- |
| **Diamond Benchmark (10% Loss, $\sigma=0$)** | Distance-A* (0.583), GPSR (0.583), AODV (0.603), OLSR (0.565) | **0.729 PDR** ($\pm 0.025$) | **$+14.6$ to $+20.6$ pp PDR** ($p < 10^{-5}$, rank-biserial $0.88$) |
| **Channel Heterogeneity ($\sigma=0 \to 4\text{ dB}$)** | GPSR (Patrol lanes, 10 UAVs, 20 m/s) | **0.968 PDR** (at 4 dB) | Margin expands from **$+6.5$ pp** ($\sigma=0$) to **$+30.8$ pp** ($\sigma=4\text{ dB}$) ($p < 10^{-5}$) |
| **Shared Airtime Contention (802.11 DCF)** | GPSR, AODV, OLSR, QLGR (55% airtime load) | **Top-2 ranking** across all loaded patrol cells | Distance margins remain positive and significant (**$+6.8$ to $+29.3$ pp**, $p \le 0.0032$) |
| **Carrier-Sense Sensitivity ($600 \to 250\text{ m}$)** | 8 Protocol models across 33,600 runs | **Bounded null:** top-4 in every cell | Ranking survives collision growth; breaks at $200\text{ m}$ (sub-decode stress regime) |
| **Swarm Scale ($N=50$ UAVs)** | AODV (0.875), GPSR (0.861), OLSR (0.857) | **0.939 PDR** ($\pm 0.076$) | **$+6.4$ to $+8.2$ pp PDR** ($p \le 0.0002$); protocol compute: $0.082\text{ ms/node/tick}$ |
| **High-Speed Aerial Dynamics (50 m/s)** | GPSR, AODV, OLSR, QLGR | **Robust link tracking** | Validated up to 50 m/s fixed-wing velocity profiles |
| **Task Allocation (CBBA Extension)** | Loss-naive CBBA (0–60% channel loss) | **$+40$ to $+65$ pp task completion** | **$6.3\times$ lower route cost** ($p \le 0.0073$) |
| **Collision Avoidance (ORCA Extension)**| Standard ORCA (69.5% link retention) | **100.0% link retention** | **$+43.9\%$ relative retention** ($p = 1.3 \times 10^{-4}$), zero collisions |

---

## Repository Structure

```
.
├── src/                    # ROS 2 packages, network models & core routing engines
│   ├── mpcare_agent/       # MP-CARE router, cost engine, link predictor, event trigger
│   ├── network_model/      # Physical channel, log-distance PRR, shadow fading & 802.11 DCF
│   ├── qlgr_routing/       # Q-Learning Geographic Routing baseline
│   ├── olsr_routing/       # RFC 3626 OLSR & LET-OLSR baselines
│   ├── aodv_routing/       # RFC 3561 AODV routing baseline
│   ├── gpsr_forwarding/    # GPSR greedy + perimeter forwarding baseline
│   ├── cbba_allocator/     # Communication-aware CBBA task allocation node
│   ├── orca_avoidance/     # Connectivity-aware ORCA collision avoidance node
│   └── logger/             # 27-column high-rate telemetry logging system
├── scripts/                # Headless discrete-event simulation & benchmark scripts
│   ├── run_sim.py          # Primary headless simulation engine (0.1s tick)
│   ├── run_sweeps.py       # Loss, mobility, and node-failure evaluation campaigns
│   ├── run_dcf.py          # 802.11 DCF CSMA/CA contention & carrier-sense sweeps
│   ├── run_phase8.py       # Patrol topology & QLGR learning baseline sweeps
│   ├── run_phase9.py       # Log-normal shadow fading evaluation campaigns
│   ├── run_phase10.py      # Spatially uniform vs heterogeneous loss shift sweeps
│   ├── run_phase5.py       # ORCA (Ext. A) and CBBA (Ext. B) evaluation campaigns
│   ├── run_phase6.py       # N=5 to N=50 swarm scalability campaigns
│   ├── run_phase7.py       # Ablation study (A1-A7) and weight sensitivity Pareto front
│   ├── summarize_cs_sweep.py # Carrier-sense boundary analysis
│   └── stats_utils.py      # Mann-Whitney U, Holm-Bonferroni, Kruskal-Wallis implementations
├── tests/                  # Unit and verification test suite (113 pytest checks)
├── launch/                 # ROS 2 launch files (mpcare_sim.launch.py)
├── config/                 # ROS 2 YAML configuration files
└── worlds/                 # Gazebo SDF simulation world files
```

> **Data & Manuscript Note:** Per-trial telemetry CSVs and aggregated summary JSONs are openly deposited on Zenodo with DOI cited in the manuscript. All paper source files and versioned PDFs reside in `paper/`.

---

## Quickstart & Installation

### Requirements
* **Python**: Python 3.9+ (tested on Python 3.9, 3.11, 3.13 across Windows & Linux/WSL2)
* **Python Dependencies**: `pytest`, `matplotlib`, `numpy`, `scipy`
* **ROS 2 & Gazebo (Optional for Middleware Validation)**: Ubuntu 24.04/26.04 LTS + ROS 2 + Gazebo

### Setup
```bash
# Clone the repository
git clone https://github.com/sanmitramukherjee/MP-CARE.git
cd MP-CARE

# Install required dependencies
pip install pytest matplotlib numpy scipy
```

---

## Verification & Campaign Reproduction

### 1. Run Verification Test Suite (113 Checks)
Validates log-distance propagation, shadow fading correlation, 802.11 DCF backoff slots, routing engines, and ROS 2 message schemas:
```bash
pytest
```

### 2. Run Headless Simulation Smoke Test
```bash
python scripts/run_sim.py
```

### 3. Run Benchmark Campaigns
```bash
# 802.11 DCF Contention & Carrier-Sense Sweeps (600 m down to 150 m)
python scripts/run_dcf.py

# Correlated Log-Normal Shadow Fading Campaigns (sigma = 0, 2, 4 dB)
python scripts/run_phase9.py

# Crossing-Patrol Topology & QLGR Sweeps
python scripts/run_phase8.py

# Loss-Shift Analysis (Uniform vs Heterogeneous Loss)
python scripts/run_phase10.py

# Core Loss, Mobility (up to 50 m/s), and Relay Failure Sweeps
python scripts/run_sweeps.py

# Swarm Scalability Sweeps (N=5 to N=50)
python scripts/run_phase6.py

# Ablation Study & Pareto Weight Analysis
python scripts/run_phase7.py
```

---

## Citation

```bibtex
@article{mukherjee2026modelfidelity,
  author    = {Mukherjee, Sanmitra},
  title     = {Model fidelity and protocol ranking in aerial ad-hoc network simulation: a verified, reproducible study of mobility-predictive routing},
  journal   = {Simulation Modelling Practice and Theory},
  year      = {2026},
  note      = {Under review. Preliminary preprint at SSRN: \url{https://doi.org/10.2139/ssrn.7647778}}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
