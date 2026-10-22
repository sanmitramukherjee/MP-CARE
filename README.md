# MP-CARE: Mobility-Predictive, Event-Triggered Communication-Aware Routing for Multi-UAV Swarms

Official reference implementation and simulation framework for **MP-CARE** (*Ad Hoc Networks* manuscript submission, Elsevier).

**Author:** Sanmitra Mukherjee (Sole Author)  
**Email:** [sanmitra.mukherjee15@gmail.com](mailto:sanmitra.mukherjee15@gmail.com)  
**Repository:** [https://github.com/sanmitramukherjee/MP-CARE](https://github.com/sanmitramukherjee/MP-CARE)  
**Preprint (SSRN):** [https://doi.org/10.2139/ssrn.7647778](https://doi.org/10.2139/ssrn.7647778)

---

## Abstract

Autonomous multi-Unmanned Aerial Vehicle (UAV) swarms operating in dynamic airborne environments require robust wireless communication for flight coordination and payload delivery. Traditional flight path planners minimize distance while treating the network as a static disk model, whereas standard Flying Ad-Hoc Network (FANET) routing protocols treat UAV motion as an exogenous disturbance and cannot inform flight trajectories when links degrade. 

**MP-CARE** is a decentralized, closed-loop routing framework that tightly couples flight motion with link-quality dynamics. MP-CARE scores candidate routes with a five-term composite cost engine (path length, predicted ETX, predicted latency, queue congestion, and hop count) over two-hop candidate expansions, and augments 1-hop beaconing with relative radial-velocity prediction of link quality and Link Expiration Time (LET) estimation. Re-routing is governed by a five-condition, edge-triggered hysteresis manager with a formal dwell-time stability bound.

```
       +--------------------+      +------------------+      +------------------+      +------------------+
       |   Neighbor Table   | ---> |  Link Predictor  | ---> |   Cost Engine    | ---> |  Event Trigger   |
       |  (1 Hz beacons B_i) |      | (ETX_pred, LET)  |      | (5-term min-max) |      |  (T1-T5 triggers)|
       +--------------------+      +------------------+      +------------------+      +------------------+
                 ^                                                                               |
                 |               Re-route feedback (delivery outcomes -> T1 evidence)            |
                 +-------------------------------------------------------------------------------+
```

---

## Empirical Benchmark Highlights

| Experiment / Campaign | Baseline(s) | MP-CARE Performance | Margin / Statistical Significance |
| :--- | :--- | :--- | :--- |
| **Diamond Topology (10% Loss)** | A* (0.583), GPSR (0.583), AODV (0.603), OLSR (0.565) | **0.729 PDR** ($\pm 0.025$) | **$+14.6$ to $+20.6$ pp PDR** ($p < 10^{-5}$, rank-biserial $0.88$) |
| **Mid-Flight Relay Failure** | AODV (0.695 PDR) | **0.992 PDR** ($0.2\text{ s}$ median recovery) | **$+29.7$ pp PDR** ($p < 10^{-5}$, rank-biserial $0.95$) |
| **Swarm Scale ($N=50$ UAVs)** | AODV (0.875), GPSR (0.861), OLSR (0.857) | **0.939 PDR** ($\pm 0.076$) | **$+6.4$ to $+8.2$ pp PDR** ($p \le 0.0002$, Kruskal-Wallis $p = 7.7 \times 10^{-5}$) |
| **Protocol Compute ($N=50$)** | OLSR ($0.224\text{ ms/node/tick}$) | **$0.082\text{ ms/node/tick}$** | **$2.7\times$ lower compute**, $O(\mathrm{deg})$ scaling |
| **Extension A (ORCA Safety)** | Standard ORCA (69.5% link retention) | **100.0% link retention** | **$+43.9\%$ relative retention** ($p = 1.3 \times 10^{-4}$), zero collisions |
| **Extension B (CBBA Task Alloc.)** | Loss-naive CBBA (0–60% channel loss) | **$+40$ to $+65$ pp task completion** | **$6.3\times$ lower route cost** ($p \le 0.0073$) |

---

## Repository Structure

```
.
├── src/                    # ROS 2 packages & core algorithms
│   ├── mpcare_agent/       # MP-CARE router, cost engine, link predictor, event trigger
│   ├── network_model/      # IEEE 802.11b log-distance PRR/ETX physical channel model
│   ├── gpsr_forwarding/    # GPSR greedy + perimeter forwarding baseline
│   ├── aodv_routing/       # RFC 3561 AODV routing baseline
│   ├── olsr_routing/       # RFC 3626 OLSR routing baseline
│   ├── cbba_allocator/     # Communication-aware CBBA task allocation node
│   ├── orca_avoidance/     # Connectivity-aware ORCA collision avoidance node
│   └── logger/             # 27-column high-rate telemetry logging system
├── scripts/                # Headless discrete-event simulation engine & benchmark scripts
│   ├── run_sim.py          # Primary 0.1s tick headless simulation engine
│   ├── run_sweeps.py       # Loss, mobility, and node-failure evaluation campaigns
│   ├── run_phase5.py       # ORCA (Ext. A) and CBBA (Ext. B) evaluation campaigns
│   ├── run_phase6.py       # N=5 to N=50 swarm scalability campaigns
│   ├── run_phase7.py       # Ablation study (A1-A7) and weight sensitivity Pareto front
│   ├── gen_paper_arch_fig.py # Publication architecture diagram generator
│   └── stats_utils.py      # Mann-Whitney U, Holm-Bonferroni, Kruskal-Wallis implementations
├── tests/                  # Unit and integration test suite (101 pytest checks)
├── launch/                 # ROS 2 launch files (mpcare_sim.launch.py)
├── config/                 # ROS 2 YAML configuration files
└── worlds/                 # Gazebo SDF simulation world files
```

Datasets (per-trial telemetry CSVs and summary JSONs) are deposited separately on
Zenodo with a DOI cited in the paper, and the LaTeX manuscript is submitted
directly to the journal — neither is tracked in this repository.

---

## Quickstart & Installation

### Requirements
* **Python**: Python 3.9+ (tested on Python 3.13)
* **Python Dependencies**: `matplotlib`, `pytest`, `pypdfium2`
* **ROS 2 & Gazebo (Optional for Middleware Validation)**: Ubuntu 26.04 LTS + ROS 2 Lyrical + Gazebo Jetty

### Python Environment Setup
```bash
# Clone the repository
git clone https://github.com/sanmitramukherjee/MP-CARE.git
cd MP-CARE

# Install optional test and plotting dependencies
pip install pytest matplotlib pypdfium2
```

---

## Reproducing Paper Results & Figures

### 1. Run the Test Suite (101 Checks)
```bash
pytest
```

### 2. Headless Simulation Smoke Test
```bash
python scripts/run_sim.py
```

### 3. Run Benchmark Campaigns & Generate Paper Figures
```bash
# Loss, mobility, and node-failure sweeps (Figures 2, 3, 4, 5)
python scripts/run_sweeps.py

# ORCA & CBBA extensions (Figures 6, 7)
python scripts/run_phase5.py

# Swarm scalability sweeps N=5 to N=50 (Figures 8, 9)
python scripts/run_phase6.py

# Ablation study & weight sensitivity (Figures 10, 11)
python scripts/run_phase7.py

# Generate black-and-white IEEE architecture figure (Figure 1)
python scripts/gen_paper_arch_fig.py
```

### 4. ROS 2 & Gazebo Launch (Middleware Validation)
```bash
colcon build --symlink-install
source install/setup.bash
ros2 launch launch/mpcare_sim.launch.py trial_id:=trial_001
```

---

## Compiling the LaTeX Paper

The compiled PDF is self-contained at `paper/main.pdf`. To rebuild `main.pdf` from the LaTeX source:

### Using Tectonic (Recommended)
```bash
cd paper
tectonic main.tex
```

### Using Standard TeX Live / MikTeX
```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## Citation

If you find this work or codebase useful in your research, please cite our paper:

```bibtex
@article{mukherjee2026mpcare,
  author    = {Mukherjee, Sanmitra},
  title     = {MP-CARE: Mobility-Predictive, Communication-Aware Routing with Event-Triggered Re-planning for Multi-UAV Swarms},
  journal   = {SSRN Electronic Journal},
  year      = {2026},
  doi       = {10.2139/ssrn.7647778},
  url       = {https://doi.org/10.2139/ssrn.7647778},
  note      = {Preprint under consideration at Elsevier Ad Hoc Networks}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
