# IEC 61850 Digital Substation — Complex Network Scalability Simulation

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

## Abstract

This repository accompanies a study on the scalability limits of IEC 61850 digital substations modelled as complex networks. We construct a five-layer graph (bay devices → process-bus switches → station-bus switch → bay protection → station monitor) and sweep the number of protection bays from 1 to 10, evaluating end-to-end latency, packet-loss rate, and algebraic-connectivity metrics for both 100 Mbps (base) and 1 Gbps (upgraded) inter-switch trunk scenarios. The model integrates IEEE 802.1Q Strict Priority Queuing, M/M/1/K buffer-loss theory, and a GOOSE burst-retransmission schedule to capture the transient behaviour that follows a protection event, providing quantitative guidance for substation network dimensioning aligned with IEC 61850 Class P6 hard deadlines.

---

## Repository Structure

```
CodigoDinamico/
├── config.py          # Global constants, traffic specs, VLAN priority map
├── model.py           # Device generation, demand tensor, GOOSE burst, topology
├── analysis.py        # M/M/1/K weights, network metrics, CBR, recovery time
├── simulation.py      # Run loop, Excel exports, all figures  ← entry point
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── results/           # Created automatically on first run
│   ├── Global_Report.xlsx
│   ├── Metrics_By_Scenario.xlsx
│   ├── Detail_N{n}.xlsx          (one per bay count)
│   └── Fig{2-9}_*.png            (all publication figures)
│
└── legacy/
    └── EscalabilidadOptimizado2.py   # Original reference (unchanged, Spanish)
```

### Module Responsibilities

| Module | Key Functions | Role in Paper |
|---|---|---|
| `config.py` | `get_vlan_priority`, `get_mbps` | Protocol & timing constants (Section II) |
| `model.py` | `generate_devices`, `generate_demand_tensor`, `build_topology` | Substation graph construction (Section III) |
| `analysis.py` | `compute_weights`, `analyze_network`, `compute_cbr` | Queuing model & metrics (Section IV) |
| `simulation.py` | `run_simulation`, `export_*`, `plot_*` | Experiments & visualisation (Section V) |

---

## Installation

Python 3.10 or newer is required.

```bash
pip install -r requirements.txt
```

**Dependencies:** `numpy`, `pandas`, `networkx`, `matplotlib`, `scipy`, `seaborn`, `openpyxl`

---

## Usage

```bash
python simulation.py
```

The script runs both scenarios (`base` and `upgraded`) for bay counts N = 1…10, then writes all outputs to `results/`. Total runtime is approximately 3–8 minutes depending on hardware.

No arguments are required. To change the maximum bay count or scenario list, edit the `__main__` block at the bottom of `simulation.py`.

---

## Generated Outputs

### Excel Reports

| File | Contents |
|---|---|
| `results/Global_Report.xlsx` | Six sheets: Executive Summary, Graph Metrics, Recovery Time, Vulnerability, CBR Top Nodes, Raw Data |
| `results/Metrics_By_Scenario.xlsx` | One sheet per scenario with all per-snapshot metrics |
| `results/Detail_N{n}.xlsx` | Per-hop latency and PLR breakdown for each bay count N |

### Figures

| File | Figure | Description |
|---|---|---|
| `results/Fig2_LatencyScalability.png` | Fig. 2 | Average end-to-end latency vs. N bays for all priorities and both scenarios |
| `results/Fig3_ComplexNetworkMetrics.png` | Fig. 3 | Fiedler value λ₂, global efficiency, load Gini coefficient vs. N bays |
| `results/Fig4_GooseBurst_N{n}.png` | Fig. 4 | Transient latency profile during GOOSE burst retransmission sequence |
| `results/Fig5_CentralityHeatmap_N{n}_{sc}.png` | Fig. 5 | Betweenness-centrality heatmap with node vulnerability index overlay |
| `results/Fig67_CBR_Vulnerability_N{n}_{sc}.png` | Fig. 6–7 | Critical bandwidth ratio and vulnerability distribution across nodes |
| `results/Fig8_MaxDelay_ByProtocol.png` | Fig. 8 | Maximum per-priority delay (PTP P7, SV P4, GOOSE P6) vs. N bays |
| `results/Fig9_PLR_ByProtocol.png` | Fig. 9 | Packet loss rate per priority (max and 95th percentile) vs. N bays |

---

## Model Overview

### Network Layers

```
L1  Bay devices       PP1, PP2 (protection IEDs), MU1, MU2 (merging units) — per bay
L2  Process switches  SW1, SW2 — aggregate bay traffic via 100 Mbps links
L3  Station switch    SW3 — inter-switch trunk (100 Mbps base / 1 Gbps upgraded)
L4  Bay protection    BBP — 1 Gbps uplink aggregating SV from all bays
L5  Station monitor   SMC — station-wide control and monitoring
    GPS               PTP grandmaster — IEEE 1588 time synchronisation
```

### Traffic Classes

| Class | Protocol | Rate | Frame size | Priority |
|---|---|---|---|---|
| PTP | IEEE 1588 | 3 Hz | 80 B | P7 |
| SV | IEC 61869-9 | 4 800 Hz | 149 B | P4 |
| GOOSE | IEC 61850-8-1 | 1 Hz (steady) + burst | 187 B | P6 |
| MON | MMS monitoring | 1 Hz | 171 B | P5 |
| SMC | Station controller | 1 Hz | 211 B | P5 |
| MU_RES | Merging unit reply | 1 Hz | 340 B | P5 |

### Key Parameters

| Parameter | Value |
|---|---|
| IEC 61850 hard deadline (T_MAX) | 3 ms |
| Bay device link capacity | 100 Mbps |
| Inter-switch trunk — base | 100 Mbps |
| Inter-switch trunk — upgraded | 1 Gbps |
| BBP uplink capacity | 1 Gbps |
| Switch processing delay | 4 µs per hop |
| Buffer capacity (M/M/1/K) | 28 149 packets (4 MiB / 149 B) |

### Packet Loss Model

Packet loss at each output port is computed using the M/M/1/K (Erlang-B) formula:

```
PLR = ρ^K (1 − ρ) / (1 − ρ^(K+1))
```

where ρ = λ/μ is the traffic intensity (can exceed 1 for overloaded links) and K = 28 149 is the buffer size. The implementation uses a numerically stable branch for ρ < 1, ρ = 1, and ρ > 1 to avoid floating-point overflow.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{AUTHOR_YEAR,
  title   = {Scalability Analysis of IEC 61850 Digital Substations
             as Complex Communication Networks},
  author  = {Last, First and Last, First},
  journal = {Journal Name},
  year    = {2026},
  volume  = {XX},
  pages   = {XX--XX},
  doi     = {10.XXXX/XXXXXXX}
}
```

---

## License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2026 [Author Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
