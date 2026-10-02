# Multilayer complex-network assessment of IEC 61850 digital-substation designs

Code and data of the paper *A Multilayer Complex-Network Tool for the Event-Driven
and Scalability-Aware Assessment of Digital Substation Designs* (J. A. Tovar Lozada,
F. Forero Rodríguez, M. T. Gutierrez Morales, G. A. Ramos López, Universidad de los
Andes). The tag `v1.0-paper` is the version used for the submitted manuscript.

The tool represents a switched IEC 61850 substation network as a five-layer
multilayer graph (equipment, ingress ports, VLANs, priority queues, egress ports)
whose edge weights are traffic-derived delays. The end-to-end delay of every
Sampled Value (SV) and GOOSE flow is a path sum over the graph (N·D/D/1 for
periodic SV streams, non-preemptive priority for the other classes, deterministic
fluid model for saturated queues). Graph indicators (structural, flow-constrained,
congestion threshold map) and component availability/maintainability are computed
on the same graph. The case study is a breaker-failure (50BF) scheme scaled from 1
to 10 bays.

The packet-level discrete-event simulator used as reference is in a separate
repository: <https://github.com/adriant25/iec61850-network-simulator> (tag
`v1.0-paper`).

## Installation

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt
```

Tested with Python 3.14, numpy 2.4, pandas 3.0, networkx 3.6, scipy 1.17,
matplotlib 3.10, openpyxl 3.1.

## Analytical model

| File | Content |
|------|---------|
| `config.py` | Traffic specification (frame sizes on the wire: frame + 20 B preamble/inter-frame gap), VLANs, priorities, switch parameters |
| `model.py` | Devices, demand tensor, GOOSE burst timeline, five-layer supra-adjacency matrix |
| `analysis.py` | Edge weights (N·D/D/1, Cobham, M/M/1/K, fluid overload), end-to-end path delays, graph indicators |
| `simulation.py` | Sweep N = 1..10, three link scenarios, eleven burst snapshots → `results/Global_Report.xlsx`, `results/Metrics_By_Scenario.xlsx` |

## Studies of the paper

Run from the repository root (each script documents its outputs):

| Script | Paper section |
|--------|---------------|
| `simulation.py` | Results: stability boundary, structural and flow-constrained indicators |
| `sensitivity.py` | Design sensitivity (two and four SV streams per bay) |
| `event_scope.py`, `single_bay_des.py` | Effect of the protection event and its scope |
| `congestion_threshold.py` | Congestion threshold map (rate-weighted flow betweenness) |
| `robustness.py t1` / `cap` / `scale` | Robustness: GOOSE T1, link capacity, computational scaling |
| `maintainability.py` | Availability and maintainability |
| `nc_bound.py`, `nc_panco.py`, `benchmark_figure.py` | Baselines: network-calculus bounds (own hop-by-hop and panco) |
| `extract_des_reference.py`, `des_replicates.py`, `compare_des.py`, `des_kpi_table.py` | Cross-validation against the DES |
| `paper_figures.py` | Figures of the paper (`results/paper/`) |
| `paper_numbers.py` | Every case-dependent number quoted in the text (`results/paper_numbers.txt`) |
| `run_wire_analytic.sh` | Runs the analytical chain in order |

`des_validation/` verifies the simulator against closed-form queueing results and
contains the external benchmark against the OPNET study of Mekkanen et al. (2014),
`external_benchmark.py` (see `des_validation/README.md`). `legacy/` keeps files of
earlier versions of the model for traceability; they are not used by the paper.

### Network calculus with panco

`nc_panco.py` uses panco (A. Bouillard, BSD-3) and lp_solve:

```bash
git clone https://github.com/anne-bou/panco tools/panco      # commit f035ccc
git -C tools/panco apply ../../panco_local_changes.patch    # only lets panco call a native lp_solve
pip install -e tools/panco
# lp_solve 5.5.2.11 (https://sourceforge.net/projects/lpsolve/) in tools/lp_solve/,
# or set PANCO_LPSOLVE to the lp_solve executable
python nc_panco.py
```

## DES data

`des_data/` contains every simulator run used in the paper, one folder per size
(`CASO<N>BAHIA(S)`), each with the KPI workbooks (`KPIs_PB.xlsx`, `KPIs_SB.xlsx`),
the exact input tables (`Inputs_used/`), the answers given to the simulator
prompts (`runtime_answers.txt`, including the seed) and the traffic generator
used (`traffic_generator_used.py`):

| Folder | Runs |
|--------|------|
| `CASO<N>BAHIA(S)/Results_wire` | Base design (100 Mbps), N = 1..10, seed 42 |
| `CASO7BAHIAS/Replicas_wire`, `CASO8BAHIAS/Replicas_wire` | Seeds 1–5 next to the stability boundary |
| `CASO7BAHIAS/SingleBay7_wire` | Breaker failure in one bay (seeds 42, 1, 2) |
| `CASO9BAHIAS/1000_wire`, `CASO10BAHIAS/Results1000_wire` | Trunk + BBP link at 1 Gbps |
| `CASO8BAHIAS/Trace_wire` | Per-frame trace of the SV flow MU1B2→BBP (8 bays) |

`des_runs/make_des_run.py` and `des_runs/make_des_run_wire.py` build a run folder
from the simulator code and the case input tables, adding the 20 B physical
overhead to every frame (set `DES_CASES_DIR` and `DES_SIMULATOR_DIR`). Point
`DES_DATA_DIR` to another folder to analyse other runs.

## License

MIT (see `LICENSE`). panco and lp_solve are third-party tools under their own
licenses and are not included.
