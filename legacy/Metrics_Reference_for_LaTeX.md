# Metrics Reference for LaTeX Paper
## IEC 61850 Digital Substation — Complex Network Scalability Simulation

This document provides a complete, equation-level explanation of every metric
computed by the simulation codebase (`analysis.py`, `model.py`, `simulation.py`,
`config.py`). It is intended to be passed verbatim to a LaTeX-writing agent so
that each metric can be included in the paper with its proper mathematical
definition, physical interpretation, units, and implementation notes.

---

## 0. Physical Context and Normalisation

All latency weights are computed as dimensionless values normalised by:

```
T_MAX = 0.003 s  (3 ms)
```

`T_MAX` is the hard real-time deadline imposed by IEC 61850 for Class P6
(GOOSE Protection) and Class P4 (Sampled Values) traffic. A normalised weight
`W[i,j] = 1.0` therefore means the edge delay equals exactly the 3 ms IEC
deadline. Weights are converted back to milliseconds for reporting:

```
latency_ms = W[i,j] * T_MAX * 1000
```

Link capacities used in the model:

| Parameter              | Value      | Description                              |
|------------------------|------------|------------------------------------------|
| `CAPACITY_DEFAULT`     | 100 Mbps   | Standard bay-device access links         |
| `CAPACITY_TRUNK`       | 100 Mbps   | Inter-switch trunk (base scenario)       |
| `CAPACITY_TRUNK_UPGRADED` | 1 Gbps  | Inter-switch trunk (upgraded scenario)   |
| `FABRIC_CAPACITY`      | 68 000 Mbps | Switch backplane capacity               |
| `T_PROC_SWITCH`        | 4 µs       | Fixed per-hop switch processing delay    |
| `K_BUFFER`             | 28 149 pkts | M/M/1/K buffer size (4 MiB / 149 B)    |

---

## 1. Edge Weight Model (Physical Layer Delays)

The 5-layer directed graph assigns three weight matrices to every directed edge
`(i → j)`:

- **W[i,j]** — normalised end-to-end latency (dimensionless, range [0, 1])
- **J[i,j]** — normalised jitter (same units as W)
- **PL[i,j]** — packet-loss probability (dimensionless, range [0, 1])

Each layer uses a different physical model described below.

---

### 1.1 Transmission Delay (Layer L1 → L2)

**What it models:** The time required to serialise one Ethernet frame onto the
access link, i.e., the transmission delay at the ingress port of the device.

**Equation:**

The weighted-average frame size across all VLANs that device `d` uses is:

```
s_avg = Σ (size_k * freq_k) / Σ freq_k
```

where the sums run over all VLANs `k` carrying traffic from device `d`, and
`size_k` (bytes), `freq_k` (Hz) are the frame size and transmission frequency
of VLAN `k` respectively. The normalised transmission weight is then:

```
W[L1_d, L2_d] = (s_avg * 8) / (CAPACITY_DEFAULT * T_MAX)
```

**Terms:**
- `s_avg * 8` — average frame size in bits
- `CAPACITY_DEFAULT` — access link speed (100 Mbps = 100 × 10^6 bps)
- `T_MAX` — 3 ms normalisation constant

**Units:** Dimensionless (ratio of transmission time to deadline).

**Physical meaning:** A value close to 1 means the serialisation of a single
average-size frame already consumes a significant fraction of the allowed
3 ms budget.

---

### 1.2 Switch Processing Delay (Layer L2 → L3)

**What it models:** The fixed, deterministic store-and-forward latency inside a
managed Ethernet switch, independent of traffic load.

**Equation:**

```
W[L2_d, L3_vlan] = T_PROC_SWITCH / T_MAX = 4e-6 / 0.003 ≈ 0.00133
```

**Terms:**
- `T_PROC_SWITCH = 4 µs` — empirical per-hop processing delay of a managed switch.

**Units:** Dimensionless.

**Physical meaning:** At 4 µs per hop, this term is very small relative to
queuing and transmission delays. It becomes significant only when frames
traverse many switch hops in succession.

---

### 1.3 HOL-Blocking Delay (Layer L3 → L4): Head-of-Line Blocking Model

**What it models:** Contention inside the switch fabric (backplane) caused by
multiple input ports simultaneously trying to reach the same output port. This
is the Head-of-Line (HOL) blocking phenomenon in input-buffered switches.

#### 1.3.1 Fabric Utilisation per Priority

Traffic loads `λ_sw[sw][p]` (Mbps) are accumulated per switch `sw` and per
IEEE 802.1Q priority `p`. Under strict-priority queuing, the effective load seen
by priority class `p` includes all traffic at priority `p` and all higher
priorities:

```
u_p[sw][p] = ( Σ_{k=p}^{7} λ_sw[sw][k] ) / FABRIC_CAPACITY
```

where `FABRIC_CAPACITY = 68 000 Mbps` is the switch backplane throughput.

**Terms:**
- `λ_sw[sw][k]` — aggregate traffic load in Mbps injected into switch `sw` at
  priority `k` (summed over all source devices and all flows).
- `FABRIC_CAPACITY` — backplane capacity of the switch in Mbps.
- `u_p[sw][p]` — fabric utilisation seen by priority `p` (dimensionless, ρ-like
  quantity; values near or above 1 indicate fabric saturation).

#### 1.3.2 HOL-Blocking Latency

Using an M/D/1 approximation for HOL blocking (deterministic service time
inside the fabric):

```
W_hol[p] = ( T_PROC_SWITCH * (u_p / (2 * (1 - u_p))) ) / T_MAX
         , valid when 0 < u_p < 1
```

This is the Pollaczek-Khinchine (P-K) mean value formula for the M/D/1 queue.

**Terms:**
- `T_PROC_SWITCH` — base service time (4 µs) used as the fabric slot duration.
- `u_p / (2*(1-u_p))` — mean number of fabric slots spent waiting (M/D/1 mean
  waiting time factor).

**Units:** Dimensionless (normalised by T_MAX).

#### 1.3.3 HOL-Blocking Jitter

The jitter on the L3 → L4 edges is derived from the variance of the HOL waiting
time distribution (coefficient of variation of M/D/1):

```
J_hol[p] = W_hol[p] * sqrt( (2 - u_p) / u_p )
```

**Terms:**
- The factor `sqrt((2 - u_p) / u_p)` is the normalised standard deviation of
  the M/D/1 waiting time, which captures how variable the HOL delay is as a
  function of utilisation `u_p`.

**Physical meaning:** As `u_p → 1`, both the mean and variance of HOL blocking
grow without bound, indicating that the fabric is approaching saturation and
frame delivery timing becomes increasingly unpredictable.

---

### 1.4 Output-Port Queuing Delay (Layer L4 → L5): M/M/1 Model

**What it models:** Delay experienced by frames waiting in the IEEE 802.1Q
strict-priority output queue at each egress port of the switch.

#### 1.4.1 Average Frame Size and Traffic Intensity

For each output port and each priority class `k`:

```
l_avg_k = (ssf_k / sf_k) * 8   [bits]
```

where `ssf_k = Σ size_j * freq_j` and `sf_k = Σ freq_j` are accumulated over
all flows at priority `k` entering that port.

The aggregate offered load (traffic intensity) for priority `k` under
strict-priority queuing is:

```
bps_accum_k = Σ_{p=k}^{7} bps_p   [bps]
ρ_k = bps_accum_k / C
```

where `C` is the output port capacity (100 Mbps for access ports; 1 Gbps for
the upgraded trunk).

**Terms:**
- `ρ_k` — traffic intensity at priority `k`. When `ρ_k < 1` the queue is
  stable; when `ρ_k ≥ 1` the queue is saturated and delay is theoretically
  unbounded.

#### 1.4.2 M/M/1 Queuing Delay

For a stable queue (`0 < ρ_k < 1`), the mean frame waiting time in the
output queue is given by the M/M/1 Pollaczek-Khinchine formula (modified by
an empirical scale factor `FACTOR_16 = 16` that accounts for IEC 61850
multi-frame SV bursts within a single scheduling epoch):

```
delay_s = (ρ_k * l_avg_k) / (FACTOR_16 * C * (1 - ρ_k))   [seconds]
W_queue  = delay_s / T_MAX
```

For `ρ_k ≥ 1` (saturation), `W_queue` is clamped to 1.0 (deadline exceeded).

**Terms:**
- `ρ_k * l_avg_k / (C * (1-ρ_k))` — standard M/M/1 mean sojourn time (Little's
  law applied to an M/M/1 queue with average service time `l_avg/C`).
- `FACTOR_16` — scaling factor that empirically corrects the M/M/1 formula for
  the bursty, multi-flow IEC 61850 traffic pattern (SV streams from multiple
  merging units arriving in near-synchrony at 4 800 Hz).

#### 1.4.3 Total Edge Weight

The total normalised delay on a L4 → L5 edge at priority `k` accumulates the
transmission weight and the queuing weight:

```
W_total_k = min(1.0, W_trans_k + W_queue_k + W_accum_{k+1})
```

where `W_accum` carries the delay already accumulated by higher-priority
preemption (strict-priority inheritance across classes).

#### 1.4.4 Output-Port Jitter (M/M/1)

The jitter on L4 → L5 edges captures the standard deviation of M/M/1 sojourn
time:

```
J_queue = W_queue * sqrt( (2 - ρ_k) / ρ_k )
```

This expression is derived from the second moment of the M/M/1 waiting time
distribution and expresses how irregular the frame departure timing is relative
to the mean queuing delay.

---

### 1.5 M/M/1/K Packet Loss Rate (Altman–Jean-Marie Formula)

**What it models:** The probability that an arriving frame finds all `K` buffer
slots occupied and is therefore dropped. This is the Packet Loss Rate (PLR) at
each output port modelled as a finite-buffer M/M/1/K queue.

**Buffer size:**
```
K_BUFFER = floor(4 MiB / 149 bytes) = 28 149 packets
```

**Equation (Altman–Jean-Marie, 1997):**

For an M/M/1/K queue with traffic intensity `ρ`:

```
PLR = ρ^K * (1 - ρ) / (1 - ρ^(K+1))
```

Three numerically stable branches are used in the implementation:

| Case                | Formula used                          | Physical interpretation                       |
|---------------------|---------------------------------------|-----------------------------------------------|
| `ρ = 0`             | `PLR = 0`                             | No traffic, no loss                           |
| `ρ ≈ 1` (ε < 1e-9) | `PLR = 1 / (K + 1)`                  | Symmetric balanced state                      |
| `ρ < 1`             | Standard Altman-JM formula (above)   | Buffer rarely fills; PLR ≈ 0 for large K      |
| `ρ > 1`             | `PLR = (ρ - 1) / ρ` (limit as K→∞)  | Always below 100%; excess load is dropped     |

**Terms:**
- `ρ` — traffic intensity at the output port (same `ρ_k` as Section 1.4).
- `K = K_BUFFER = 28 149` — maximum number of frames the output buffer can hold.
- For `ρ < 1` and large `K`, `ρ^K → 0` (underflow), giving `PLR ≈ 0` — the
  buffer is large enough that it virtually never fills under stable load.
- For `ρ > 1`, frames arrive faster than they can be served; the fraction
  `(ρ-1)/ρ` of arriving traffic must be dropped on average.

**Units:** Dimensionless probability, reported as a percentage (multiplied by 100).

**Physical meaning:** PLR directly determines the reliability of protection
commands. IEC 61850 practical targets require PLR < 0.1 % for P6 (GOOSE
Protection) traffic. Values above this threshold indicate that the substation
network cannot guarantee reliable delivery of protection messages.

---

## 2. Graph-Theoretic Metrics

These metrics are computed on the graph formed by the 5-layer topology with
edge weights set to actual latency values (in ms), using the NetworkX library.

---

### 2.1 Algebraic Connectivity — Fiedler Eigenvalue (λ₂)

**What it measures:** The robustness of the network against partitioning, i.e.,
its ability to remain connected when nodes or links fail.

**Mathematical definition:**

Given the undirected symmetrised graph `G_sym` with Laplacian matrix `L`,
where `L = D - A_sym` (`D` = degree matrix, `A_sym` = symmetrised adjacency
matrix with latency weights), the eigenvalues of `L` satisfy:

```
0 = λ₁ ≤ λ₂ ≤ λ₃ ≤ ... ≤ λ_n
```

The **Fiedler eigenvalue** `λ₂` is the second-smallest eigenvalue (the first
being always zero for a connected graph). It is also called the **algebraic
connectivity** of the graph.

**Computation:**
The symmetrised weight matrix is formed as:

```
W_sym = (W + W^T) / 2
```

Then `λ₂` is computed via Lanczos iteration (numerically stable for sparse
graphs):

```
λ₂ = algebraic_connectivity(G_sym, weight='weight', method='lanczos')
```

**Interpretation:**
- `λ₂ > 0` — the graph is connected.
- Larger `λ₂` → better topological robustness (harder to partition into
  disconnected components by removing a few nodes or edges).
- `λ₂ → 0` — the graph has a bottleneck (a small cut) that can easily disconnect
  the network.
- The eigenvector associated with `λ₂` (the Fiedler vector) identifies which
  nodes lie near this bottleneck.

**Units:** Weighted (ms-weighted graph), so units are ms⁻¹ conceptually, though
the value is reported as a dimensionless ratio in the paper's scalability plots.

**Relevance to IEC 61850:** As the number of bays `N` grows, `λ₂` reveals
whether the two-switch architecture maintains adequate topological redundancy
or develops structural weak points (e.g., the inter-switch trunk becomes a
single point of failure).

---

### 2.2 Global Network Efficiency — Latora–Marchiori Efficiency (E)

**What it measures:** The average ease of information exchange (reachability)
across all node pairs in the network, with longer (higher-latency) paths
contributing less to the total efficiency.

**Mathematical definition (Latora & Marchiori, 2001):**

```
E(G) = (1 / (n*(n-1))) * Σ_{i≠j} 1 / d(i,j)
```

where:
- `n` = number of nodes in the graph
- `d(i,j)` = shortest weighted path distance between nodes `i` and `j`
  (measured in ms, using Dijkstra's algorithm on the symmetrised graph).
- If no path exists between `i` and `j`, the term `1/d(i,j) = 0` (consistent
  with the convention that `1/∞ = 0`).

**Computation:**

```python
global_efficiency = nx.global_efficiency(G_sym)
```

NetworkX computes this directly on the symmetrised, latency-weighted graph.

**Interpretation:**
- `E = 1` — perfectly efficient network where every pair of nodes is directly
  connected with zero latency (theoretical maximum).
- `E → 0` — extremely inefficient or nearly disconnected network.
- A high `E` with latency weights means that, on average, any device can
  communicate with any other device via short (low-latency) paths.
- A decrease in `E` as `N` grows signals that the average path between devices
  is getting longer (higher latency), or that some node pairs become
  unreachable.

**Units:** Dimensionless (normalised harmonic mean of inverse distances).

**Relevance to IEC 61850:** Efficiency quantifies the overall QoS of the
substation communication fabric. A drop in efficiency as `N` increases indicates
that protection and control messages must traverse more hops or encounter higher
delays, potentially violating IEC 61850 timing constraints.

---

### 2.3 Betweenness Centrality (BC)

**What it measures:** The importance of each node as a relay point on the
shortest paths between all other node pairs. High betweenness identifies
nodes whose failure would disconnect many communication routes.

**Mathematical definition:**

```
BC(v) = Σ_{s≠v≠t} [ σ(s,t|v) / σ(s,t) ]
```

where:
- `σ(s,t)` = total number of shortest (minimum-latency) paths from source `s`
  to target `t` in the directed, latency-weighted graph `G_latency`.
- `σ(s,t|v)` = number of those shortest paths that pass through node `v`.
- The sum ranges over all ordered pairs `(s,t)` with `s ≠ t ≠ v`.

**Normalisation:** The normalised betweenness divides by the maximum possible
number of paths, `(n-1)*(n-2)` for directed graphs:

```
BC_norm(v) = BC(v) / ((n-1)*(n-2))
```

**Computation:**

```python
bet_cen = nx.betweenness_centrality(G_latency, weight='weight', normalized=True)
```

For large graphs (`N ≥ 6` bays, i.e., more than ~50 nodes), a k-approximate
sampling version is used:

```python
k_val = max(10, 40 - len(labels) // 10)
bet_cen = nx.betweenness_centrality(G_latency, weight='weight', normalized=True, k=k_val)
```

**Interpretation:**
- Nodes with `BC_norm ≈ 1` lie on virtually every shortest path in the network
  — their failure would catastrophically disrupt most communications.
- Nodes with `BC_norm ≈ 0` are peripheral; their removal has minimal impact
  on the network's routing efficiency.
- In IEC 61850 substations, VLAN nodes (`L3`) and priority-queue nodes (`L4`)
  corresponding to protection traffic are expected to have the highest
  betweenness, as they carry the most critical flows.

**Units:** Dimensionless, range [0, 1].

**Relevance to IEC 61850:** Betweenness centrality guides redundancy investments:
nodes with consistently high BC (both at steady state and during GOOSE bursts)
should be duplicated or protected with fast failover mechanisms.

---

### 2.4 Node Vulnerability Index (V)

**What it measures:** The fractional drop in the global efficiency of the
network caused by the removal of a single node. It directly quantifies how
critical each node is to the overall communication fabric.

**Mathematical definition:**

```
V(v) = max(0, [ E(G) - E(G \ {v}) ] / E(G) )
```

where:
- `E(G)` = global network efficiency of the full graph (see Section 2.2).
- `E(G \ {v})` = global network efficiency after removing node `v` and all
  its incident edges.
- The `max(0, ·)` clamp prevents negative values due to numerical noise.

**Computation:**
Computed only at steady state (GOOSE Δt = 1000 ms) and only for the top-20
nodes by betweenness centrality (the most expensive metric due to `n`
efficiency re-computations):

```python
V_i = max(0.0, (E_G - E_G_minus_i) / E_G)
```

**Interpretation:**
- `V(v) = 0` — removing node `v` causes no degradation in network efficiency;
  the node is non-critical.
- `V(v) → 1` — removing node `v` collapses the network efficiency to near zero;
  the node is a single point of failure.
- In practice, `V(v) > 0.1` (10 % efficiency drop) is considered a significant
  vulnerability requiring redundancy.

**Units:** Dimensionless, range [0, 1].

**Relevance to IEC 61850:** The Vulnerability Index is the most operationally
useful structural metric because it directly translates topological criticality
into a QoS impact (efficiency loss), which is directly related to the
probability of IEC 61850 timing deadline violations when that node fails.

---

### 2.5 Load Gini Coefficient

**What it measures:** The degree of inequality (imbalance) in the distribution
of traffic load across all edges of the network. A Gini coefficient of 0
indicates perfectly uniform load; 1 indicates all load is concentrated on a
single edge.

**Mathematical definition:**

Given the vector of `m` non-zero edge weights `w_1 ≤ w_2 ≤ ... ≤ w_m`
(sorted in ascending order), the Gini coefficient is:

```
G = (2 * Σ_{i=1}^{m} i * w_i) / (m * Σ_{i=1}^{m} w_i) - (m+1)/m
```

This is the standard discrete Gini index derived from the Lorenz curve.

**Computation:**

```python
w_sorted = np.sort(edge_weights)        # ascending order
n_w = len(w_sorted)
gini = (2 * np.sum(np.arange(1, n_w+1) * w_sorted) /
        (n_w * np.sum(w_sorted))) - (n_w+1) / n_w
```

**Interpretation:**
- `Gini = 0` — all edges carry exactly the same latency/load (ideal uniformity).
- `Gini → 1` — almost all delay is concentrated on one or a few edges (extreme
  imbalance); a few edges are heavily loaded while the rest are nearly idle.
- A high Gini coefficient in the latency weight matrix indicates that the
  network has severe bottlenecks: a small number of output ports or trunk links
  are operating near saturation while most paths are lightly loaded.

**Units:** Dimensionless, range [0, 1].

**Relevance to IEC 61850:** A high Gini coefficient predicts where IEC 61850
deadline violations will first occur as `N` grows. It highlights the asymmetry
of load distribution between the two switches (SW1 aggregates all global
devices) and flags the BBP output port as the primary bottleneck in the base
scenario.

---

## 3. Per-Priority QoS Metrics

These metrics are reported separately for each IEEE 802.1Q priority class used
in the IEC 61850 substation model:

| Priority | Protocol     | Traffic type                             | IEC 61850 deadline |
|----------|--------------|------------------------------------------|--------------------|
| P7       | PTP          | IEEE 1588 time synchronisation           | Not hard-bounded   |
| P6       | GOOSE        | Protection commands (critical)           | 3 ms               |
| P5       | SMC/Control  | Station Monitor Controller status        | Best-effort        |
| P4       | SV           | IEC 61869-9 Sampled Values (high-rate)  | 3 ms               |

All QoS metrics are collected from the L4 → L5 edges (output-port queues) of
the 5-layer graph, which is where the dominant latency, jitter, and loss
occur under load.

---

### 3.1 Latency Metrics

For each priority class `p ∈ {4, 5, 6, 7}`, the following statistics are
collected from all L4_Priority edges of that priority:

**Average Latency:**
```
Latency_Avg_Pp = mean( W[i,j] * T_MAX * 1000 )   [ms]
```
over all edges `(i → j)` where `i` is a `L4_Priority_*_Prio_p` node.

**Maximum Latency:**
```
Latency_Max_Pp = max( W[i,j] * T_MAX * 1000 )   [ms]
```

**Physical meaning:** `Latency_Max_P6` is the key compliance indicator — if
it exceeds 3 ms, the network violates the IEC 61850 hard deadline for
protection traffic. `Latency_Max_P4` is similarly bounded by the 3 ms SV
deadline.

**IEC Violation flag:**
```
IEC_Violation = 1   if Latency_Max_P6 > 3 ms  OR  Latency_Max_P4 > 3 ms
              = 0   otherwise
```

---

### 3.2 Jitter Metrics

**Average Jitter:**
```
Jitter_Avg_Pp = mean( J[i,j] * T_MAX * 1000 )   [ms]
```

where `J[i,j]` is the normalised jitter computed either from HOL-blocking
(L3→L4 edges) or from the M/M/1 variance formula (L4→L5 edges):

```
J = W_queue * sqrt( (2 - ρ) / ρ )
```

**Physical meaning:** Jitter quantifies the variability of frame delivery
timing. For IEC 61869-9 Sampled Values (P4), excessive jitter causes
synchronisation errors in the merging units and protection relays. For PTP
(P7), jitter directly degrades the accuracy of the IEEE 1588 grandmaster
clock distribution.

**Units:** Milliseconds [ms].

---

### 3.3 Packet Loss Rate (PLR) Metrics

For each priority class `p`, three statistics are derived from the M/M/1/K
loss probabilities `PL[i,j]` (see Section 1.5):

**Average PLR:**
```
Loss_Avg_Pp = mean( PL[i,j] * 100 )   [%]
```

**Maximum PLR:**
```
Loss_Max_Pp = max( PL[i,j] * 100 )   [%]
```

**95th Percentile PLR:**
```
Loss_P95_Pp = percentile_95( PL[i,j] * 100 )   [%]
```

**IEC 61850 compliance threshold:** PLR < 0.1 % for P6 (GOOSE Protection).
Values above this threshold are flagged in the Executive Summary as
`FAIL (Packet Loss)`.

**Physical meaning:**
- `Loss_Avg` gives the system-level expected packet loss across all output
  ports at a given priority.
- `Loss_Max` identifies the single worst-case port (typically the BBP
  output port, which aggregates SV from all bays).
- `Loss_P95` is useful for understanding tail behaviour: 95 % of output
  ports have a PLR at or below this value, while 5 % are worse.

---

### 3.4 Switch Fabric Utilisation

**Per-switch maximum utilisation:**

```
Max_Util_SW1 = max_p( u_p['SW1'][p] )
Max_Util_SW2 = max_p( u_p['SW2'][p] )
```

where `u_p[sw][p]` is the accumulated fabric utilisation at switch `sw` for
priority class `p` and all higher priorities (see Section 1.3.1).

**Units:** Dimensionless ratio (0 to 1, where 1 = 100 % fabric utilisation).

**Physical meaning:** When `Max_Util > 0.8`, the switch fabric is approaching
saturation and HOL-blocking delays grow rapidly. A value above 1.0 indicates
fabric overload (not physically possible in practice — the simulation models
this as a loss condition).

---

## 4. Centrality Burst Ratio (CBR)

**What it measures:** The amplification of a node's topological importance
during a GOOSE fault burst, relative to its steady-state importance. A high
CBR means the node becomes a disproportionate bottleneck specifically during
the fault — even if it is not critical under normal conditions.

**Mathematical definition:**

```
CBR(v) = C_v(burst) / C_v(steady)
```

where:
- `C_v(steady)` = mean normalised betweenness centrality of node `v` at
  steady state (GOOSE retransmission interval Δt = 1000 ms).
- `C_v(burst)` = mean normalised betweenness centrality of node `v` at the
  burst peak (GOOSE retransmission interval Δt = 4 ms, first two
  retransmissions after fault detection).

**Computation:**
```python
CBR_df['CBR'] = CBR_df['C_burst'] / CBR_df['C_steady']
```
Nodes with `C_steady = 0` are excluded (ratio undefined).

**Interpretation:**
- `CBR = 1` — node importance is unchanged during the burst; the GOOSE flood
  does not alter routing topology through that node.
- `CBR > 1` — the node becomes proportionally more important during the burst;
  flows concentrate through it as GOOSE traffic saturates alternative paths.
- `CBR >> 1` (e.g., > 5) — the node is a critical transient bottleneck that
  does not appear critical in steady state; this is the most actionable
  finding, as it identifies infrastructure that needs hardening specifically
  for fault scenarios.

**Units:** Dimensionless ratio, range [1, ∞) for nodes that gain importance.

**Relevance to IEC 61850:** The CBR directly prioritises which network nodes
need redundancy investment for fault resilience. Trunk nodes, VLAN aggregation
nodes, and priority-queue nodes with high CBR are those that, when they fail
*during* a fault, will cause cascading protection failures — the worst possible
scenario in a substation.

---

## 5. Recovery Time

**What it measures:** The time required for the P6 (GOOSE Protection) latency
to return to within a defined tolerance of its steady-state value after a
GOOSE burst event.

**Mathematical definition:**

Let `L_steady` be the mean maximum P6 latency at steady state (Δt = 1000 ms).
Define the recovery threshold as:

```
L_threshold = threshold_factor * L_steady   (default: 1.1 × L_steady)
```

The **Recovery Time** `T_rec` is the earliest post-fault snapshot time at which:

```
Latency_Max_P6(t) ≤ L_threshold
```

If the latency never falls back within the threshold during the simulated
burst sequence, `T_rec = NaN` (no recovery observed).

**Computation:**
```python
threshold = steady_lat * 1.1        # 10% tolerance band
recovered = burst_group[burst_group['Latency_Max_P6'] <= threshold]
recovery_time = recovered['Snapshot_Time_ms'].min()
```

**GOOSE burst timeline (from IEC 61850-8-1):**
The burst sequence simulated is:
- t = 0 ms: steady state snapshot (Δt = 1000 ms)
- t = 4 ms: first fast retransmission (Δt = 4 ms)
- t = 8 ms: second fast retransmission (Δt = 4 ms)
- t = 16 ms: Δt = 8 ms
- t = 32 ms: Δt = 16 ms
- ... (exponential back-off doubling)
- Final snapshot: Δt = 1000 ms (steady state restored)

**Units:** Milliseconds [ms].

**Physical meaning:** A short recovery time means the network can absorb the
GOOSE burst and return to normal operation quickly. A long recovery time (or
NaN) means the network remains congested long after the initial fault, which
is especially dangerous because protection relays may issue additional GOOSE
retransmissions during this period, further stressing the network.

**Relevance to IEC 61850:** IEC 61850 requires that protection functions
remain operational throughout the fault event. A large recovery time indicates
that the network is spending an extended period above the 3 ms latency limit,
during which protection co-ordination may be compromised.

---

## 6. Summary Table of All Metrics

| Metric | Symbol | Formula / Source | Units | IEC 61850 threshold |
|--------|--------|-----------------|-------|---------------------|
| Transmission delay | W_trans | (s_avg·8) / (C · T_MAX) | — | — |
| Switch processing delay | W_proc | T_PROC / T_MAX | — | — |
| HOL-blocking delay | W_hol | T_PROC · u/(2(1-u)) / T_MAX | — | — |
| HOL-blocking jitter | J_hol | W_hol · √((2-u)/u) | — | — |
| M/M/1 queuing delay | W_queue | ρ·l_avg / (16·C·(1-ρ) · T_MAX) | — | — |
| M/M/1 jitter | J_queue | W_queue · √((2-ρ)/ρ) | — | — |
| M/M/1/K PLR | PLR | ρ^K(1-ρ)/(1-ρ^{K+1}) | — | < 0.1 % for P6 |
| Algebraic connectivity | λ₂ | 2nd eigenvalue of L(G_sym) | (ms)⁻¹ | Maximise |
| Global efficiency | E | (1/n(n-1)) Σ 1/d(i,j) | — | Maximise |
| Betweenness centrality | BC | σ(s,t\|v)/σ(s,t) normalised | — | Monitor |
| Node vulnerability | V | (E(G) - E(G\{v})) / E(G) | — | Minimise |
| Load Gini coefficient | G | Lorenz curve integral | — | Minimise |
| Latency Average P{p} | Lat_Avg | mean(W · T_MAX · 1000) | ms | — |
| Latency Maximum P{p} | Lat_Max | max(W · T_MAX · 1000) | ms | < 3 ms (P4, P6) |
| Jitter Average P{p} | Jit_Avg | mean(J · T_MAX · 1000) | ms | — |
| PLR Average P{p} | Loss_Avg | mean(PL · 100) | % | < 0.1 % (P6) |
| PLR Maximum P{p} | Loss_Max | max(PL · 100) | % | < 0.1 % (P6) |
| PLR 95th percentile P{p} | Loss_P95 | percentile_95(PL · 100) | % | — |
| Switch utilisation | u_p | Σλ_p / C_fabric | ratio | < 0.8 recommended |
| Centrality Burst Ratio | CBR | C(burst) / C(steady) | — | Minimise (> 1 = amplified) |
| Recovery time | T_rec | min t : Lat_Max_P6(t) ≤ 1.1·L_ss | ms | Minimise |

---

## 7. Key References

- **M/M/1 queuing:** Kleinrock, L. (1975). *Queueing Systems, Vol. 1*.
- **M/M/1/K PLR:** Altman, E., & Jean-Marie, A. (1997). Loss probabilities for
  finite-buffer Markovian queues. *Queueing Systems*.
- **Algebraic connectivity:** Fiedler, M. (1973). Algebraic connectivity of
  graphs. *Czechoslovak Mathematical Journal*.
- **Global efficiency:** Latora, V., & Marchiori, M. (2001). Efficient behavior
  of small-world networks. *Physical Review Letters*, 87(19).
- **Betweenness centrality:** Brandes, U. (2001). A faster algorithm for
  betweenness centrality. *Journal of Mathematical Sociology*, 25(2).
- **HOL blocking:** Hluchyj, M. G., & Karol, M. J. (1988). Queuing in
  high-performance packet switching. *IEEE Journal on Selected Areas in
  Communications*.
- **IEC 61850 timing:** IEC 61850-5:2013, Communication networks and systems
  for power utility automation — Part 5: Communication requirements for
  functions and device models.
- **GOOSE burst:** IEC 61850-8-1:2011, Specific communication service mapping
  (SCSM) — Mappings to MMS and to ISO/IEC 8802-3.
