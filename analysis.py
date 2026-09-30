# -*- coding: utf-8 -*-
"""
analysis.py -- Physical weight computation and complex network metrics.

IEC 61850 Digital Substation -- Complex Network Scalability Simulation
----------------------------------------------------------------------
This module translates the 5-layer graph topology (from model.py) into
quantitative QoS metrics by:

  1. compute_weights()  -- Assigns latency (W), jitter (J) and packet-loss
                          probability (PL) to each directed edge using:
                            • M/M/1 queuing (delay at output ports)
                            • M/M/1/K queuing (PLR at output ports,
                              Altman–Jean-Marie formula, valid for ρ > 1)
                            • HOL-blocking model (fabric contention)

  2. analyze_network()  -- Derives graph-theoretic and QoS metrics from W/J/PL:
                            • Algebraic connectivity (Fiedler λ₂)
                            • Global network efficiency (Latora–Marchiori)
                            • Betweenness centrality (weighted, directed)
                            • Node vulnerability index
                            • Load Gini coefficient
                            • Per-priority latency, jitter, and PLR statistics

  3. compute_cbr()      -- Centrality Burst Ratio: ratio of burst-phase to
                          steady-state betweenness centrality per node.

  4. compute_recovery_time() -- Time for P6 (GOOSE) latency to return within
                               10 % of its steady-state value after a burst.
"""

from functools import lru_cache
from math import comb

import numpy as np
import pandas as pd
import networkx as nx

from config import (
    T_MAX, CAPACITY_DEFAULT, CAPACITY_TRUNK, CAPACITY_TRUNK_UPGRADED,
    FABRIC_CAPACITY, T_PROC_SWITCH, K_BUFFER, PORT_BUFFER_BYTES, OBS_WINDOW_S,
    SV_VLAN_IDS, GOOSE_VLAN_IDS, TRAFFIC_SPECS,
    get_vlan_priority, get_vlan_id, get_frame_specs, is_burst_vlan,
)

# =============================================================================
# 1. PHYSICAL WEIGHT COMPUTATION
# =============================================================================

# Scenarios:  'base'     -- every link at 100 Mbps
#             'upgraded' -- inter-switch trunk and BBP link at 1 Gbps
#             'bbp_only' -- only the BBP link at 1 Gbps
SCENARIOS = ('base', 'upgraded', 'bbp_only')


SV_PERIOD_S: float = 1.0 / TRAFFIC_SPECS['SV']['freq']
SV_FRAME_BITS: float = TRAFFIC_SPECS['SV']['size'] * 8


@lru_cache(maxsize=None)
def _ndd1_wait_units(n_other: int, D: float, pts: int = 600) -> float:
    """Mean N*D/D/1 wait in units of the service time (see ndd1_mean_wait)."""
    if n_other <= 0:
        return 0.0
    if n_other >= D:
        return float('inf')
    xs = np.linspace(0.0, n_other, pts)
    q = np.zeros_like(xs)
    for i, x in enumerate(xs):
        for k in range(int(np.floor(x)) + 1, n_other + 1):
            a = (k - x) / D
            if a >= 1.0:
                continue
            q[i] += (comb(n_other, k) * a**k * (1 - a)**(n_other - k)
                     * (D - n_other + x) / (D - k + x))
    return float(np.trapezoid(q, xs))


def ndd1_mean_wait(n_other: int, period_s: float, service_s: float) -> float:
    """
    Mean waiting time [s] of a frame that arrives at a FIFO queue fed by
    ``n_other`` independent periodic streams (period ``period_s``, deterministic
    service ``service_s``, uniformly random phases): the N*D/D/1 queue.

    Uses the Benes / Roberts-Virtamo virtual-waiting-time tail
        Q(x) = sum_{x<k<=n} C(n,k) ((k-x)/D)^k (1-(k-x)/D)^(n-k) (D-n+x)/(D-k+x),
    with x and D = period/service in service-time units, and W = s * int Q(x) dx.
    A frame of one of N streams sees the other N-1 streams (n_other = N-1).
    """
    return service_s * _ndd1_wait_units(int(n_other), round(period_s / service_s, 6))


def egress_wait(stats: dict, k: int, cap: float) -> float:
    """
    Mean waiting time [s] of a priority-k frame in a strict-priority,
    non-preemptive egress queue of capacity ``cap`` (bit/s).

    ``stats[p]`` holds per-priority offered load ('bps'), rate-weighted size
    sums ('ssf', 'sf'), and the number and load of periodic SV streams
    ('n_sv', 'bps_sv'). Service times are deterministic (fixed-size frames).

    * Residual work of the frame in service (any priority):
          R = sum_j rho_j * s_j / 2
    * Class with periodic SV streams: the SV streams interfere as an N*D/D/1
      queue (exact for independent random phases); the non-SV part of the load
      adds its residual, and higher-priority load dilates the wait:
          W_k = (W_NDD1(n_sv - 1) + R_nonSV) / (1 - sigma_{k+1})
    * Other classes (GOOSE, PTP, monitoring): Cobham's formula
          W_k = R / ((1 - sigma_{k+1}) (1 - sigma_k))
    with sigma_k the cumulative utilization of priority k and above.
    """
    def service(p):
        return (stats[p]['ssf'] / stats[p]['sf']) * 8 / cap if stats[p]['sf'] > 0 else 0.0

    rho = {p: stats[p]['bps'] / cap for p in range(8)}
    sigma_k = sum(rho[p] for p in range(k, 8))
    sigma_above = sum(rho[p] for p in range(k + 1, 8))
    if sigma_k >= 1.0:
        return float('inf')

    s_sv = SV_FRAME_BITS / cap
    r_sv = sum(stats[p]['bps_sv'] / cap * s_sv / 2 for p in range(8))
    r_all = sum(rho[p] * service(p) / 2 for p in range(8))
    r_non_sv = max(0.0, r_all - r_sv)

    n_sv = stats[k]['n_sv']
    if n_sv > 0:
        w_sv = ndd1_mean_wait(n_sv - 1, SV_PERIOD_S, s_sv)
        return (w_sv + r_non_sv) / (1.0 - sigma_above)
    return r_all / ((1.0 - sigma_above) * (1.0 - sigma_k))


def trunk_capacity_for(scenario: str) -> float:
    """Inter-switch trunk capacity [bps] in a scenario."""
    return CAPACITY_TRUNK_UPGRADED if scenario == 'upgraded' else CAPACITY_TRUNK


def port_capacity(p_label: str, scenario: str) -> float:
    """Capacity [bps] of the link behind an egress/trunk port node."""
    if 'Trunk' in p_label:
        return trunk_capacity_for(scenario)
    if p_label == 'L5_OutPort_BBP' and scenario in ('upgraded', 'bbp_only'):
        return CAPACITY_TRUNK_UPGRADED
    return CAPACITY_DEFAULT

def compute_weights(
    labels: list[str],
    A: np.ndarray,
    D: dict,
    sw_map: dict[str, str],
    all_devices: list[str],
    delta_t: float = 1000.0,
    scenario: str = 'base',
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Compute the normalised latency (W), jitter (J) and packet-loss (PL) matrices.

    All weight values are normalised by T_MAX so that W[i,j] = 1.0 represents
    a delay equal to the IEC 61850 3 ms deadline.

    Edge-weight models by layer:
      L1->L2  : Transmission delay = (avg_frame_size × 8) / CAPACITY_DEFAULT / T_MAX
      L2->L3  : Fixed processing delay = T_PROC_SWITCH / T_MAX
      L3->L4  : HOL-blocking weight from switch fabric utilisation
      L4->L5  : M/D/1 mean queue wait + M/M/1/K packet-loss probability (Altman–Jean-Marie)
      L4->L4  : Trunk cable serialization (S_avg,trunk / C_trunk)
      L5->L1  : Egress serialization (S_avg,m / C_m)

    Args:
        labels:      Ordered node label list from build_topology().
        A:           Adjacency matrix from build_topology().
        D:           Current demand tensor.
        sw_map:      Device-to-switch assignment.
        all_devices: Ordered device list.
        delta_t:     Current GOOSE retransmission interval [ms].
        scenario:    'base' or 'upgraded' (controls trunk link capacity).

    Returns:
        W:   Normalised latency matrix  (n_nodes × n_nodes).
        J:   Normalised jitter matrix   (n_nodes × n_nodes).
        PL:  Packet-loss probability    (n_nodes × n_nodes), range [0, 1].
        u_p: Dict {'SW1': {prio: util}, 'SW2': {prio: util}} -- fabric utilisation
             per switch per priority (used for HOL-blocking and reporting).
    """
    trunk_capacity = trunk_capacity_for(scenario)

    n_nodes = len(labels)
    W  = np.zeros((n_nodes, n_nodes), dtype=float)
    J  = np.zeros((n_nodes, n_nodes), dtype=float)
    PL = np.zeros((n_nodes, n_nodes), dtype=float)

    # -------------------------------------------------------------------------
    # Layer 1 -> Layer 2: weighted-average transmission delay per device
    # -------------------------------------------------------------------------
    goose_freq = 1000.0 / delta_t if delta_t > 0 else 0.0

    for d in all_devices:
        sum_size_freq = 0.0
        sum_freq      = 0.0

        for vlan, df in D.items():
            if not df.loc[d].any():
                continue  # Device sends no traffic on this VLAN

            # Resolve per-VLAN frame specs (GOOSE freq is dynamic)
            specs = get_frame_specs(vlan, src=d, goose_freq=goose_freq)

            sum_size_freq += specs['size'] * specs['freq']
            sum_freq      += specs['freq']

        if sum_freq > 0:
            s_avg  = sum_size_freq / sum_freq
            weight = ((s_avg * 8) / CAPACITY_DEFAULT) / T_MAX
            try:
                W[labels.index(f'L1_Equipment_{d}'),
                  labels.index(f'L2_InPort_{d}')] = weight
            except ValueError:
                pass

    # -------------------------------------------------------------------------
    # Layer 2 -> Layer 3: fixed switch processing delay
    # -------------------------------------------------------------------------
    proc_weight = T_PROC_SWITCH / T_MAX
    for i, label in enumerate(labels):
        if 'L2_InPort' in label:
            for j in np.where(A[i, :] == 1)[0]:
                if 'L3_VLAN' in labels[j]:
                    W[i, j] = proc_weight

    # Trunk ingress -> VLAN: same store-and-forward delay as device ingress.
    # When a frame arrives at the remote switch via the trunk it still goes
    # through the switch fabric before reaching the egress queue, incurring
    # the same T_PROC_SWITCH delay.  This edge only exists in A when there
    # is actual cross-switch traffic on that VLAN direction.
    for i, label in enumerate(labels):
        if 'L4_Trunk' in label:
            for j in np.where(A[i, :] == 1)[0]:
                if 'L3_VLAN' in labels[j]:
                    W[i, j] = proc_weight

    # -------------------------------------------------------------------------
    # Layer 3 -> Layer 4: HOL-blocking model (switch fabric contention)
    # -------------------------------------------------------------------------

    # Accumulate per-switch per-priority load (Mbps)
    sw1_devs = [d for d in all_devices if sw_map.get(d) == 'SW1']
    sw2_devs = [d for d in all_devices if sw_map.get(d) == 'SW2']
    lambda_sw = {'SW1': {p: 0.0 for p in range(8)},
                 'SW2': {p: 0.0 for p in range(8)}}

    for vlan, df in D.items():
        prio = get_vlan_priority(vlan)
        for src in df.index[df.any(axis=1)]:
            row     = df.loc[src]
            dests   = row[row > 0]
            if dests.empty:
                continue
            bw     = dests.iloc[0]  # All destinations receive the same bandwidth
            sw_src = sw_map[src]

            # Count how many copies the source switch must forward
            dests_sw1 = int(dests.index.isin(sw1_devs).sum())
            dests_sw2 = int(dests.index.isin(sw2_devs).sum())
            replicas  = dests_sw1 if sw_src == 'SW1' else dests_sw2
            # Add one replica for the trunk if any destination is on the remote switch
            if (sw_src == 'SW1' and dests_sw2 > 0) or (sw_src == 'SW2' and dests_sw1 > 0):
                replicas += 1

            lambda_sw[sw_src][prio] += bw * replicas

            # Load imposed on the remote switch by traffic arriving via the trunk
            sw_remote   = 'SW2' if sw_src == 'SW1' else 'SW1'
            dests_remote = dests_sw2 if sw_src == 'SW1' else dests_sw1
            if dests_remote > 0:
                lambda_sw[sw_remote][prio] += bw * dests_remote

    # Compute HOL weights and assign to L3->L4 edges
    u_p = {'SW1': {}, 'SW2': {}}
    for sw in ('SW1', 'SW2'):
        for p in range(7, -1, -1):
            # Strict-priority: accumulate all traffic at this and higher priorities
            load_accum = sum(lambda_sw[sw][k] for k in range(p, 8))
            u_val = load_accum / FABRIC_CAPACITY
            u_p[sw][p] = u_val

            if 0 < u_val < 1:
                # HOL-blocking delay from M/D/1 approximation
                hol_weight = (T_PROC_SWITCH * (u_val / (2 * (1 - u_val)))) / T_MAX
                prio_node  = f'L4_Priority_{sw}_Prio_{p}'
                if prio_node in labels:
                    idx_dst = labels.index(prio_node)
                    for idx_src in np.where(A[:, idx_dst] == 1)[0]:
                        W[idx_src, idx_dst] = hol_weight
                        # Jitter from variance of HOL blocking time
                        J[idx_src, idx_dst] = hol_weight * np.sqrt((2 - u_val) / u_val)

    # -------------------------------------------------------------------------
    # Layer 4 / Trunk -> Layer 5: M/M/1 queuing delay + M/M/1/K packet loss
    # -------------------------------------------------------------------------

    # Accumulate per-output-port traffic statistics broken down by priority
    port_stats: dict[str, dict[int, dict]] = {}

    for vlan, df in D.items():
        prio = get_vlan_priority(vlan)

        # Iterate only over non-zero (src, dst) pairs (sparse inner loop)
        nonzero = df.stack()
        nonzero = nonzero[nonzero > 0]
        for (src, dst), bw in nonzero.items():
            specs = get_frame_specs(vlan, src=src, goose_freq=goose_freq)
            ssf = specs['size'] * specs['freq']  # size × freq (for average frame size)
            sf  = specs['freq']
            sw_s, sw_d = sw_map[src], sw_map[dst]

            # Determine which output ports this flow traverses
            ports = []
            if sw_s == sw_d:
                # Intra-switch: single output port toward destination
                ports.append(f'L5_OutPort_{dst}')
            else:
                # Inter-switch: trunk port on source switch + output port on destination switch
                ports.append(f'L4_Trunk_{sw_s}_to_{sw_d}')
                ports.append(f'L5_OutPort_{dst}')

            is_sv = get_vlan_id(vlan) in SV_VLAN_IDS
            for p_label in ports:
                if p_label not in port_stats:
                    port_stats[p_label] = {p: {'bps': 0.0, 'ssf': 0.0, 'sf': 0.0,
                                               'n_sv': 0, 'bps_sv': 0.0}
                                           for p in range(8)}
                port_stats[p_label][prio]['bps'] += bw * 1e6
                port_stats[p_label][prio]['ssf'] += ssf
                port_stats[p_label][prio]['sf']  += sf
                if is_sv:
                    port_stats[p_label][prio]['n_sv']   += 1
                    port_stats[p_label][prio]['bps_sv'] += bw * 1e6

    # Compute queuing delay and PLR for each output port
    for p_label, stats in port_stats.items():
        if p_label not in labels:
            continue
        idx_dst = labels.index(p_label)

        # Link capacity by port type and scenario (see SCENARIOS)
        cap = port_capacity(p_label, scenario)

        # Process priorities from highest (7) to lowest (4) to model preemption
        for k in range(7, -1, -1):
            # Cumulative frame-size average across priority k and all higher priorities
            # (paper Eq. 16: L_p = sum(f_p*S_p, p>=k) / sum(f_p, p>=k))
            ssf_accum = sum(stats[p]['ssf'] for p in range(k, 8))
            sf_accum  = sum(stats[p]['sf']  for p in range(k, 8))
            if sf_accum <= 0:
                continue
            l_avg   = (ssf_accum / sf_accum) * 8   # bits

            # Strict Priority: cumulative load from priority k upward (paper Eq. 15)
            bps_accum = sum(stats[p]['bps'] for p in range(k, 8))
            rho       = bps_accum / cap

            w_queue = 0.0
            p_loss  = 0.0

            # Mean waiting time in the egress queue (egress_wait): N*D/D/1 for
            # the periodic SV class, Cobham non-preemptive priority otherwise.
            # This edge carries ONLY the waiting time; the serialization of the
            # frame onto the outgoing wire is the next edge (L5->L1 or trunk cable).
            if 0 < rho < 1.0:
                w_queue = egress_wait(stats, k, cap) / T_MAX
            elif rho >= 1.0:
                w_queue = 1.0  # Queue saturated -- delay exceeds T_MAX

            # M/M/1/K Packet Loss Rate (Altman–Jean-Marie, 1997)
            #   PLR = ρ^M (1-ρ) / (1 - ρ^(M+1))
            # Numerically stable branching (K = 28,149 causes overflow for naive pow):
            #   ρ < 1  -> ρ^K underflows to 0   -> PLR ~= 0 (large buffer rarely fills)
            #   ρ > 1  -> ρ^{-K} underflows to 0 -> PLR = (ρ-1)/ρ  (always < 100 %)
            if rho <= 0:
                p_loss = 0.0
            elif abs(rho - 1.0) < 1e-9:
                p_loss = 1.0 / (K_BUFFER + 1)
            elif rho < 1.0:
                rho_K = rho ** K_BUFFER
                if rho_K == 0.0:
                    # Underflow: buffer so large that loss is negligible
                    p_loss = 0.0
                else:
                    p_loss = float(np.clip(
                        rho_K * (1.0 - rho) / (1.0 - rho ** (K_BUFFER + 1)),
                        0.0, 1.0))
            else:  # rho > 1
                # ρ^{-K} safely underflows to 0; loss = (ρ-1)/ρ < 1
                rho_neg_K = rho ** (-K_BUFFER)
                p_loss = float(np.clip((1.0 - rho) / (rho_neg_K - rho), 0.0, 1.0))

            # Independent weight per priority level (paper Eq. 17 -- no accumulation)
            w_total = min(1.0, w_queue)

            # Assign weights to the L4 -> L5 edges of the correct priority
            for idx_src in np.where(A[:, idx_dst] == 1)[0]:
                if f'_Prio_{k}' in labels[idx_src]:
                    W[idx_src, idx_dst]  = w_total
                    PL[idx_src, idx_dst] = p_loss
                    if 0 < rho < 1.0:
                        # Jitter from variance of M/M/1 service time
                        J[idx_src, idx_dst] = w_queue * np.sqrt((2 - rho) / rho)

    # -------------------------------------------------------------------------
    # Intra-layer Trunk -> Trunk: physical cable transmission delay
    #
    # The edge L4_Trunk_SW1_to_SW2 -> L4_Trunk_SW2_to_SW1 (and the reverse)
    # represents the physical Ethernet cable connecting the two switches.
    # Its weight is the transmission delay of an average-sized frame over
    # the trunk capacity, normalised by T_MAX:
    #
    #   w_trunk = (S_avg / C_trunk) / T_MAX
    #
    # S_avg is estimated from all traffic that actually crosses the trunk
    # (flows where src and dst are on different switches), making the weight
    # dynamic and traffic-aware rather than a fixed constant.
    # -------------------------------------------------------------------------
    trunk_pairs = [
        ('L4_Trunk_SW1_to_SW2', 'L4_Trunk_SW2_to_SW1'),
        ('L4_Trunk_SW2_to_SW1', 'L4_Trunk_SW1_to_SW2'),
    ]
    for src_trunk, dst_trunk in trunk_pairs:
        if src_trunk in labels and dst_trunk in labels:
            # Collect frame-size and frequency stats for cross-trunk traffic
            ssf_total = 0.0
            sf_total  = 0.0

            # Use the already-accumulated port_stats for the source trunk port
            # (it holds exactly the traffic load going in that direction)
            if src_trunk in port_stats:
                for k in range(8):
                    data = port_stats[src_trunk][k]
                    ssf_total += data['ssf']
                    sf_total  += data['sf']

            if sf_total > 0:
                s_avg_trunk  = (ssf_total / sf_total) * 8   # bits
                w_trunk_link = (s_avg_trunk / trunk_capacity) / T_MAX
            else:
                # No cross-trunk traffic at this snapshot -- use a reference
                # 100-byte Ethernet frame as a conservative lower bound
                w_trunk_link = (100 * 8 / trunk_capacity) / T_MAX

            idx_s = labels.index(src_trunk)
            idx_d = labels.index(dst_trunk)
            W[idx_s, idx_d] = min(1.0, w_trunk_link)

    # -------------------------------------------------------------------------
    # Layer 5 -> Layer 1: serialization of the frames leaving egress port m
    #   w = (S_avg,m · 8 / C_m) / T_MAX, with S_avg,m the rate-weighted mean
    #   frame size of all traffic exiting that port (paper Table: L5 -> L1).
    # -------------------------------------------------------------------------
    for d in all_devices:
        if d == 'GPS':
            continue
        p_label = f'L5_OutPort_{d}'
        stats = port_stats.get(p_label)
        ssf_total = sum(stats[k]['ssf'] for k in range(8)) if stats else 0.0
        sf_total  = sum(stats[k]['sf']  for k in range(8)) if stats else 0.0
        s_avg_bits = (ssf_total / sf_total) * 8 if sf_total > 0 else 100 * 8
        cap = port_capacity(p_label, scenario)
        try:
            W[labels.index(p_label),
              labels.index(f'L1_Equipment_{d}')] = (s_avg_bits / cap) / T_MAX
        except ValueError:
            pass

    return W, J, PL, u_p


# =============================================================================
# 2. NETWORK ANALYSIS AND METRICS
# =============================================================================

def weighted_global_efficiency(G: nx.DiGraph, endpoints: list[int]) -> float:
    """
    Latora–Marchiori global efficiency over equipment (L1) node pairs:

        E = 1 / (n (n-1)) · Σ_{i≠j ∈ L1} 1 / d(i, j)

    where d(i, j) is the minimum-latency directed path length in ms through
    the switching fabric and unreachable pairs contribute 0. Restricting the
    sum to L1 endpoints keeps E an end-to-end quantity: internal pipeline
    nodes (VLAN, queue) have sub-µs edges that would otherwise dominate.

    Args:
        G:         Directed graph whose 'weight' attribute is the edge delay [s].
        endpoints: Indices of the L1 equipment nodes (the n in the normaliser,
                   also for pairs whose node was removed from G).
    """
    n = len(endpoints)
    if n < 2:
        return 0.0
    targets = set(endpoints)
    total = 0.0
    for src in endpoints:
        if src not in G:
            continue
        dists = nx.single_source_dijkstra_path_length(G, src, weight='weight')
        for dst, d in dists.items():
            if dst != src and dst in targets and d > 0:
                total += 1.0 / (d * 1e3)  # 1/ms
    return total / (n * (n - 1))


def analyze_network(
    n_bays: int,
    labels: list[str],
    W: np.ndarray,
    J: np.ndarray,
    PL: np.ndarray,
    u_p: dict,
    use_approx: bool = False,
    compute_vulnerability: bool = False,
) -> dict:
    """
    Derive all graph-theoretic and QoS metrics for one simulation snapshot.

    Complex network metrics (require an undirected symmetrised graph):
      - Fiedler_Lambda2     : algebraic connectivity (robustness to partitioning)
      - Global_Efficiency   : Latora–Marchiori efficiency (reachability)
      - Load_Gini           : Gini coefficient of edge weights (load imbalance)
      - Betweenness         : normalised weighted betweenness centrality (per node)
      - Vulnerability_Index : efficiency drop on node removal (steady state only)

    QoS metrics (per IEEE 802.1Q priority class P4/P5/P6/P7):
      - Latency_Avg_P{p} / Latency_Max_P{p} [ms]
      - Jitter_Avg_P{p}  [ms]
      - Loss_Avg_P{p} / Loss_Max_P{p} / Loss_P95_P{p}  [%]

    Args:
        n_bays:                Number of bays (stored in result dict).
        labels:                Node label list from build_topology().
        W:                     Normalised latency matrix.
        J:                     Normalised jitter matrix.
        PL:                    Packet-loss probability matrix.
        u_p:                   Fabric utilisation dict from compute_weights().
        use_approx:            If True, use approximate betweenness (k-sampling)
                               to reduce computation time for large graphs.
        compute_vulnerability: If True, compute the Node Vulnerability Index
                               for the top-20 nodes by betweenness centrality.
                               Should be True only at steady state (expensive).

    Returns:
        metrics: Dict of scalar metrics + 'centrality_dict' and 'vuln_dict'
                 (the latter two are popped by run_simulation before storing).
    """
    # -------------------------------------------------------------------------
    # Build graph objects
    # -------------------------------------------------------------------------
    W_lat     = W * T_MAX
    G_latency = nx.from_numpy_array(W_lat, create_using=nx.DiGraph)

    lambda_2          = 0.0
    global_efficiency = 0.0
    bet_cen_labeled   = {}
    vuln_dict         = {}

    try:
        # Symmetrise W for undirected-only metrics (algebraic connectivity, efficiency)
        W_sym = (W + W.T) / 2
        # Convert delay weights to conductance: lower latency = stronger connection.
        # W encodes normalised delay (0..1), so conductance = 1/W. Edges with W=0
        # (no link) stay at 0 conductance. Clip to [1e-6, 1] before inverting to
        # avoid extreme conductance ratios that cause ARPACK non-convergence.
        W_clipped  = np.where(W_sym > 0, np.clip(W_sym, 1e-6, 1.0), 0.0)
        W_conduct  = np.divide(1.0, W_clipped, out=np.zeros_like(W_clipped), where=W_clipped > 0)
        G_sym = nx.from_numpy_array(W_conduct, create_using=nx.Graph())

        # Fiedler eigenvalue -- use tracemin_lu (more stable than lanczos for
        # graphs with high conductance variance).
        lambda_2 = nx.algebraic_connectivity(G_sym, weight='weight', method='tracemin_lu')

        # Latency-weighted global efficiency on the directed forwarding graph
        # (nx.global_efficiency ignores weights, so it is computed explicitly).
        l1_nodes = [i for i, lbl in enumerate(labels) if lbl.startswith('L1_')]
        global_efficiency = weighted_global_efficiency(G_latency, l1_nodes)

        # Exact weighted betweenness centrality on the directed graph
        # (graphs have < 400 nodes, so sampling is unnecessary).
        bet_cen = nx.betweenness_centrality(
            G_latency, weight='weight', normalized=True
        )
        bet_cen_labeled = {labels[k]: v for k, v in bet_cen.items()}

        # Node Vulnerability Index: V_i = (E(G) - E(G \ {i})) / E(G)
        # Computed only at steady state for the top-20 most central nodes.
        if compute_vulnerability and global_efficiency > 0 and bet_cen_labeled:
            top_nodes = sorted(
                bet_cen_labeled, key=bet_cen_labeled.get, reverse=True
            )[:20]
            for node_label in top_nodes:
                if node_label not in labels:
                    continue
                G_copy = G_latency.copy()
                G_copy.remove_node(labels.index(node_label))
                # Same endpoint set as the intact graph: a removed device simply
                # contributes 0, so removal is never rewarded by a smaller n.
                e_removed = weighted_global_efficiency(G_copy, l1_nodes)
                vuln_dict[node_label] = max(0.0,
                    (global_efficiency - e_removed) / global_efficiency)

    except Exception as exc:
        print(f"Warning -- graph metric computation failed: {exc}")

    # -------------------------------------------------------------------------
    # Load Gini coefficient (inequality of edge weights)
    # -------------------------------------------------------------------------
    edge_weights = W[W > 0].flatten()
    if len(edge_weights) > 1:
        w_sorted = np.sort(edge_weights)
        n_w      = len(w_sorted)
        gini = (2 * np.sum(np.arange(1, n_w + 1) * w_sorted)
                / (n_w * np.sum(w_sorted))) - (n_w + 1) / n_w
    else:
        gini = 0.0

    # -------------------------------------------------------------------------
    # Per-priority QoS statistics (collected from L4 priority-queue edges)
    # -------------------------------------------------------------------------
    stats_prio = {p: {'lat': [], 'jit': [], 'loss': []} for p in [4, 5, 6, 7]}

    for i, label in enumerate(labels):
        if 'L4_Priority_' in label and 'Trunk' not in label:
            try:
                p = int(label.split('_')[-1])
                for j in np.where(W[i, :] > 0)[0]:
                    stats_prio[p]['lat'].append(W[i, j]  * T_MAX * 1000)  # ms
                    stats_prio[p]['jit'].append(J[i, j]  * T_MAX * 1000)  # ms
                    stats_prio[p]['loss'].append(PL[i, j] * 100)           # %
            except (ValueError, KeyError):
                pass

    # -------------------------------------------------------------------------
    # Assemble result dictionary
    # -------------------------------------------------------------------------
    lat_max_p6 = np.max(stats_prio[6]['lat']) if stats_prio[6]['lat'] else 0.0
    lat_max_p4 = np.max(stats_prio[4]['lat']) if stats_prio[4]['lat'] else 0.0

    metrics: dict = {
        'N_Bays':             n_bays,
        'Total_Nodes':        len(labels),
        'Fiedler_Lambda2':    lambda_2,
        'Global_Efficiency':  global_efficiency,
        'Load_Gini':          gini,
        'centrality_dict':    bet_cen_labeled,   # popped by run_simulation
        'vuln_dict':          vuln_dict,          # popped by run_simulation
        'Max_Util_SW1': max(u_p['SW1'].values()) if u_p.get('SW1') else 0.0,
        'Max_Util_SW2': max(u_p['SW2'].values()) if u_p.get('SW2') else 0.0,
        # IEC 61850 violation flag: 1 if any P6 or P4 latency reaches or exceeds 3 ms
        # (paper: w >= 1 signals a violation; W is clamped to 1.0 at saturation)
        'IEC_Violation': 1 if (lat_max_p6 >= 3.0 or lat_max_p4 >= 3.0) else 0,
    }

    for p in [4, 5, 6, 7]:
        lats   = stats_prio[p]['lat']
        losses = stats_prio[p]['loss']
        metrics[f'Latency_Avg_P{p}'] = float(np.mean(lats))              if lats   else 0.0
        metrics[f'Latency_Max_P{p}'] = float(np.max(lats))               if lats   else 0.0
        metrics[f'Jitter_Avg_P{p}']  = float(np.mean(stats_prio[p]['jit'])) if lats else 0.0
        metrics[f'Loss_Avg_P{p}']    = float(np.mean(losses))            if losses else 0.0
        metrics[f'Loss_Max_P{p}']    = float(np.max(losses))             if losses else 0.0
        metrics[f'Loss_P95_P{p}']    = float(np.percentile(losses, 95))  if losses else 0.0

    return metrics


# =============================================================================
# 2b. END-TO-END FLOW LATENCY (path sum of edge weights)
# =============================================================================

def flow_path(src: str, dst: str, vlan: str, sw_map: dict[str, str]) -> list[str]:
    """
    Node sequence traversed by one flow through the 5-layer graph.

    Intra-switch:  L1 -> L2 -> L3 -> L4(prio) -> L5 -> L1
    Inter-switch:  L1 -> L2 -> L3@Sa -> L4(prio@Sa) -> Trunk(Sa->Sb) -> Trunk(Sb->Sa)
                   -> L3@Sb -> L4(prio@Sb) -> L5 -> L1
    """
    prio = get_vlan_priority(vlan)
    sw_s, sw_d = sw_map[src], sw_map[dst]
    path = [f'L1_Equipment_{src}', f'L2_InPort_{src}',
            f'{vlan}_{sw_s}', f'L4_Priority_{sw_s}_Prio_{prio}']
    if sw_s != sw_d:
        path += [f'L4_Trunk_{sw_s}_to_{sw_d}', f'L4_Trunk_{sw_d}_to_{sw_s}',
                 f'{vlan}_{sw_d}', f'L4_Priority_{sw_d}_Prio_{prio}']
    path += [f'L5_OutPort_{dst}', f'L1_Equipment_{dst}']
    return path


def fluid_overload(rho: float, cap_bps: float,
                   window_s: float = OBS_WINDOW_S,
                   buffer_bytes: float = PORT_BUFFER_BYTES) -> tuple[float, float, float]:
    """
    Deterministic fluid model of an output queue overloaded (ρ > 1) from t = 0.

    The backlog grows at (ρ-1)·C, so the waiting time of a frame arriving at t
    is (ρ-1)·t until the buffer B fills at t_fill = 8B / ((ρ-1)·C); afterwards
    the wait is capped at 8B/C and a fraction (ρ-1)/ρ of arrivals is dropped.

    Returns (mean wait [s], max wait [s], mean loss ratio) over [0, window_s].
    """
    if rho <= 1.0:
        return 0.0, 0.0, 0.0
    slope  = rho - 1.0
    d_cap  = 8.0 * buffer_bytes / cap_bps
    t_fill = d_cap / slope
    if window_s <= t_fill:
        return slope * window_s / 2.0, slope * window_s, 0.0
    mean = (d_cap * t_fill / 2.0 + d_cap * (window_s - t_fill)) / window_s
    loss = (rho - 1.0) / rho * (window_s - t_fill) / window_s
    return mean, d_cap, loss


def _port_cumulative_rho(
    D: dict,
    sw_map: dict[str, str],
    scenario: str,
) -> dict[tuple[str, int], tuple[float, float]]:
    """
    Strict-priority cumulative load of every egress/trunk port:
    {(port_label, prio): (ρ_p, C_port)}, ρ_p = Σ_{k≥p} Λ_k / C_port.
    Same port and capacity rules as compute_weights().
    """
    load: dict[str, dict[int, float]] = {}
    for vlan, df in D.items():
        prio = get_vlan_priority(vlan)
        nonzero = df.stack()
        for (src, dst), bw in nonzero[nonzero > 0].items():
            sw_s, sw_d = sw_map[src], sw_map[dst]
            ports = [f'L5_OutPort_{dst}']
            if sw_s != sw_d:
                ports.append(f'L4_Trunk_{sw_s}_to_{sw_d}')
            for p_label in ports:
                load.setdefault(p_label, {k: 0.0 for k in range(8)})[prio] += bw * 1e6
    out = {}
    for p_label, by_prio in load.items():
        cap = port_capacity(p_label, scenario)
        for p in range(8):
            out[(p_label, p)] = (sum(by_prio[k] for k in range(p, 8)) / cap, cap)
    return out


def _serialization_link_capacity(a: str, b: str, scenario: str) -> float | None:
    """Capacity [bps] of a wire edge a->b (device link or trunk cable), else None."""
    if a.startswith('L1_') and b.startswith('L2_'):
        return CAPACITY_DEFAULT
    if (a.startswith('L4_Trunk_') and b.startswith('L4_Trunk_')) or \
            (a.startswith('L5_') and b.startswith('L1_')):
        return port_capacity(a, scenario)
    return None


def _device_sv_streams(D: dict) -> dict[str, int]:
    """Number of periodic SV streams (frames per sample, multicast counted once)
    published by each device."""
    n = {}
    for vlan, df in D.items():
        if get_vlan_id(vlan) in SV_VLAN_IDS:
            for src in df.index[df.any(axis=1)]:
                n[src] = n.get(src, 0) + 1
    return n


def _synchronized_burst_waits(
    D: dict, sw_map: dict[str, str], scenario: str, resources: str = 'egress',
    burst_vlans: set[str] | None = None,
) -> dict[tuple[str, str, str], float]:
    """
    Extra waiting time [s] of each burst GOOSE flow caused by the other frames
    of the 50BF cascade that are released at the same instant.

    A frame is one (VLAN, source) pair; multicast copies share it until the
    egress port. At a shared resource a frame waits for every simultaneous
    frame of higher priority and, in random order, for half of those of equal
    priority (non-preemptive, strict priority).

    ``resources`` selects where simultaneity is assumed:
      'publisher' -- the publisher link only: the frames a device activates
                     together are released exactly together (FIFO device link).
      'egress'    -- publisher link + destination egress port (default): frames
                     of different publishers triggered by the same event
                     converge on the egress port of a common subscriber.
      'network'   -- additionally the serial switch fabric and the trunk
                     (upper bound: assumes perfect synchronism network-wide).

    ``burst_vlans`` restricts the released frames to the VLANs of the event
    (None = every burst-capable VLAN, i.e. an event in all bays).

    Returns {(vlan, src, dst): wait_s}.
    """
    frames = []   # (vlan, src, prio, bits, set(dst))
    for vlan, df in D.items():
        if not is_burst_vlan(vlan) or (burst_vlans is not None and vlan not in burst_vlans):
            continue
        prio = get_vlan_priority(vlan)
        for src in df.index[df.any(axis=1)]:
            dsts = set(df.columns[df.loc[src] > 0])
            frames.append((vlan, src, prio, get_frame_specs(vlan, src=src)['size'] * 8, dsts))

    def enters(fr, sw):
        """True if frame ``fr`` is processed by the fabric of switch ``sw``."""
        _, src, _, _, dsts = fr
        return sw_map[src] == sw or any(sw_map[d] == sw for d in dsts)

    def crosses(fr, sa, sb):
        _, src, _, _, dsts = fr
        return sw_map[src] == sa and any(sw_map[d] == sb for d in dsts)

    def wait_among(me, members, service):
        w = 0.0
        for g in members:
            if g is me:
                continue
            if g[2] > me[2]:
                w += service(g)
            elif g[2] == me[2]:
                w += 0.5 * service(g)
        return w

    out = {}
    for fr in frames:
        vlan, src, prio, bits, dsts = fr
        sw_s = sw_map[src]
        base = wait_among(fr, [g for g in frames if g[1] == src],
                          lambda g: g[3] / CAPACITY_DEFAULT)
        if resources == 'network':
            base += wait_among(fr, [g for g in frames if enters(g, sw_s)],
                               lambda g: T_PROC_SWITCH)
        for dst in dsts:
            w = base
            sw_d = sw_map[dst]
            if resources == 'network' and sw_d != sw_s:
                cap_t = trunk_capacity_for(scenario)
                w += wait_among(fr, [g for g in frames if crosses(g, sw_s, sw_d)],
                                lambda g: g[3] / cap_t)
                w += wait_among(fr, [g for g in frames if enters(g, sw_d)],
                                lambda g: T_PROC_SWITCH)
            if resources in ('egress', 'network'):
                cap_e = port_capacity(f'L5_OutPort_{dst}', scenario)
                w += wait_among(fr, [g for g in frames if dst in g[4]],
                                lambda g: g[3] / cap_e)
            out[(vlan, src, dst)] = w
    return out


def compute_e2e_metrics(
    labels: list[str],
    W: np.ndarray,
    PL: np.ndarray,
    D: dict,
    sw_map: dict[str, str],
    scenario: str = 'base',
    window_s: float = OBS_WINDOW_S,
    D_window: dict | None = None,
    synchronized_burst: bool = False,
    burst_vlans: set[str] | None = None,
) -> dict:
    """
    End-to-end latency and loss of every SV and GOOSE flow.

    The latency of a flow is the sum of the edge delays along its path
    (serialization + processing + HOL + queuing + serialization ...), which is
    the quantity measured by the DES. The flow PLR is 1 - prod(1 - p_hop).

    Queue edges whose cumulative load is ρ ≥ 1 have no steady state; for those
    hops the waiting time and loss are taken from the fluid overload model over
    the observation window (fluid_overload), so the reported delay of a
    saturated flow is its mean over [0, window_s] instead of the clamp T_MAX.
    The fluid load is taken from D_window (the demand averaged over the
    window, i.e. steady state) when given: a GOOSE burst lasts milliseconds
    and must not be treated as if it persisted for the whole window.

    Publisher link: each device transmits its frames FIFO on its own link, so a
    frame also waits for the device's periodic SV streams (N*D/D/1). When
    ``synchronized_burst`` is True (burst snapshots), burst GOOSE frames also
    wait for the other frames released at the same instant
    (_synchronized_burst_waits).

    Returns scalar metrics in µs / %:
      E2E_Avg_{SV,GOOSE}_us, E2E_Max_{SV,GOOSE}_us,
      E2E_PLR_Avg_{SV,GOOSE}, E2E_PLR_Max_{SV,GOOSE}, E2E_{SV,GOOSE}_Compliance
    where *_Compliance is the fraction of flows with delay < T_MAX, and the
    flow-constrained graph indicators, computed on the actual paths of the
    time-critical flows F instead of shortest paths:
      Flow_Efficiency   E_F = (1/|F|) sum_f 1/D_f                [ms^-1]
      qdc_dict          QDC(v) = sum_f w_v(f) / sum_f sum_u w_u(f)
                        (share of the flows' queuing wait accrued at node v)
      flow_vuln_dict    V_F(v) = sum_{f through v} (1/D_f) / sum_f (1/D_f)
                        (share of E_F lost if node v fails)
      QDC_Top_Node / QDC_Top_Share  -- the node with the largest QDC.
    """
    idx = {lbl: i for i, lbl in enumerate(labels)}
    classes = {'SV': SV_VLAN_IDS, 'GOOSE': GOOSE_VLAN_IDS}
    delays = {c: [] for c in classes}
    losses = {c: [] for c in classes}
    port_rho = _port_cumulative_rho(D, sw_map, scenario)
    fluid_rho = (_port_cumulative_rho(D_window, sw_map, scenario)
                 if D_window is not None else port_rho)
    dev_sv = _device_sv_streams(D)
    s_dev_sv = SV_FRAME_BITS / CAPACITY_DEFAULT
    burst_wait = (_synchronized_burst_waits(D, sw_map, scenario, burst_vlans=burst_vlans)
                  if synchronized_burst else {})
    event_goose = []   # delays [us] of GOOSE flows that belong to the event

    queue_wait_at: dict[str, float] = {}   # node -> summed queuing wait of critical flows [s]
    flows = []                            # (path nodes, end-to-end delay [s])

    def add_wait(node, w):
        if w > 0:
            queue_wait_at[node] = queue_wait_at.get(node, 0.0) + w

    for vlan, df in D.items():
        vid = get_vlan_id(vlan)
        cls = next((c for c, ids in classes.items() if vid in ids), None)
        if cls is None:
            continue
        nonzero = df.stack()
        prio = get_vlan_priority(vlan)
        for (src, dst), bw in nonzero[nonzero > 0].items():
            path = flow_path(src, dst, vlan, sw_map)
            frame_bits = get_frame_specs(vlan, src=src)['size'] * 8
            # Publisher-link FIFO: an SV frame sees the device's other SV
            # streams; any other frame arrives at a random time and sees all.
            m = dev_sv.get(src, 0)
            w_pub = ndd1_mean_wait(m - 1 if cls == 'SV' else m, SV_PERIOD_S, s_dev_sv)
            w_pub += burst_wait.get((vlan, src, dst), 0.0)
            add_wait(path[0], w_pub)
            d_s, keep = w_pub, 1.0
            for a, b in zip(path, path[1:]):
                i, j = idx[a], idx[b]
                rho, cap = port_rho.get((b, prio), (0.0, 1.0))
                link_cap = _serialization_link_capacity(a, b, scenario)
                if link_cap is not None:
                    # Wire edge: serialize THIS flow's frame (the edge weight in W
                    # uses the port's rate-weighted mean frame for graph metrics).
                    d_s  += frame_bits / link_cap
                    keep *= 1.0 - PL[i, j]
                elif a.startswith('L4_Priority_') and rho >= 1.0:
                    rho_w, cap_w = fluid_rho.get((b, prio), (rho, cap))
                    wait, _, loss = fluid_overload(rho_w, cap_w, window_s)
                    d_s  += wait
                    keep *= 1.0 - loss
                    add_wait(b, wait)
                else:
                    hop = W[i, j] * T_MAX
                    d_s  += hop
                    keep *= 1.0 - PL[i, j]
                    if not (a.startswith('L2_') or b.startswith('L3_')):
                        add_wait(b, hop)      # HOL and egress/trunk queue waits
            delays[cls].append(d_s * 1e6)
            losses[cls].append((1.0 - keep) * 100)
            flows.append((path, d_s))
            if cls == 'GOOSE' and (burst_vlans is None or vlan in burst_vlans):
                event_goose.append(d_s * 1e6)

    out = {}
    ev = np.array(event_goose)
    out['E2E_Avg_GOOSE_Event_us'] = float(ev.mean()) if ev.size else 0.0
    out['E2E_Max_GOOSE_Event_us'] = float(ev.max()) if ev.size else 0.0
    # Flow-constrained indicators over all time-critical (SV + GOOSE) flows
    inv = np.array([1.0 / (d * 1e3) for _, d in flows]) if flows else np.array([])
    out['Flow_Efficiency'] = float(inv.mean()) if inv.size else 0.0      # ms^-1
    total_q = sum(queue_wait_at.values())
    qdc = {n: w / total_q for n, w in queue_wait_at.items()} if total_q > 0 else {}
    vf = {}
    if inv.size and inv.sum() > 0:
        for (path, _), e in zip(flows, inv):
            for node in set(path):
                vf[node] = vf.get(node, 0.0) + e
        vf = {n: v / inv.sum() for n, v in vf.items()}
    out['qdc_dict'] = qdc          # queuing-delay centrality (popped by caller)
    out['flow_vuln_dict'] = vf     # flow vulnerability (popped by caller)
    top = max(qdc, key=qdc.get) if qdc else ''
    out['QDC_Top_Node'] = top
    out['QDC_Top_Share'] = qdc.get(top, 0.0)
    for cls in classes:
        d = np.array(delays[cls])
        l = np.array(losses[cls])
        out[f'E2E_Avg_{cls}_us']     = float(d.mean()) if d.size else 0.0
        out[f'E2E_Max_{cls}_us']     = float(d.max())  if d.size else 0.0
        out[f'E2E_PLR_Avg_{cls}']    = float(l.mean()) if l.size else 0.0
        out[f'E2E_PLR_Max_{cls}']    = float(l.max())  if l.size else 0.0
        out[f'E2E_{cls}_Compliance'] = float((d < T_MAX * 1e6).mean()) if d.size else 1.0
    return out


# =============================================================================
# 3. CENTRALITY BURST RATIO
# =============================================================================

def compute_cbr(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Centrality Burst Ratio (CBR) for each graph node.

    CBR_i = C_i(burst peak) / C_i(steady state)

    A high CBR indicates that a node becomes a disproportionate bottleneck
    specifically during a GOOSE fault burst, even if it is not critical under
    normal operating conditions.  This metric helps prioritise redundancy
    investments.

    Args:
        df: Main simulation DataFrame (long format, one row per node per snapshot).

    Returns:
        DataFrame with columns: N_Bays, Scenario, Node_Label, CBR.
        Sorted by (N_Bays, Scenario, CBR descending).
    """
    # Steady-state baseline: average centrality at deltat = 1000 ms
    steady = (df[df['Delta_T_ms'] == 1000.0]
              .groupby(['N_Bays', 'Scenario', 'Node_Label'])['Centrality']
              .mean()
              .reset_index()
              .rename(columns={'Centrality': 'C_steady'}))

    # Burst peak: average centrality at deltat = 4 ms (first GOOSE retransmission)
    burst = (df[df['Delta_T_ms'] == 4.0]
             .groupby(['N_Bays', 'Scenario', 'Node_Label'])['Centrality']
             .mean()
             .reset_index()
             .rename(columns={'Centrality': 'C_burst'}))

    cbr_df = pd.merge(steady, burst, on=['N_Bays', 'Scenario', 'Node_Label'], how='inner')
    # Exclude nodes with zero steady-state centrality (undefined ratio)
    cbr_df = cbr_df[cbr_df['C_steady'] > 0].copy()
    cbr_df['CBR'] = cbr_df['C_burst'] / cbr_df['C_steady']

    return cbr_df.sort_values(
        ['N_Bays', 'Scenario', 'CBR'], ascending=[True, True, False]
    )


# =============================================================================
# 4. RECOVERY TIME
# =============================================================================

def compute_recovery_time(
    df: pd.DataFrame,
    threshold_factor: float = 1.1,
) -> pd.DataFrame:
    """
    Estimate the time for P6 (GOOSE Protection) latency to recover after a burst.

    Recovery is defined as the first snapshot after the fault at which the
    maximum P6 latency falls back to within (threshold_factor × steady_state).
    A threshold factor of 1.1 corresponds to a 10 % tolerance above steady state.

    Args:
        df:               Main simulation DataFrame.
        threshold_factor: Multiplier above steady-state latency that defines
                          the recovery boundary (default 1.1 = 10 % tolerance).

    Returns:
        DataFrame with columns: N_Bays, Scenario, Steady_Lat_P6_ms,
        Recovery_Time_ms (NaN if full recovery was not observed).
    """
    df_snap = df.drop(columns=['Node_Label', 'Centrality']).drop_duplicates().copy()
    records = []

    for (n_bays, scenario), group in df_snap.groupby(['N_Bays', 'Scenario']):
        steady_lat = group[group['Delta_T_ms'] == 1000.0]['Latency_Max_P6'].mean()
        threshold  = steady_lat * threshold_factor

        # Burst region: exclude the initial pre-fault steady-state snapshot (t = 0)
        burst_group = group[group['Snapshot_Time_ms'] > 0].sort_values('Snapshot_Time_ms')
        recovered   = burst_group[burst_group['Latency_Max_P6'] <= threshold]

        recovery_time = recovered['Snapshot_Time_ms'].min() if not recovered.empty else float('nan')
        records.append({
            'N_Bays':            n_bays,
            'Scenario':          scenario,
            'Steady_Lat_P6_ms':  steady_lat,
            'Recovery_Time_ms':  recovery_time,
        })

    return pd.DataFrame(records)
