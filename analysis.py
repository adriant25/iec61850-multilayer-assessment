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

import numpy as np
import pandas as pd
import networkx as nx

from config import (
    T_MAX, CAPACITY_DEFAULT, CAPACITY_TRUNK, CAPACITY_TRUNK_UPGRADED,
    FABRIC_CAPACITY, T_PROC_SWITCH, K_BUFFER,
    SV_VLAN_IDS, GOOSE_VLAN_IDS,
    get_vlan_priority, get_vlan_id, get_frame_specs,
)

# =============================================================================
# 1. PHYSICAL WEIGHT COMPUTATION
# =============================================================================

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
    trunk_capacity = CAPACITY_TRUNK_UPGRADED if scenario == 'upgraded' else CAPACITY_TRUNK

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

            for p_label in ports:
                if p_label not in port_stats:
                    port_stats[p_label] = {p: {'bps': 0.0, 'ssf': 0.0, 'sf': 0.0}
                                           for p in range(8)}
                port_stats[p_label][prio]['bps'] += bw * 1e6
                port_stats[p_label][prio]['ssf'] += ssf
                port_stats[p_label][prio]['sf']  += sf

    # Compute queuing delay and PLR for each output port
    for p_label, stats in port_stats.items():
        if p_label not in labels:
            continue
        idx_dst = labels.index(p_label)

        # Select link capacity based on port type and scenario.
        # Base scenario: all links at 100 Mbps -- BBP saturates at N>=9 (expected violation).
        # Upgraded scenario: trunk links and the BBP uplink both raised to 1 Gbps,
        # matching the paper's case study ("1 Gbps links for both trunks and BBP connection").
        is_trunk = 'Trunk' in p_label
        is_bbp   = p_label == 'L5_OutPort_BBP'
        if (is_trunk or is_bbp) and scenario == 'upgraded':
            cap = trunk_capacity     # Upgraded: 1 Gbps trunk or BBP link
        else:
            cap = CAPACITY_DEFAULT   # Base: 100 Mbps for all links

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

            # Mean waiting time in queue, M/G/1 Pollaczek-Khinchine with
            # deterministic service (E[S^2] = E[S]^2):  W = 0.5 · (L/C) · ρ/(1-ρ).
            # This edge carries ONLY the waiting time; the serialization of the
            # frame onto the outgoing wire is the next edge (L5->L1 or trunk cable).
            if 0 < rho < 1.0:
                delay_s = 0.5 * (l_avg / cap) * rho / (1 - rho)
                w_queue = delay_s / T_MAX
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
        cap = (trunk_capacity if (scenario == 'upgraded' and d == 'BBP')
               else CAPACITY_DEFAULT)
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


def compute_e2e_metrics(
    labels: list[str],
    W: np.ndarray,
    PL: np.ndarray,
    D: dict,
    sw_map: dict[str, str],
) -> dict:
    """
    End-to-end latency and loss of every SV and GOOSE flow.

    The latency of a flow is the sum of the edge delays along its path
    (serialization + processing + HOL + queuing + serialization ...), which is
    the quantity measured by the DES. The flow PLR is 1 - prod(1 - p_hop).

    Returns scalar metrics in µs / %:
      E2E_Avg_{SV,GOOSE}_us, E2E_Max_{SV,GOOSE}_us,
      E2E_PLR_Avg_{SV,GOOSE}, E2E_PLR_Max_{SV,GOOSE}, E2E_{SV,GOOSE}_Compliance
    where *_Compliance is the fraction of flows with delay < T_MAX.
    """
    idx = {lbl: i for i, lbl in enumerate(labels)}
    classes = {'SV': SV_VLAN_IDS, 'GOOSE': GOOSE_VLAN_IDS}
    delays = {c: [] for c in classes}
    losses = {c: [] for c in classes}

    for vlan, df in D.items():
        vid = get_vlan_id(vlan)
        cls = next((c for c, ids in classes.items() if vid in ids), None)
        if cls is None:
            continue
        nonzero = df.stack()
        for (src, dst), bw in nonzero[nonzero > 0].items():
            path = flow_path(src, dst, vlan, sw_map)
            d_s, keep = 0.0, 1.0
            for a, b in zip(path, path[1:]):
                i, j = idx[a], idx[b]
                d_s  += W[i, j] * T_MAX
                keep *= 1.0 - PL[i, j]
            delays[cls].append(d_s * 1e6)
            losses[cls].append((1.0 - keep) * 100)

    out = {}
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
