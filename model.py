# -*- coding: utf-8 -*-
"""
model.py — Substation device model and 5-layer graph topology.

IEC 61850 Digital Substation — Complex Network Scalability Simulation
----------------------------------------------------------------------
This module is responsible for:
  1. Generating the list of IEC 61850 devices and their switch assignments.
  2. Building the traffic demand tensor D — a dict of DataFrames, one per VLAN,
     encoding who sends how much bandwidth to whom.
  3. Efficiently updating only the GOOSE VLANs during burst retransmission.
  4. Constructing the 5-layer directed graph adjacency matrix A:
       L1 (Equipment) → L2 (InPort) → L3 (VLAN) → L4 (Priority/Trunk)
                      → L5 (OutPort) → L1

All functions are pure (no global state mutations) and re-entrant.
"""

import numpy as np
import pandas as pd

import config
from config import (TRAFFIC_SPECS, GOOSE_VLAN_IDS, V5_WIRE_BYTES, get_vlan_priority, get_mbps,
                    get_frame_specs)

# =============================================================================
# 1. GOOSE BURST TIMELINE
# =============================================================================

def generate_goose_burst_timeline() -> tuple[list[float], list[float]]:
    """
    Build the time-snapshot and inter-snapshot interval vectors for one
    complete GOOSE burst event as defined in IEC 61850-8-1.

    The burst pattern starts at steady state (Δt = 1000 ms), fires two
    immediate retransmissions at 4 ms, then doubles the interval
    (8 → 16 → 32 → … ms) until Δt reaches 1000 ms again.

    Returns:
        t_snapshots:  Cumulative time instants [ms] at which the simulation
                      takes a measurement (length N+1).
        delta_t_list: Inter-snapshot intervals [ms] corresponding to the
                      GOOSE retransmission period at each snapshot (length N+1).
    """
    # Start with one steady-state snapshot before the fault
    delta_t_list = [1000.0]

    # Two immediate retransmissions at t=4 ms (fault detection phase)
    delta_t_list.extend([4.0, 4.0])

    # Exponential back-off: 8, 16, 32, … ms until Δt ≥ 1000 ms
    next_delta = 8.0
    while next_delta < 1000.0:
        delta_t_list.append(next_delta)
        next_delta *= 2

    # Final steady-state snapshot to measure full recovery
    delta_t_list.append(1000.0)

    # Convert interval list to cumulative timestamps
    t_snapshots = [0.0]
    for i in range(len(delta_t_list) - 1):
        t_snapshots.append(t_snapshots[i] + delta_t_list[i])

    return t_snapshots, delta_t_list


# =============================================================================
# 2. DEVICE GENERATION
# =============================================================================

def generate_devices(n_bays: int) -> tuple[list[str], dict[str, str]]:
    """
    Generate the ordered device list and switch assignment map.

    Global devices (always on SW1):
      - GPS  : PTP grandmaster clock
      - BBP  : Bay Bus Processor / station HMI
      - SMC  : Station Monitor Controller

    Per-bay devices (4 per bay):
      - PP1BnL : Protection Panel 1, Bay n
      - PP2BnL : Protection Panel 2, Bay n
      - MU1Bn  : Merging Unit 1, Bay n
      - MU2Bn  : Merging Unit 2, Bay n

    Load balancing: for substations with more than 5 bays (24-port limit),
    bays 1 … floor(N/2) stay on SW1 and the rest go to SW2. This is the same
    split used by the DES case files (CASO{N}BAHIAS/*SWPB.xlsx).

    Args:
        n_bays: Number of protection bays to simulate.

    Returns:
        all_devices: Ordered list of device identifier strings.
        sw_map:      Dict mapping each device ID to 'SW1' or 'SW2'.
    """
    global_devices = ['GPS', 'BBP', 'SMC']
    bay_devices = []
    for n in range(1, n_bays + 1):
        bay_devices.extend([f'PP1B{n}L', f'PP2B{n}L', f'MU1B{n}', f'MU2B{n}'])

    all_devices = global_devices + bay_devices

    # All global devices reside on SW1 (the primary switch)
    sw_map = {d: 'SW1' for d in global_devices}

    for n in range(1, n_bays + 1):
        # Contiguous split once the bays exceed single-switch capacity
        target_sw = 'SW1' if (n_bays <= 5 or n <= n_bays // 2) else 'SW2'
        for unit in [f'PP1B{n}L', f'PP2B{n}L', f'MU1B{n}', f'MU2B{n}']:
            sw_map[unit] = target_sw

    return all_devices, sw_map


# =============================================================================
# 3. TRAFFIC DEMAND TENSOR
# =============================================================================

def generate_demand_tensor(
    n_bays: int,
    all_devices: list[str],
    delta_t: float = 1000.0,
) -> dict[str, pd.DataFrame]:
    """
    Build the traffic demand tensor D.

    D is a dictionary keyed by VLAN label. Each value is a square DataFrame
    indexed and columned by device IDs; cell D[vlan].at[src, dst] holds the
    bandwidth in Mbps that device 'src' transmits toward 'dst' on that VLAN.

    VLAN structure per bay (VIDs 1–17, excluding 5):
      V1/V2   : SV from MU → BBP + local PP
      V3/V4   : GOOSE Protection (PP ↔ PP, PP → MU)
      V5      : BBP global broadcast (all bay devices)
      V6/V7   : Control / status (SMC ↔ MU)
      V8–V15  : Monitoring flows (MU → BBP/SMC/PP)
      V16/V17 : Additional SV (MU → PP, intra-bay)

    Args:
        n_bays:      Number of bays.
        all_devices: Ordered device list from generate_devices().
        delta_t:     Current GOOSE retransmission interval [ms].
                     Used to set GOOSE bandwidth at steady state (1000 ms)
                     or during the burst phase.

    Returns:
        D: Demand tensor as described above.
    """
    # Define all VLAN keys upfront
    vlan_ids_bay = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    vlan_keys = ['L3_VLAN_PTP', 'L3_VLAN_V5']
    for n in range(1, n_bays + 1):
        for vid in vlan_ids_bay:
            vlan_keys.append(f'L3_VLAN_B{n}_V{vid}')

    # Initialise all DataFrames to zero (no traffic by default)
    D = {vlan: pd.DataFrame(0.0, index=all_devices, columns=all_devices)
         for vlan in vlan_keys}

    # -------------------------------------------------------------------------
    # A. Station-wide traffic (all bays see these flows)
    # -------------------------------------------------------------------------

    # PTP: GPS broadcasts Sync frames to every device except itself
    bw_ptp = get_mbps(TRAFFIC_SPECS['PTP']['size'], TRAFFIC_SPECS['PTP']['freq'])
    for target in all_devices:
        if target != 'GPS':
            D['L3_VLAN_PTP'].at['GPS', target] = bw_ptp

    # V5 (BBP global): BBP broadcast GOOSE to all bay devices (bursts on 50BF)
    bw_v5 = get_mbps(V5_WIRE_BYTES, 1000.0 / delta_t if delta_t > 0 else 0.0)  # 179 B frame + 20 B
    for n in range(1, n_bays + 1):
        for target in [f'PP1B{n}L', f'PP2B{n}L', f'MU1B{n}', f'MU2B{n}', 'SMC']:
            D['L3_VLAN_V5'].at['BBP', target] = bw_v5

    # -------------------------------------------------------------------------
    # B. Per-bay traffic
    # -------------------------------------------------------------------------
    freq_goose = 1000.0 / delta_t if delta_t > 0 else 0.0
    bw_goose = get_mbps(TRAFFIC_SPECS['GOOSE']['size'], freq_goose)
    bw_sv    = get_mbps(TRAFFIC_SPECS['SV']['size'],    TRAFFIC_SPECS['SV']['freq'])
    bw_smc   = get_mbps(TRAFFIC_SPECS['SMC']['size'],   1)
    bw_mu_r  = get_mbps(TRAFFIC_SPECS['MU_RES']['size'], 1)
    bw_mon   = get_mbps(TRAFFIC_SPECS['MON']['size'],   1)

    for n in range(1, n_bays + 1):
        pp1, pp2 = f'PP1B{n}L', f'PP2B{n}L'
        mu1, mu2 = f'MU1B{n}',  f'MU2B{n}'

        # V1/V2 — Sampled Values: each MU streams to BBP and its local PP
        D[f'L3_VLAN_B{n}_V1'].at[mu1, 'BBP'] = bw_sv
        D[f'L3_VLAN_B{n}_V1'].at[mu1,  pp1]  = bw_sv
        D[f'L3_VLAN_B{n}_V2'].at[mu2, 'BBP'] = bw_sv
        D[f'L3_VLAN_B{n}_V2'].at[mu2,  pp2]  = bw_sv

        # V16/V17 — Additional intra-bay SV (MU → PP; also → BBP when the
        # busbar protection subscribes four streams per bay)
        D[f'L3_VLAN_B{n}_V16'].at[mu1, pp1] = bw_sv
        D[f'L3_VLAN_B{n}_V17'].at[mu2, pp2] = bw_sv
        if config.BBP_SV_STREAMS_PER_BAY == 4:
            D[f'L3_VLAN_B{n}_V16'].at[mu1, 'BBP'] = bw_sv
            D[f'L3_VLAN_B{n}_V17'].at[mu2, 'BBP'] = bw_sv

        # V3/V4 — GOOSE Protection: cross-bay and to local MU
        # PP1 sends GOOSE to PP2 and MU1; PP2 sends GOOSE to PP1 and MU2
        D[f'L3_VLAN_B{n}_V3'].at[pp1, pp2] = bw_goose
        D[f'L3_VLAN_B{n}_V3'].at[pp1, mu1] = bw_goose
        D[f'L3_VLAN_B{n}_V4'].at[pp2, pp1] = bw_goose
        D[f'L3_VLAN_B{n}_V4'].at[pp2, mu2] = bw_goose

        # V6/V7 — Control: SMC polls each MU; MU replies
        D[f'L3_VLAN_B{n}_V6'].at['SMC', mu1] = bw_smc
        D[f'L3_VLAN_B{n}_V6'].at[mu1, 'SMC'] = bw_mu_r
        D[f'L3_VLAN_B{n}_V7'].at['SMC', mu2] = bw_smc
        D[f'L3_VLAN_B{n}_V7'].at[mu2, 'SMC'] = bw_mu_r

        # V8–V13 — MU breaker-status GOOSE (burst during the 50BF cascade)
        # V14/V15 — MU monitoring to the SMC (1 Hz, not event-driven)
        mon_routing = {
            8:  (mu1, 'BBP'), 9:  (mu2, 'BBP'),
            10: (mu1,  pp1),  11: (mu1,  pp1),
            12: (mu2,  pp2),  13: (mu2,  pp2),
            14: (mu1, 'SMC'), 15: (mu2, 'SMC'),
        }
        bw_status = get_mbps(TRAFFIC_SPECS['MON']['size'], freq_goose)
        for v, (src, dst) in mon_routing.items():
            D[f'L3_VLAN_B{n}_V{v}'].at[src, dst] = bw_status if v in GOOSE_VLAN_IDS else bw_mon

    return D


def event_burst_vlans(n_bays: int, burst_bays: list[int] | None = None) -> list[str]:
    """
    VLANs that retransmit in burst mode during a 50BF event.

    burst_bays = None -> every bay (worst case, as in the DES scenario: the event
    is injected simultaneously in all bays). Otherwise only the listed bays
    (trip V3/V4 and MU status V8-V13) burst, together with the BBP bus-trip
    broadcast V5, which is always part of the cascade.
    """
    bays = range(1, n_bays + 1) if burst_bays is None else burst_bays
    return ['L3_VLAN_V5'] + [f'L3_VLAN_B{n}_V{vid}' for n in bays for vid in GOOSE_VLAN_IDS]


def update_goose_demand(
    D_base: dict[str, pd.DataFrame],
    n_bays: int,
    delta_t: float,
    burst_bays: list[int] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Return an updated demand tensor with GOOSE VLANs recalculated for delta_t.

    Only the burst VLANs of the event (event_burst_vlans: BBP broadcast V5 and,
    for the bays in ``burst_bays``, trip V3/V4 and MU status V8–V13) change
    between burst snapshots. All other VLANs are shared by reference
    (shallow-copy of the dict) to avoid redundant DataFrame copies.

    Args:
        D_base:     Base demand tensor at steady state (delta_t = 1000 ms).
        n_bays:     Number of bays.
        delta_t:    Current GOOSE retransmission interval [ms].
        burst_bays: Bays involved in the event (None = all bays).

    Returns:
        Updated demand tensor D with fresh burst-VLAN DataFrames.
    """
    D = dict(D_base)  # Shallow copy — non-GOOSE DFs are shared, not duplicated

    freq_goose = 1000.0 / delta_t if delta_t > 0 else 0.0

    for vlan in event_burst_vlans(n_bays, burst_bays):
        size = get_frame_specs(vlan)['size']
        df = D_base[vlan].copy()
        df[df > 0] = get_mbps(size, freq_goose)
        D[vlan] = df

    return D


# =============================================================================
# 4. 5-LAYER GRAPH TOPOLOGY
# =============================================================================

def build_topology(
    n_bays: int,
    all_devices: list[str],
    D: dict[str, pd.DataFrame],
    sw_map: dict[str, str],
) -> tuple[list[str], np.ndarray]:
    """
    Build the 5-layer directed graph adjacency matrix and node label list.

    Layer structure (directed edges flow left to right):
      L1 (Equipment) → L2 (InPort) → L3 (VLAN@Switch) → L4 (Priority/Trunk)
                     → L5 (OutPort) → L1

    This multi-layer model captures:
      - L1→L2 : transmission delay at device ingress
      - L2→L3 : fixed switch processing delay
      - L3→L4 : fabric HOL-blocking contention
      - L4→L5 : output-port queuing delay and packet loss (M/M/1/K)
      - L5→L1 : physical egress (nominal weight)

    Args:
        n_bays:      Number of bays.
        all_devices: Ordered device list.
        D:           Current demand tensor (used to determine active VLANs).
        sw_map:      Device-to-switch assignment map.

    Returns:
        labels: List of node label strings (defines node index order).
        A:      Integer adjacency matrix of shape (N_nodes, N_nodes);
                A[i, j] = 1 if a directed edge exists from node i to node j.
    """
    # -------------------------------------------------------------------------
    # Build node label lists for each layer
    # -------------------------------------------------------------------------

    labels_l1 = [f'L1_Equipment_{d}' for d in all_devices]
    labels_l2 = [f'L2_InPort_{d}'    for d in all_devices]

    # L3: one node per (VLAN, switch) pair where that VLAN is active on that switch
    vlan_presence: dict[str, set[str]] = {vlan: set() for vlan in D}
    for vlan, df in D.items():
        # A VLAN is active on a switch if any sender or receiver is connected there
        active_devs = df.index[df.any(axis=1)].union(df.columns[df.any(axis=0)])
        for d in active_devs:
            vlan_presence[vlan].add(sw_map[d])

    labels_l3 = [
        f'{vlan}_{sw}'
        for vlan, switches in vlan_presence.items()
        for sw in sorted(switches)
    ]

    # L4: one priority-queue node per (switch, priority) pair observed in L3
    prio_presence: set[tuple[str, int]] = set()
    for l3 in labels_l3:
        vlan_base, sw_id = l3.rsplit('_', 1)
        prio_presence.add((sw_id, get_vlan_priority(vlan_base)))

    labels_l4 = [
        f'L4_Priority_{sw}_Prio_{p}'
        for sw, p in sorted(prio_presence)
    ]

    # Add inter-switch trunk nodes when the substation spans two switches
    if n_bays > 5:
        labels_l4 += ['L4_Trunk_SW1_to_SW2', 'L4_Trunk_SW2_to_SW1']

    # L5: one output-port node per device (GPS has no egress queue in this model)
    labels_l5 = [f'L5_OutPort_{d}' for d in all_devices if d != 'GPS']

    # Concatenate all layers into a single ordered label list
    labels = labels_l1 + labels_l2 + labels_l3 + labels_l4 + labels_l5
    n_nodes = len(labels)
    A = np.zeros((n_nodes, n_nodes), dtype=int)

    def add_edge(u: str, v: str) -> None:
        """Add a directed edge u → v; silently ignored if either node is absent."""
        try:
            A[labels.index(u), labels.index(v)] = 1
        except ValueError:
            pass

    # -------------------------------------------------------------------------
    # Wire the layers
    # -------------------------------------------------------------------------

    # Layer 1 → Layer 2: every device enters its ingress port
    for d in all_devices:
        add_edge(f'L1_Equipment_{d}', f'L2_InPort_{d}')

    # Layer 2 → Layer 3: senders connect to the VLAN node on their switch
    for vlan, df in D.items():
        for src in df.index[df.any(axis=1)]:
            add_edge(f'L2_InPort_{src}', f'{vlan}_{sw_map[src]}')

    # Layer 3 → Layer 4: each VLAN node connects to its priority queue
    for l3 in labels_l3:
        vlan_base, sw = l3.rsplit('_', 1)
        prio = get_vlan_priority(vlan_base)
        add_edge(l3, f'L4_Priority_{sw}_Prio_{prio}')

    # Layer 4 trunks (only for N > 5 — two-switch topology)
    if n_bays > 5:
        # Physical trunk connection (bidirectional)
        add_edge('L4_Trunk_SW1_to_SW2', 'L4_Trunk_SW2_to_SW1')
        add_edge('L4_Trunk_SW2_to_SW1', 'L4_Trunk_SW1_to_SW2')

        sw1_devs = [d for d in all_devices if sw_map[d] == 'SW1']
        sw2_devs = [d for d in all_devices if sw_map[d] == 'SW2']

        for vlan, df in D.items():
            prio = get_vlan_priority(vlan)

            # Traffic flowing SW1 → SW2
            if df.loc[sw1_devs, sw2_devs].values.any():
                add_edge(f'L4_Priority_SW1_Prio_{prio}', 'L4_Trunk_SW1_to_SW2')
                # Trunk egress feeds into the destination VLAN node on SW2
                add_edge('L4_Trunk_SW2_to_SW1', f'{vlan}_SW2')

            # Traffic flowing SW2 → SW1
            if df.loc[sw2_devs, sw1_devs].values.any():
                add_edge(f'L4_Priority_SW2_Prio_{prio}', 'L4_Trunk_SW2_to_SW1')
                add_edge('L4_Trunk_SW1_to_SW2', f'{vlan}_SW1')

    # Layer 4 → Layer 5: priority queue feeds the output port of each destination
    for vlan, df in D.items():
        prio = get_vlan_priority(vlan)
        for src in df.index:
            for dst in df.columns:
                if df.at[src, dst] > 0:
                    sw_dst = sw_map[dst]
                    add_edge(f'L4_Priority_{sw_dst}_Prio_{prio}', f'L5_OutPort_{dst}')

    # Layer 5 → Layer 1: output port delivers frame to destination equipment
    for d in all_devices:
        if d != 'GPS':
            add_edge(f'L5_OutPort_{d}', f'L1_Equipment_{d}')

    return labels, A
