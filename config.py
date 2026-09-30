# -*- coding: utf-8 -*-
"""
config.py — Global constants and protocol specifications.

IEC 61850 Digital Substation — Complex Network Scalability Simulation
----------------------------------------------------------------------
Defines all physical, timing, and protocol parameters used across the
simulation. Centralising constants here makes it easy to adapt the model
to different substation configurations without touching simulation logic.
"""

import numpy as np

# =============================================================================
# TIMING AND CAPACITY PARAMETERS
# =============================================================================

# IEC 61850 hard deadline for protection and SV traffic (Class P6 / P4)
T_MAX: float = 0.003            # 3 ms in seconds

# Link capacities (bits per second)
CAPACITY_DEFAULT: float = 100e6          # 100 Mbps — standard bay device links
CAPACITY_TRUNK: float = 100e6            # 100 Mbps — inter-switch trunk, base scenario
CAPACITY_TRUNK_UPGRADED: float = 1000e6  # 1 Gbps  — inter-switch trunk, upgraded scenario
CAPACITY_BBP: float = 1000e6             # 1 Gbps  — BBP uplink (aggregates SV from every bay)

# Switch fabric (backplane) capacity used for HOL-blocking model
FABRIC_CAPACITY: float = 68000.0  # Mbps

# Fixed per-hop processing delay inside a managed switch.
# Must match the DES reference (Simulador/main.py: switch_processing_time = 2e-6).
T_PROC_SWITCH: float = 2e-6  # 2 µs in seconds

# M/M/1/K buffer capacity expressed in packets.
# Buffer size = 4,000,000 B (same as the DES port_buffer_size); packet size = 149 B (SV frame).
K_BUFFER: int = int(np.floor(4_000_000 / 149))  # = 26,845 packets

# Output-port buffer [bytes] and observation window [s] used by the fluid
# overload model of saturated ports (same buffer and duration as the DES runs).
PORT_BUFFER_BYTES: float = 4_000_000
OBS_WINDOW_S: float = 7.0

# Legacy constant, still imported by single_bay_two_sw.py. The main model uses
# the M/D/1 mean wait 0.5 · (L/C) · ρ/(1-ρ) directly in analysis.compute_weights.
FACTOR_4: int = 4

# SV / GOOSE VLAN IDs (per-bay). Used to select end-to-end flows and frame specs.
SV_VLAN_IDS: tuple = (1, 2, 16, 17)
# GOOSE flows that retransmit in burst mode during the 50BF cascade, matching the
# DES response table: line-protection trip (V3/V4: PP -> peer PP + own MU) and the
# breaker-status reply of each MU (V8/V9: MU -> BBP, V10-V13: MU -> own PP).
GOOSE_VLAN_IDS: tuple = (3, 4, 8, 9, 10, 11, 12, 13)

# =============================================================================
# TRAFFIC SPECIFICATIONS PER PROTOCOL
# =============================================================================

# Each entry holds the steady-state transmission frequency (Hz) and
# Ethernet frame size (bytes, including header) of each traffic class.
TRAFFIC_SPECS: dict = {
    "PTP":    {"freq": 3,    "size": 80},   # IEEE 1588 Sync/Follow-Up
    "SV":     {"freq": 4800, "size": 149},  # IEC 61869-9 Sampled Values (80 smp/cycle @ 60 Hz)
    "GOOSE":  {"freq": 1,    "size": 187},  # IEC 61850-8-1 keep-alive (burst modelled separately)
    "MON":    {"freq": 1,    "size": 171},  # Generic monitoring / MMS
    "SMC":    {"freq": 1,    "size": 211},  # Station Monitor Controller request
    "MU_RES": {"freq": 1,    "size": 340},  # Merging Unit response
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_vlan_priority(vlan_name: str) -> int:
    """
    Map a VLAN identifier to its IEEE 802.1Q Priority Code Point (PCP).

    Priority mapping follows IEC 61850 traffic class recommendations:
      - P7: PTP time-sync (highest)
      - P6: Protection / GOOSE critical (VLANs 3, 4, 5, 8–12)
      - P5: Control / status exchange  (VLANs 6, 7, 11, 13)
      - P4: Sampled Values / low-prio  (VLANs 1, 2, 14–17)  [default]

    Args:
        vlan_name: VLAN label string, e.g. 'L3_VLAN_B1_V3_SW1'.

    Returns:
        Integer priority in range [4, 7].
    """
    if "PTP" in vlan_name:
        return 7  # Highest — time synchronisation must not be delayed

    if "V5" in vlan_name:
        return 6  # BBP global protection broadcast

    # Extract numeric VLAN ID from the trailing _V<id> suffix
    try:
        vid = int(vlan_name.split('_V')[-1])
        if vid in [1, 2, 14, 15, 16, 17]:
            return 4  # Sampled Values and low-priority monitoring
        if vid in [6, 7, 11, 13]:
            return 5  # Control / status (SMC ↔ MU)
        if vid in [3, 4, 8, 9, 10, 12]:
            return 6  # Protection-critical (GOOSE, differential, etc.)
    except (ValueError, IndexError):
        pass

    return 4  # Default: treat unknown VLANs as low-priority


def get_vlan_id(vlan_name: str) -> int | None:
    """
    Return the numeric per-bay VLAN ID of a label such as 'L3_VLAN_B3_V16', or
    None for station-wide VLANs ('L3_VLAN_PTP', 'L3_VLAN_V5').

    Exact parsing avoids substring collisions ('_V1' also matches '_V10'…'_V17').
    """
    if 'PTP' in vlan_name or vlan_name.endswith('_V5') or '_B' not in vlan_name:
        return None
    try:
        return int(vlan_name.split('_V')[-1])
    except (ValueError, IndexError):
        return None


def is_burst_vlan(vlan_name: str) -> bool:
    """True for VLANs whose GOOSE retransmits in burst mode during the 50BF
    cascade: per-bay trip/status VLANs (GOOSE_VLAN_IDS) and the BBP broadcast V5."""
    if 'PTP' in vlan_name:
        return False
    vid = get_vlan_id(vlan_name)
    return vid in GOOSE_VLAN_IDS if vid is not None else vlan_name.endswith('_V5')


def get_frame_specs(vlan_name: str, src: str | None = None,
                    goose_freq: float = 1.0) -> dict:
    """
    Frame size (bytes) and rate (Hz) of the flows carried by a VLAN.

    Args:
        vlan_name:  VLAN label.
        src:        Sending device (only needed for V6/V7, whose direction
                    determines SMC request vs. MU response frames).
        goose_freq: Current GOOSE retransmission rate [Hz] of the burst VLANs.
    """
    if 'PTP' in vlan_name:
        return TRAFFIC_SPECS['PTP']
    vid = get_vlan_id(vlan_name)
    if vid is None:  # V5: BBP broadcast GOOSE (also bursts in the 50BF cascade)
        return {'freq': goose_freq, 'size': 179}
    if vid in SV_VLAN_IDS:
        return TRAFFIC_SPECS['SV']
    if vid in (3, 4):   # line-protection trip GOOSE
        return {'freq': goose_freq, 'size': TRAFFIC_SPECS['GOOSE']['size']}
    if vid in GOOSE_VLAN_IDS:   # MU breaker-status GOOSE (V8-V13)
        return {'freq': goose_freq, 'size': TRAFFIC_SPECS['MON']['size']}
    if vid in (6, 7):
        return TRAFFIC_SPECS['SMC'] if src == 'SMC' else TRAFFIC_SPECS['MU_RES']
    return TRAFFIC_SPECS['MON']


def get_mbps(size_bytes: int, freq_hz: float) -> float:
    """
    Convert packet size and transmission frequency to bandwidth in Mbps.

    Args:
        size_bytes: Frame size in bytes (Ethernet payload + headers).
        freq_hz:    Transmission frequency in frames per second.

    Returns:
        Bandwidth in Megabits per second (Mbps).
    """
    return (size_bytes * 8 * freq_hz) / 1e6
