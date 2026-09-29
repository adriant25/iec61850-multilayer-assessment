# -*- coding: utf-8 -*-
"""
single_bay_two_sw.py
====================
Standalone script — does NOT modify any original file.

Simulates 1 protection bay distributed across TWO managed switches and
generates a publication-quality 5-layer complex-network topology figure
suitable for an IEEE journal or conference paper.

Device assignment
-----------------
  SW1 (primary)  : GPS, BBP, SMC, PP1B1L, MU1B1
  SW2 (redundant): PP2B1L, MU2B1

Cross-switch traffic via the inter-switch trunk:
  GOOSE protection : PP1B1L <-> PP2B1L
  Sampled Values   : MU2B1  ->  BBP
  Control/status   : SMC    <-> MU2B1

Run:  python single_bay_two_sw.py
Outputs (written to results/):
  IEEE_1Bay_2SW.pdf   <- vector, recommended for LaTeX \\includegraphics
  IEEE_1Bay_2SW.png   <- raster 600 dpi fallback
"""

import os
import sys
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')          # headless — no interactive window
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# ── allow imports from project root ──────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import (
    T_MAX, get_vlan_priority, get_mbps, TRAFFIC_SPECS,
    CAPACITY_DEFAULT, CAPACITY_TRUNK, FABRIC_CAPACITY,
    T_PROC_SWITCH, K_BUFFER, FACTOR_4,
)
from model   import generate_demand_tensor
from analysis import compute_weights

OUT = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)


# =============================================================================
# 1.  DEVICE CONFIGURATION — 1 bay, 2 switches
# =============================================================================

ALL_DEVICES = ['GPS', 'BBP', 'SMC', 'PP1B1L', 'PP2B1L', 'MU1B1', 'MU2B1']

SW_MAP = {
    'GPS':    'SW1',   # PTP grandmaster — always on primary switch
    'BBP':    'SW1',   # Bay Bus Processor / HMI
    'SMC':    'SW1',   # Station Monitor Controller
    'PP1B1L': 'SW1',   # Protection Panel 1 — primary relay
    'MU1B1':  'SW1',   # Merging Unit 1
    'PP2B1L': 'SW2',   # Protection Panel 2 — backup relay on SW2
    'MU2B1':  'SW2',   # Merging Unit 2
}

N_BAYS   = 1
SCENARIO = 'base'


# =============================================================================
# 2.  TOPOLOGY BUILDER  (trunk always present regardless of bay count)
# =============================================================================

def build_topology():
    """
    Construct the 5-layer adjacency matrix and node label list.

    Returns
    -------
    labels : list[str]   ordered node identifiers
    A      : ndarray     binary adjacency matrix (n x n)
    D      : dict        traffic demand tensor (used for weight computation)
    """
    D = generate_demand_tensor(N_BAYS, ALL_DEVICES, delta_t=1000.0)

    # ── Layer 1 & 2 ──────────────────────────────────────────────────────────
    labels_l1 = [f'L1_Equipment_{d}' for d in ALL_DEVICES]
    labels_l2 = [f'L2_InPort_{d}'    for d in ALL_DEVICES]

    # ── Layer 3: one node per (VLAN, switch) pair ────────────────────────────
    vlan_presence = {v: set() for v in D}
    for vlan, df in D.items():
        active = df.index[df.any(axis=1)].union(df.columns[df.any(axis=0)])
        for d in active:
            vlan_presence[vlan].add(SW_MAP[d])

    labels_l3 = [
        f'{v}_{sw}'
        for v, sws in vlan_presence.items()
        for sw in sorted(sws)
    ]

    # ── Layer 4: priority queues + forced trunk nodes ────────────────────────
    prio_seen: set = set()
    for l3 in labels_l3:
        vb, sw = l3.rsplit('_', 1)
        prio_seen.add((sw, get_vlan_priority(vb)))

    labels_l4 = (
        [f'L4_Priority_{sw}_Prio_{p}' for sw, p in sorted(prio_seen)]
        + ['L4_Trunk_SW1_to_SW2', 'L4_Trunk_SW2_to_SW1']
    )

    # ── Layer 5 ───────────────────────────────────────────────────────────────
    labels_l5 = [f'L5_OutPort_{d}' for d in ALL_DEVICES if d != 'GPS']

    labels = labels_l1 + labels_l2 + labels_l3 + labels_l4 + labels_l5
    n      = len(labels)
    A      = np.zeros((n, n), dtype=np.int8)

    def e(u, v):
        try:
            A[labels.index(u), labels.index(v)] = 1
        except ValueError:
            pass

    sw1_devs = [d for d in ALL_DEVICES if SW_MAP[d] == 'SW1']
    sw2_devs = [d for d in ALL_DEVICES if SW_MAP[d] == 'SW2']

    # L1 -> L2
    for d in ALL_DEVICES:
        e(f'L1_Equipment_{d}', f'L2_InPort_{d}')

    # L2 -> L3  (sender injects into its switch's VLAN node)
    for vlan, df in D.items():
        for src in df.index[df.any(axis=1)]:
            e(f'L2_InPort_{src}', f'{vlan}_{SW_MAP[src]}')

    # L3 -> L4  (VLAN node feeds the appropriate priority queue)
    for l3 in labels_l3:
        vb, sw = l3.rsplit('_', 1)
        e(l3, f'L4_Priority_{sw}_Prio_{get_vlan_priority(vb)}')

    # Trunk physical cable (bidirectional)
    e('L4_Trunk_SW1_to_SW2', 'L4_Trunk_SW2_to_SW1')
    e('L4_Trunk_SW2_to_SW1', 'L4_Trunk_SW1_to_SW2')

    # Cross-switch traffic -> trunk ingress / VLAN injection on remote switch
    for vlan, df in D.items():
        p = get_vlan_priority(vlan)
        # SW1 -> SW2
        if df.loc[sw1_devs, sw2_devs].values.any():
            e(f'L4_Priority_SW1_Prio_{p}', 'L4_Trunk_SW1_to_SW2')
            if f'{vlan}_SW2' in labels:
                e('L4_Trunk_SW2_to_SW1', f'{vlan}_SW2')
        # SW2 -> SW1
        if df.loc[sw2_devs, sw1_devs].values.any():
            e(f'L4_Priority_SW2_Prio_{p}', 'L4_Trunk_SW2_to_SW1')
            if f'{vlan}_SW1' in labels:
                e('L4_Trunk_SW1_to_SW2', f'{vlan}_SW1')

    # L4 priority -> L5 output port of each destination
    for vlan, df in D.items():
        p = get_vlan_priority(vlan)
        for src in df.index:
            for dst in df.columns:
                if df.at[src, dst] > 0 and dst != 'GPS':
                    e(f'L4_Priority_{SW_MAP[dst]}_Prio_{p}', f'L5_OutPort_{dst}')

    # L5 -> L1  (frame delivered to destination equipment)
    for d in ALL_DEVICES:
        if d != 'GPS':
            e(f'L5_OutPort_{d}', f'L1_Equipment_{d}')

    return labels, A, D


# =============================================================================
# 3.  NODE LABEL HELPERS
# =============================================================================

def device_of(lbl: str):
    """Extract the device identifier from any layer label, or None."""
    for pfx in ('L1_Equipment_', 'L2_InPort_', 'L5_OutPort_'):
        if lbl.startswith(pfx):
            return lbl[len(pfx):]
    return None


def short(lbl: str) -> str:
    """Compact, human-readable label used in the figure."""
    if lbl == 'L4_Trunk_SW1_to_SW2':
        return 'Tr₁→₂'   # Tr₁→₂
    if lbl == 'L4_Trunk_SW2_to_SW1':
        return 'Tr₂→₁'   # Tr₂→₁
    if 'L4_Priority_SW1_Prio_' in lbl:
        return f'P{lbl[-1]}·₁'   # P6·₁
    if 'L4_Priority_SW2_Prio_' in lbl:
        return f'P{lbl[-1]}·₂'   # P6·₂
    if lbl.startswith('L3_VLAN_'):
        sw_digit = '1' if lbl.endswith('_SW1') else '2'
        if 'PTP' in lbl:
            return f'PTP·{sw_digit}'
        if '_V5_' in lbl or lbl.endswith('_V5_SW1') or lbl.endswith('_V5_SW2'):
            return f'V5·{sw_digit}'
        for tok in lbl.split('_'):
            if tok.startswith('V') and tok[1:].isdigit():
                return f'{tok}·{sw_digit}'
        return lbl[-9:]
    for pfx in ('L1_Equipment_', 'L2_InPort_', 'L5_OutPort_'):
        if lbl.startswith(pfx):
            d = lbl[len(pfx):]
            # Shorten device names for L1/L5 labels
            d = d.replace('B1L', '').replace('B1', '')   # PP1B1L -> PP1, MU1B1 -> MU1
            return d
    return lbl


# =============================================================================
# 4.  HIERARCHICAL LAYOUT
# =============================================================================

#  y-bands:  4 = L1 Equipment
#            3 = L2 InPort
#            2 = L3 VLAN
#            1 = L4 Priority / Trunk
#            0 = L5 OutPort
#
#  x-split:  [0.0, 0.5) = SW1 domain
#             [0.5, 1.0] = SW2 domain  (divider at x = 0.5)

def make_positions(labels: list) -> dict:
    """Return {node_index: (x, y)} for a hierarchical 5-band layout."""
    pos = {}

    def _place(idxs, y, x0, x1):
        n = max(len(idxs), 1)
        for k, i in enumerate(idxs):
            pos[i] = (x0 + (k + 0.5) / n * (x1 - x0), float(y))

    l1 = [i for i, l in enumerate(labels) if l.startswith('L1_')]
    l2 = [i for i, l in enumerate(labels) if l.startswith('L2_')]
    l3 = [i for i, l in enumerate(labels) if l.startswith('L3_')]
    l4 = [i for i, l in enumerate(labels) if l.startswith('L4_')]
    l5 = [i for i, l in enumerate(labels) if l.startswith('L5_')]

    # L1, L2, L5 — split by device's switch
    for band_nodes, y in [(l1, 4.0), (l2, 3.0), (l5, 0.0)]:
        s1 = [i for i in band_nodes if SW_MAP.get(device_of(labels[i])) != 'SW2']
        s2 = [i for i in band_nodes if SW_MAP.get(device_of(labels[i])) == 'SW2']
        _place(s1, y, 0.01, 0.49)
        _place(s2, y, 0.51, 0.99)

    # L3 — SW1 VLANs on left, SW2 VLANs on right, each split into 2 sub-rows
    l3_s1 = [i for i in l3 if labels[i].endswith('_SW1')]
    l3_s2 = [i for i in l3 if labels[i].endswith('_SW2')]

    def _place_2rows(idxs, x0, x1):
        """Stagger nodes into 2 sub-rows to halve horizontal density."""
        row0 = idxs[0::2]   # even-indexed -> upper sub-row
        row1 = idxs[1::2]   # odd-indexed  -> lower sub-row
        _place(row0, 2.20, x0, x1)
        _place(row1, 1.80, x0, x1)

    _place_2rows(l3_s1, 0.01, 0.49)
    _place_2rows(l3_s2, 0.51, 0.99)

    # L4 — SW1 priorities | trunk nodes (centre) | SW2 priorities
    p1 = sorted([i for i in l4 if 'Priority_SW1' in labels[i]])
    tr = [i for i in l4 if 'Trunk' in labels[i]]
    p2 = sorted([i for i in l4 if 'Priority_SW2' in labels[i]])
    _place(p1, 1.0, 0.02, 0.36)
    _place(tr, 1.0, 0.42, 0.58)
    _place(p2, 1.0, 0.64, 0.98)

    return pos


# =============================================================================
# 5.  VISUAL STYLE
# =============================================================================

# ColorBrewer Set1 / Dark2 palette — distinguishable in print and colour-blind
STYLE = {
    'L1':  dict(color='#2166AC', size=420, zorder=6, marker='o'),
    'L2':  dict(color='#4DAF4A', size=140, zorder=4, marker='o'),
    'L3':  dict(color='#FF7F00', size=90,  zorder=4, marker='s'),
    'L4p': dict(color='#E41A1C', size=260, zorder=5, marker='o'),
    'L4t': dict(color='#984EA3', size=480, zorder=7, marker='D'),
    'L5':  dict(color='#1B9E77', size=200, zorder=4, marker='o'),
}


def layer_key(lbl: str) -> str:
    if lbl.startswith('L1_'): return 'L1'
    if lbl.startswith('L2_'): return 'L2'
    if lbl.startswith('L3_'): return 'L3'
    if 'Trunk'    in lbl:     return 'L4t'
    if lbl.startswith('L4_'): return 'L4p'
    if lbl.startswith('L5_'): return 'L5'
    return 'L3'


def nodes_by_layer(labels, G):
    groups = {k: [] for k in STYLE}
    for i in G.nodes():
        groups[layer_key(labels[i])].append(i)
    return groups


# =============================================================================
# 6.  MAIN FIGURE
# =============================================================================

def plot_ieee_graph():
    print("Building topology …")
    labels, A, D = build_topology()
    print(f"  Nodes: {len(labels)}")

    pos = make_positions(labels)

    # Steady-state edge weights (latency proxy)
    W, _J, _PL, _u = compute_weights(
        labels, A, D, SW_MAP, ALL_DEVICES, delta_t=1000.0, scenario=SCENARIO
    )

    # ── NetworkX graph (skip L5->L1 feedback — avoids upward arrows) ─────────
    G = nx.DiGraph()
    for i in range(len(labels)):
        G.add_node(i)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if A[i, j]:
                if 'L5_OutPort' in labels[i] and 'L1_Equipment' in labels[j]:
                    continue   # omit the L5→L1 feedback cycle
                w = float(W[i, j]) if W[i, j] > 0 else 5e-4
                G.add_edge(i, j, weight=w)

    print(f"  Edges drawn: {G.number_of_edges()}  (L5->L1 feedback omitted)")

    # ── Figure setup ───────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family':     'DejaVu Sans',
        'font.size':       8,
        'axes.linewidth':  0,
        'savefig.dpi':     600,
    })

    FIG_W, FIG_H = 12.0, 7.0    # inches — scales to IEEE double-column at ~300 dpi
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # ── Layer background bands (one solid color per layer) ───────────────────
    # y-ranges account for L3 now spanning two sub-rows [1.80, 2.20]
    layer_spans = [
        (-0.52,  0.52, '#D6EAF8'),   # L5 Egress   — light blue
        ( 0.55,  1.45, '#FDEDEC'),   # L4 Priority  — light red
        ( 1.55,  2.45, '#FEF9E7'),   # L3 VLAN      — light yellow
        ( 2.55,  3.45, '#EAFAF1'),   # L2 Ingress   — light green
        ( 3.55,  4.52, '#EBF5FB'),   # L1 Equipment — blue
    ]
    for y0, y1, color in layer_spans:
        ax.axhspan(y0, y1, color=color, alpha=0.80, zorder=0)
    # Thin horizontal separators between layers
    for y_sep in (0.535, 1.455, 1.545, 2.455, 2.545, 3.455):
        ax.axhline(y_sep, color='#AAAAAA', lw=0.7, ls='-', zorder=1, alpha=0.6)

    # ── SW1 / SW2 vertical background tint ──────────────────────────────────
    ax.axvspan(-0.02, 0.495, color='#2980B9', alpha=0.06, zorder=0)
    ax.axvspan(0.505, 1.20,  color='#8E44AD', alpha=0.06, zorder=0)

    # Vertical divider
    ax.axvline(0.5, color='#AAAAAA', lw=1.2, ls='--', zorder=1, alpha=0.8)

    # ── Edges ─────────────────────────────────────────────────────────────────
    trunk_ids = {
        labels.index('L4_Trunk_SW1_to_SW2'),
        labels.index('L4_Trunk_SW2_to_SW1'),
    }
    trunk_edges     = [(u, v) for u, v in G.edges()
                       if u in trunk_ids or v in trunk_ids]
    non_trunk_edges = [(u, v) for u, v in G.edges()
                       if (u, v) not in set(trunk_edges)]

    # Non-trunk: visible dark edges so connections between nodes are clear
    if non_trunk_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=non_trunk_edges, ax=ax,
            width=1.3, edge_color='#444444', alpha=0.72,
            arrows=True, arrowsize=7, arrowstyle='-|>',
            connectionstyle='arc3,rad=0.04',
            min_source_margin=5, min_target_margin=5,
        )

    # Trunk edges: thinner than before but still distinct via color
    if trunk_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=trunk_edges, ax=ax,
            width=1.1, edge_color='#984EA3', alpha=0.90,
            arrows=True, arrowsize=9, arrowstyle='-|>',
            connectionstyle='arc3,rad=0.18',
            min_source_margin=7, min_target_margin=7,
        )

    # ── Nodes (drawn lowest layer first so critical nodes paint on top) ───────
    grps = nodes_by_layer(labels, G)
    draw_order = ['L2', 'L3', 'L5', 'L4p', 'L1', 'L4t']
    for key in draw_order:
        idxs = grps.get(key, [])
        if not idxs:
            continue
        st = STYLE[key]
        nx.draw_networkx_nodes(
            G, pos, nodelist=idxs, ax=ax,
            node_color=st['color'],
            node_size=st['size'],
            edgecolors='#333333', linewidths=0.8,
            alpha=1.0,
        )

    # ── Node labels (L1, L4, L5 only — L2 and L3 are too dense) ─────────────
    lbl_map = {}
    for i, lbl in enumerate(labels):
        if (lbl.startswith('L1_') or
                lbl.startswith('L4_') or
                lbl.startswith('L5_')):
            lbl_map[i] = short(lbl)

    nx.draw_networkx_labels(
        G, pos, labels=lbl_map, ax=ax,
        font_size=6.8, font_color='white',
        font_weight='bold', font_family='DejaVu Sans',
    )

    # ── Band annotations (right margin) ───────────────────────────────────────
    band_text = {
        4.0: ('L1', 'Equipment\n(IEDs)'),
        3.0: ('L2', 'Ingress Port'),
        2.0: ('L3', 'VLAN nodes\n@ Switch (×2 rows)'),
        1.0: ('L4', 'Priority Queue\n& Trunk'),
        0.0: ('L5', 'Egress Port'),
    }
    for y, (layer, desc) in band_text.items():
        ax.text(1.015, y + 0.18, layer,
                transform=ax.get_yaxis_transform(),
                fontsize=8, color='#333333',
                va='center', ha='left', fontweight='bold')
        ax.text(1.015, y - 0.15, desc,
                transform=ax.get_yaxis_transform(),
                fontsize=6.3, color='#555555',
                va='center', ha='left', style='italic',
                linespacing=1.3)

    # ── Switch domain headers ─────────────────────────────────────────────────
    hdr_kw = dict(transform=ax.transData, fontsize=9,
                  fontweight='bold', va='bottom', ha='center',
                  bbox=dict(boxstyle='round,pad=0.25', fc='white',
                            ec='#CCCCCC', lw=0.8, alpha=0.9))
    ax.text(0.25, 4.60, 'Switch SW1 (Primary)', color='#1A5276', **hdr_kw)
    ax.text(0.75, 4.60, 'Switch SW2 (Redundant)', color='#6C3483', **hdr_kw)

    # ── Legend — placed BELOW the graph, outside the axes ────────────────────
    legend_handles = [
        mpatches.Patch(facecolor='#2166AC', edgecolor='#333333', lw=0.8,
                       label='L1 — IED / Equipment'),
        mpatches.Patch(facecolor='#4DAF4A', edgecolor='#333333', lw=0.8,
                       label='L2 — Ingress Port'),
        mpatches.Patch(facecolor='#FF7F00', edgecolor='#333333', lw=0.8,
                       label='L3 — VLAN node @ Switch'),
        mpatches.Patch(facecolor='#E41A1C', edgecolor='#333333', lw=0.8,
                       label='L4 — Priority Queue (P4-P7)'),
        mpatches.Patch(facecolor='#984EA3', edgecolor='#333333', lw=0.8,
                       label='L4 — Inter-Switch Trunk'),
        mpatches.Patch(facecolor='#1B9E77', edgecolor='#333333', lw=0.8,
                       label='L5 — Egress Port'),
        Line2D([0], [0], color='#444444', lw=1.3,
               label='Intra-switch traffic flow'),
        Line2D([0], [0], color='#984EA3', lw=1.1,
               label='Inter-switch trunk flow'),
    ]
    # anchor the legend below the axes at figure level
    leg = ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.04),   # just below the axes bottom edge
        bbox_transform=ax.transAxes,
        fontsize=7.2,
        framealpha=0.97,
        edgecolor='#BBBBBB',
        title='Graph Layer Legend',
        title_fontsize=7.8,
        ncol=4,                        # 4 columns keeps it compact
        handlelength=1.5,
        handleheight=1.0,
        borderpad=0.7,
        columnspacing=1.2,
        labelspacing=0.45,
    )
    leg.get_frame().set_linewidth(0.8)

    # ── Topology statistics annotation (below legend) ─────────────────────────
    n_nodes = len(labels)
    n_edges = G.number_of_edges()
    n_l3    = sum(1 for l in labels if l.startswith('L3_'))
    ax.text(0.5, -0.16,
            (f'{n_nodes} nodes  |  {n_edges} directed edges  '
             f'(L5-L1 feedback omitted)  |  {n_l3} VLAN nodes in L3'),
            ha='center', va='top', fontsize=6.8, color='#666666',
            transform=ax.transAxes, style='italic')

    # ── Axes cleanup ──────────────────────────────────────────────────────────
    ax.set_xlim(-0.02, 1.20)
    ax.set_ylim(-0.55, 5.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # leave room at the bottom for legend + stats line
    plt.subplots_adjust(bottom=0.18, top=0.97, left=0.01, right=0.92)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_png = os.path.join(OUT, 'IEEE_1Bay_2SW.png')
    out_pdf = os.path.join(OUT, 'IEEE_1Bay_2SW.pdf')

    fig.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white',
                pad_inches=0.05)
    fig.savefig(out_pdf,           bbox_inches='tight', facecolor='white',
                pad_inches=0.05)
    plt.close(fig)

    print(f"\nOutputs written:")
    print(f"  {out_png}")
    print(f"  {out_pdf}")


# =============================================================================
# 7.  ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("IEC 61850  |  1-Bay  x  2-Switch  |  IEEE Graph Figure")
    print("=" * 60)
    plot_ieee_graph()
    print("\nDone.")
