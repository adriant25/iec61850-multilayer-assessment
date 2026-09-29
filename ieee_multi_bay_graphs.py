# -*- coding: utf-8 -*-
"""
ieee_multi_bay_graphs.py
========================
Generates four IEEE-quality 5-layer topology figures, one per bay count:
    N = 1, 5, 6, 10

Switch assignment follows the ORIGINAL model.py logic:
    N <= 5  -->  all devices on SW1 (single switch)
    N >  5  -->  even-numbered bays go to SW2, rest to SW1

One PNG + PDF is saved per bay count:
    results/IEEE_N01_Bays.png / .pdf
    results/IEEE_N05_Bays.png / .pdf
    results/IEEE_N06_Bays.png / .pdf
    results/IEEE_N10_Bays.png / .pdf

Run:
    python ieee_multi_bay_graphs.py
"""

import os, sys, math
import numpy as np
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config  import T_MAX, get_vlan_priority, TRAFFIC_SPECS
from model   import generate_devices, generate_demand_tensor, build_topology
from analysis import compute_weights

OUT = os.path.join(ROOT, 'results')
os.makedirs(OUT, exist_ok=True)

BAY_COUNTS = [1, 5, 6, 10]

# ─────────────────────────────────────────────────────────────────────────────
# VISUAL STYLE
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    'L1':  '#2166AC',   # deep blue — equipment
    'L2':  '#4DAF4A',   # green     — ingress port
    'L3':  '#FF7F00',   # orange    — VLAN node
    'L4p': '#E41A1C',   # red       — priority queue
    'L4t': '#984EA3',   # purple    — inter-switch trunk
    'L5':  '#1B9E77',   # teal      — egress port
}

LAYER_BG = [
    (-0.52,  0.52, '#D6EAF8'),   # L5
    ( 0.55,  1.45, '#FDEDEC'),   # L4
    ( 1.55,  2.45, '#FEF9E7'),   # L3
    ( 2.55,  3.45, '#EAFAF1'),   # L2
    ( 3.55,  4.52, '#EBF5FB'),   # L1
]

# Node sizes per bay count
NODE_SIZES = {
    1:  {'L1': 400, 'L2': 130, 'L3': 85,  'L4p': 250, 'L4t': 460, 'L5': 185},
    5:  {'L1': 90,  'L2': 42,  'L3': 27,  'L4p': 130, 'L4t': 300, 'L5': 72},
    6:  {'L1': 75,  'L2': 35,  'L3': 22,  'L4p': 110, 'L4t': 270, 'L5': 60},
    10: {'L1': 42,  'L2': 20,  'L3': 14,  'L4p': 80,  'L4t': 210, 'L5': 36},
}

# Font size for node labels per bay count
FONT_SIZES = {1: 6.8, 5: 5.8, 6: 5.5, 10: 5.0}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def layer_key(lbl: str) -> str:
    if lbl.startswith('L1_'): return 'L1'
    if lbl.startswith('L2_'): return 'L2'
    if lbl.startswith('L3_'): return 'L3'
    if 'Trunk'    in lbl:     return 'L4t'
    if lbl.startswith('L4_'): return 'L4p'
    if lbl.startswith('L5_'): return 'L5'
    return 'L3'


def device_of(lbl: str):
    for pfx in ('L1_Equipment_', 'L2_InPort_', 'L5_OutPort_'):
        if lbl.startswith(pfx):
            return lbl[len(pfx):]
    return None


def short(lbl: str) -> str:
    """Compact label for figure annotations."""
    if lbl == 'L4_Trunk_SW1_to_SW2': return 'Tr12'
    if lbl == 'L4_Trunk_SW2_to_SW1': return 'Tr21'
    if 'L4_Priority_SW1_Prio_' in lbl: return f'P{lbl[-1]}.1'
    if 'L4_Priority_SW2_Prio_' in lbl: return f'P{lbl[-1]}.2'
    if lbl.startswith('L3_'):
        return ''   # L3 nodes never labelled (too many)
    for pfx in ('L1_Equipment_', 'L2_InPort_', 'L5_OutPort_'):
        if lbl.startswith(pfx):
            d = lbl[len(pfx):]
            d = d.replace('B1L', '').replace('B2L', '').replace('B3L', '') \
                 .replace('B4L', '').replace('B5L', '').replace('B6L', '') \
                 .replace('B7L', '').replace('B8L', '').replace('B9L', '') \
                 .replace('B10L', '').replace('B1', '').replace('B2', '') \
                 .replace('B3', '').replace('B4', '').replace('B5', '') \
                 .replace('B6', '').replace('B7', '').replace('B8', '') \
                 .replace('B9', '').replace('B10', '')
            return d
    return lbl


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

def _place_flat(idxs, y, x0, x1, pos):
    n = max(len(idxs), 1)
    for k, i in enumerate(idxs):
        pos[i] = (x0 + (k + 0.5) / n * (x1 - x0), float(y))


def _place_multirow(idxs, x0, x1, y_min, y_max, pos, max_rows=4):
    """Spread nodes into multiple sub-rows to reduce horizontal density."""
    n = len(idxs)
    if n == 0:
        return
    n_rows = min(max_rows, max(1, math.ceil(n / 28)))
    rh = (y_max - y_min) / n_rows
    for r in range(n_rows):
        row = idxs[r::n_rows]   # interleave rows for visual balance
        y   = y_min + (r + 0.5) * rh
        m   = max(len(row), 1)
        for k, i in enumerate(row):
            pos[i] = (x0 + (k + 0.5) / m * (x1 - x0), y)


def make_positions(labels, sw_map, n_bays):
    """
    Build {node_index: (x, y)} positions.

    Single switch (N<=5): all nodes fill [0,1] width, no divider.
    Two switches (N>5):   SW1 -> [0,0.5), SW2 -> (0.5,1.0].
    L3 band: [1.55, 2.45], subdivided into multiple rows.
    """
    two_sw = n_bays > 5
    pos    = {}

    l1 = [i for i, l in enumerate(labels) if l.startswith('L1_')]
    l2 = [i for i, l in enumerate(labels) if l.startswith('L2_')]
    l3 = [i for i, l in enumerate(labels) if l.startswith('L3_')]
    l4 = [i for i, l in enumerate(labels) if l.startswith('L4_')]
    l5 = [i for i, l in enumerate(labels) if l.startswith('L5_')]

    if not two_sw:
        # ── Single switch: full-width layout ─────────────────────────────
        _place_flat(l1, 4.0, 0.02, 0.98, pos)
        _place_flat(l2, 3.0, 0.02, 0.98, pos)
        _place_multirow(l3, 0.02, 0.98, 1.55, 2.45, pos, max_rows=4)
        pq = sorted([i for i in l4 if 'Priority' in labels[i]])
        _place_flat(pq, 1.0, 0.15, 0.85, pos)
        _place_flat(l5, 0.0, 0.02, 0.98, pos)

    else:
        # ── Two switches: SW1 left / SW2 right ───────────────────────────
        for band_nodes, y in [(l1, 4.0), (l2, 3.0), (l5, 0.0)]:
            s1 = [i for i in band_nodes
                  if sw_map.get(device_of(labels[i])) != 'SW2']
            s2 = [i for i in band_nodes
                  if sw_map.get(device_of(labels[i])) == 'SW2']
            _place_flat(s1, y, 0.01, 0.49, pos)
            _place_flat(s2, y, 0.51, 0.99, pos)

        l3_s1 = [i for i in l3 if labels[i].endswith('_SW1')]
        l3_s2 = [i for i in l3 if labels[i].endswith('_SW2')]
        _place_multirow(l3_s1, 0.01, 0.49, 1.55, 2.45, pos, max_rows=4)
        _place_multirow(l3_s2, 0.51, 0.99, 1.55, 2.45, pos, max_rows=4)

        p1 = sorted([i for i in l4 if 'Priority_SW1' in labels[i]])
        tr = [i for i in l4 if 'Trunk' in labels[i]]
        p2 = sorted([i for i in l4 if 'Priority_SW2' in labels[i]])
        _place_flat(p1, 1.0, 0.02, 0.36, pos)
        _place_flat(tr, 1.0, 0.42, 0.58, pos)
        _place_flat(p2, 1.0, 0.64, 0.98, pos)

    return pos


# ─────────────────────────────────────────────────────────────────────────────
# LABEL POLICY  (fewer labels as N grows — avoid clutter)
# ─────────────────────────────────────────────────────────────────────────────

_GLOBAL_DEVS = {'GPS', 'BBP', 'SMC'}


def build_label_map(labels, n_bays):
    lbl_map = {}
    for i, lbl in enumerate(labels):
        s = short(lbl)
        if not s:
            continue   # L3 or unparseable — skip
        if lbl.startswith('L4_'):
            lbl_map[i] = s          # L4 always labelled
        elif n_bays <= 1 and (lbl.startswith('L1_') or lbl.startswith('L5_')):
            lbl_map[i] = s          # N=1: show all equipment + egress
        elif n_bays <= 5 and lbl.startswith('L1_'):
            d = device_of(lbl)
            if d in _GLOBAL_DEVS:
                lbl_map[i] = d      # N=5: only GPS, BBP, SMC in L1
    return lbl_map


# ─────────────────────────────────────────────────────────────────────────────
# PER-FIGURE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

def plot_for_n(n_bays: int) -> None:
    two_sw = n_bays > 5
    print(f"\n  N={n_bays:2d} bays  ({'2 switches' if two_sw else '1 switch '})")

    # ── Build model ──────────────────────────────────────────────────────────
    devices, sw_map = generate_devices(n_bays)
    D               = generate_demand_tensor(n_bays, devices, delta_t=1000.0)
    labels, A       = build_topology(n_bays, devices, D, sw_map)
    W, _J, _PL, _u  = compute_weights(
        labels, A, D, sw_map, devices, delta_t=1000.0, scenario='base'
    )

    n_nodes = len(labels)
    print(f"    nodes={n_nodes}  VLANs={sum(1 for l in labels if l.startswith('L3_'))}")

    # ── NetworkX graph (skip L5->L1 feedback) ────────────────────────────────
    G = nx.DiGraph()
    for i in range(n_nodes):
        G.add_node(i)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if A[i, j]:
                if 'L5_OutPort' in labels[i] and 'L1_Equipment' in labels[j]:
                    continue
                G.add_edge(i, j, weight=float(W[i, j]) if W[i, j] > 0 else 5e-4)

    pos   = make_positions(labels, sw_map, n_bays)
    sizes = NODE_SIZES[n_bays]
    fsz   = FONT_SIZES[n_bays]

    # ── Figure ───────────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                         'axes.linewidth': 0})
    FIG_W, FIG_H = 12.0, 8.0
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Layer background bands
    for y0, y1, color in LAYER_BG:
        ax.axhspan(y0, y1, color=color, alpha=0.82, zorder=0)
    for y_sep in (0.535, 1.455, 1.545, 2.455, 2.545, 3.455):
        ax.axhline(y_sep, color='#AAAAAA', lw=0.7, zorder=1, alpha=0.55)

    # SW tints + divider (only for 2-switch topology)
    if two_sw:
        ax.axvspan(-0.02, 0.495, color='#2980B9', alpha=0.06, zorder=0)
        ax.axvspan(0.505, 1.20,  color='#8E44AD', alpha=0.06, zorder=0)
        ax.axvline(0.5, color='#999999', lw=1.1, ls='--', zorder=1, alpha=0.8)

    # ── Edges ────────────────────────────────────────────────────────────────
    trunk_ids = set()
    for lbl in ('L4_Trunk_SW1_to_SW2', 'L4_Trunk_SW2_to_SW1'):
        if lbl in labels:
            trunk_ids.add(labels.index(lbl))

    trunk_edges     = [(u, v) for u, v in G.edges()
                       if u in trunk_ids or v in trunk_ids]
    non_trunk_edges = [(u, v) for u, v in G.edges()
                       if (u, v) not in set(trunk_edges)]

    # Non-trunk: dark and visible
    if non_trunk_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=non_trunk_edges, ax=ax,
            width=1.2, edge_color='#3D3D3D', alpha=0.65,
            arrows=True, arrowsize=6, arrowstyle='-|>',
            connectionstyle='arc3,rad=0.04',
            min_source_margin=4, min_target_margin=4,
        )

    # Trunk: thinner but distinctively colored
    if trunk_edges:
        nx.draw_networkx_edges(
            G, pos, edgelist=trunk_edges, ax=ax,
            width=1.0, edge_color='#984EA3', alpha=0.92,
            arrows=True, arrowsize=8, arrowstyle='-|>',
            connectionstyle='arc3,rad=0.18',
            min_source_margin=6, min_target_margin=6,
        )

    # ── Nodes (low layers drawn first) ───────────────────────────────────────
    grps = {k: [] for k in COLORS}
    for i in G.nodes():
        grps[layer_key(labels[i])].append(i)

    for key in ('L2', 'L3', 'L5', 'L4p', 'L1', 'L4t'):
        idxs = grps.get(key, [])
        if not idxs:
            continue
        nx.draw_networkx_nodes(
            G, pos, nodelist=idxs, ax=ax,
            node_color=COLORS[key],
            node_size=sizes[key],
            edgecolors='#333333', linewidths=0.7,
            alpha=1.0,
        )

    # ── Labels ───────────────────────────────────────────────────────────────
    lbl_map = build_label_map(labels, n_bays)
    if lbl_map:
        nx.draw_networkx_labels(
            G, pos, labels=lbl_map, ax=ax,
            font_size=fsz, font_color='white',
            font_weight='bold', font_family='DejaVu Sans',
        )

    # ── Switch / layer annotations ───────────────────────────────────────────
    # Layer band labels (right margin)
    band_ann = {
        4.0: ('L1', 'Equipment\n(IEDs)'),
        3.0: ('L2', 'Ingress\nPort'),
        2.0: ('L3', 'VLAN\nnodes'),
        1.0: ('L4', 'Priority\nQueue & Trunk'),
        0.0: ('L5', 'Egress\nPort'),
    }
    for y, (layer, desc) in band_ann.items():
        ax.text(1.012, y + 0.20, layer,
                transform=ax.get_yaxis_transform(),
                fontsize=8, color='#333333',
                va='center', ha='left', fontweight='bold')
        ax.text(1.012, y - 0.18, desc,
                transform=ax.get_yaxis_transform(),
                fontsize=6.0, color='#555555',
                va='center', ha='left', style='italic', linespacing=1.3)

    # Switch domain headers
    hdr_kw = dict(transform=ax.transData, fontsize=8.5,
                  fontweight='bold', va='bottom', ha='center',
                  bbox=dict(boxstyle='round,pad=0.2', fc='white',
                            ec='#CCCCCC', lw=0.8, alpha=0.92))
    if two_sw:
        ax.text(0.25, 4.58, 'Switch SW1 (Primary)',   color='#1A5276', **hdr_kw)
        ax.text(0.75, 4.58, 'Switch SW2 (Redundant)', color='#6C3483', **hdr_kw)
    else:
        ax.text(0.50, 4.58, 'Switch SW1',             color='#1A5276', **hdr_kw)

    # Node / edge count subtitle (upper left corner)
    n_l3 = sum(1 for l in labels if l.startswith('L3_'))
    ax.text(0.01, 4.92,
            f'N = {n_bays} {"bay" if n_bays == 1 else "bays"}  |  '
            f'{n_nodes} nodes  |  {G.number_of_edges()} edges  |  '
            f'{n_l3} VLAN nodes (L3)',
            transform=ax.transData, fontsize=7.5,
            color='#222222', fontweight='bold', va='top', ha='left')

    # ── Legend (bottom, outside axes) ────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor='#2166AC', edgecolor='#333', lw=0.8,
                       label='L1 - IED / Equipment'),
        mpatches.Patch(facecolor='#4DAF4A', edgecolor='#333', lw=0.8,
                       label='L2 - Ingress Port'),
        mpatches.Patch(facecolor='#FF7F00', edgecolor='#333', lw=0.8,
                       label='L3 - VLAN node @ Switch'),
        mpatches.Patch(facecolor='#E41A1C', edgecolor='#333', lw=0.8,
                       label='L4 - Priority Queue (P4-P7)'),
        mpatches.Patch(facecolor='#984EA3', edgecolor='#333', lw=0.8,
                       label='L4 - Inter-Switch Trunk'),
        mpatches.Patch(facecolor='#1B9E77', edgecolor='#333', lw=0.8,
                       label='L5 - Egress Port'),
        Line2D([0], [0], color='#3D3D3D', lw=1.2,
               label='Intra-switch traffic flow'),
        Line2D([0], [0], color='#984EA3', lw=1.0,
               label='Inter-switch trunk flow'),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.04),
        bbox_transform=ax.transAxes,
        fontsize=7.0, framealpha=0.97, edgecolor='#BBBBBB',
        title='Graph Layer Legend', title_fontsize=7.5,
        ncol=4, handlelength=1.5, handleheight=1.0,
        borderpad=0.7, columnspacing=1.1, labelspacing=0.4,
    )
    leg.get_frame().set_linewidth(0.8)

    # ── Axes cleanup ─────────────────────────────────────────────────────────
    ax.set_xlim(-0.02, 1.20)
    ax.set_ylim(-0.55, 5.15)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(bottom=0.15, top=0.97, left=0.01, right=0.91)

    # ── Save ─────────────────────────────────────────────────────────────────
    tag     = f'N{n_bays:02d}'
    out_png = os.path.join(OUT, f'IEEE_{tag}_Bays.png')
    out_pdf = os.path.join(OUT, f'IEEE_{tag}_Bays.pdf')
    fig.savefig(out_png, dpi=600, bbox_inches='tight',
                facecolor='white', pad_inches=0.05)
    fig.savefig(out_pdf,           bbox_inches='tight',
                facecolor='white', pad_inches=0.05)
    plt.close(fig)
    print(f"    Saved: {out_png}")
    print(f"    Saved: {out_pdf}")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("IEC 61850  |  Multi-Bay IEEE Topology Figures")
    print(f"Bay counts: {BAY_COUNTS}")
    print("=" * 60)
    for n in BAY_COUNTS:
        plot_for_n(n)
    print("\nAll figures written to:", OUT)
