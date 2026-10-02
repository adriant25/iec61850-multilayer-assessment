# -*- coding: utf-8 -*-
"""
fig_graph_layers.py -- Figure of the five-layer multilayer graph of the case study
(N = 1, base scenario), drawn from the supra-adjacency matrix built by the model.

Nodes are neutral and arranged in one row per layer; the order inside each row
minimizes edge crossings (barycenter heuristic). The edges that cross the VLAN
layer (L2 -> L3 -> L4) are coloured by the traffic class of the VLAN, with a line
style as secondary encoding; device and port edges are grey. The six delivery
edges L5 -> L1 that close each path at the subscriber are not drawn.

Outputs: results/paper/fig_graph_layers.pdf (vector) and .png (600 dpi)
"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

from config import SV_VLAN_IDS, GOOSE_VLAN_IDS
from model import generate_devices, generate_demand_tensor, build_topology

N = 1
devs, sw = generate_devices(N)
D0 = generate_demand_tensor(N, devs, 1000.0)
labels, A = build_topology(N, devs, D0, sw)
A = np.asarray(A)
edges = [(labels[i], labels[j]) for i, j in zip(*np.nonzero(A))]

# ---------------------------------------------------------------- classes
CLASSES = {  # name: (colour, line style) -- validated categorical slots 1-4
    'SV': ('#2a78d6', '-'),
    '50BF GOOSE': ('#eb6834', '-'),
    'Other GOOSE': ('#1baf7a', (0, (4, 2))),
    'PTP': ('#eda100', (0, (1, 1.5))),
}
GREY = '#8a8a86'
INK = '#1f1f1d'


def vlan_class(node):
    m = re.search(r'L3_VLAN_(.*)_SW\d+$', node)
    name = m.group(1)
    if name == 'PTP':
        return 'PTP'
    if name == 'V5':                      # BBP bus-trip broadcast
        return '50BF GOOSE'
    vid = int(re.search(r'V(\d+)$', name).group(1))
    if vid in SV_VLAN_IDS:
        return 'SV'
    if vid in GOOSE_VLAN_IDS:
        return '50BF GOOSE'
    return 'Other GOOSE'


def layer(node):
    return int(node[1])


# ---------------------------------------------------------------- positions
rows = {k: [n for n in labels if layer(n) == k] for k in range(1, 6)}
dev_order = ['SMC', 'PP1B1L', 'MU1B1', 'BBP', 'MU2B1', 'PP2B1L', 'GPS']
dev = lambda n: n.split('_', 2)[2]
x = {}
W = 18.0
for k in (1, 2):
    for n in rows[k]:
        x[n] = W * dev_order.index(dev(n)) / (len(dev_order) - 1)
for n in rows[5]:
    x[n] = W * dev_order.index(dev(n)) / (len(dev_order) - 1)
# L4 queues: provisional even spacing, then barycentres of L3 between L2 and L4
q_order = sorted(rows[4], key=lambda n: int(n[-1]))
for i, n in enumerate(q_order):
    x[n] = 2.5 + (W - 5.0) * i / (len(q_order) - 1)
for _ in range(3):
    bc = {}
    for n in rows[3]:
        nb = [x[s] for s, t in edges if t == n] + [x[t] for s, t in edges if s == n]
        bc[n] = np.mean(nb)
    order3 = sorted(rows[3], key=lambda n: bc[n])
    for i, n in enumerate(order3):
        x[n] = W * i / (len(order3) - 1)
Y = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 0.0}
pos = {n: (x[n], Y[layer(n)]) for n in labels}

# ---------------------------------------------------------------- figure
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 7, 'mathtext.fontset': 'stix',
                     'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
fig, ax = plt.subplots(figsize=(7.16, 3.55))
ax.set_xlim(-3.6, W + 0.6)
ax.set_ylim(-0.55, 4.45)
ax.axis('off')

# layer bands and names
names = {1: 'L1  Equipment', 2: 'L2  Ingress ports', 3: 'L3  VLANs (SW1)',
         4: 'L4  Priority queues', 5: 'L5  Egress ports'}
for k, y in Y.items():
    ax.add_patch(FancyBboxPatch((-0.55, y - 0.27), W + 1.1, 0.54,
                                boxstyle='round,pad=0,rounding_size=0.12',
                                fc='#f2f2ef', ec='none', zorder=0))
    ax.text(-0.85, y, names[k], ha='right', va='center', fontsize=7, color=INK)

R = {1: 0.0, 2: 0.075, 3: 0.17, 4: 0.21, 5: 0.075}


def draw_edge(s, t, color, ls, lw, z):
    (x0, y0), (x1, y1) = pos[s], pos[t]
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>,head_length=2.6,head_width=1.3',
                                 connectionstyle='arc3,rad=0.0', color=color, lw=lw, ls=ls,
                                 shrinkA=4.5 if layer(s) != 1 else 7.5,
                                 shrinkB=4.5 if layer(t) in (2, 5) else 6.5,
                                 alpha=0.95, zorder=z, joinstyle='miter', capstyle='butt'))


for s, t in edges:
    ls_, lt = layer(s), layer(t)
    if ls_ == 5 and lt == 1:
        continue                                  # delivery edges, not drawn
    if ls_ == 2 and lt == 3:
        c, st = CLASSES[vlan_class(t)]
        draw_edge(s, t, c, st, 0.75, 2)
    elif ls_ == 3 and lt == 4:
        c, st = CLASSES[vlan_class(s)]
        draw_edge(s, t, c, st, 0.75, 2)
    else:
        draw_edge(s, t, GREY, '-', 0.55, 1)

# nodes
for n in rows[1]:
    xx, yy = pos[n]
    w = 0.2 * len(dev(n)) + 0.4
    ax.add_patch(FancyBboxPatch((xx - w / 2, yy - 0.17), w, 0.34, boxstyle='round,pad=0,rounding_size=0.08',
                                fc=INK, ec=INK, lw=0.6, zorder=4))
    ax.text(xx, yy, dev(n), ha='center', va='center', color='white', fontsize=6.6, zorder=5)
for k in (2, 5):
    for n in rows[k]:
        ax.plot(*pos[n], 'o', ms=4.2, mfc='white', mec=INK, mew=0.7, zorder=4)
for n in rows[5]:
    ax.text(pos[n][0], pos[n][1] - 0.2, dev(n), ha='center', va='top', fontsize=5.8, color='#55554f')
for n in rows[3]:
    name = re.search(r'L3_VLAN_(.*)_SW\d+$', n).group(1)
    lab = 'PTP' if name == 'PTP' else name.split('_')[-1].lstrip('V')
    ax.plot(*pos[n], 'o', ms=9.0 if lab != 'PTP' else 10.5, mfc='white', mec=INK, mew=0.7, zorder=4)
    ax.text(*pos[n], lab, ha='center', va='center', fontsize=5.2 if lab != 'PTP' else 4.4, color=INK, zorder=5)
for n in rows[4]:
    ax.plot(*pos[n], 'o', ms=12.5, mfc='white', mec=INK, mew=0.9, zorder=4)
    ax.text(*pos[n], 'P' + n[-1], ha='center', va='center', fontsize=6.4, color=INK, zorder=5)

# legend (traffic class of the VLAN edges; grey = device and port edges)
handles = [Line2D([], [], color=c, ls=st, lw=1.3, label=k) for k, (c, st) in CLASSES.items()]
handles.append(Line2D([], [], color=GREY, lw=1.0, label='Device / port edges'))
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.56, 1.075), ncol=5, frameon=False,
          fontsize=6.6, handlelength=2.6, columnspacing=1.4)

out = os.path.join('results', 'paper')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'fig_graph_layers.pdf'))
fig.savefig(os.path.join(out, 'fig_graph_layers.png'))
n_drawn = sum(1 for s, t in edges if not (layer(s) == 5 and layer(t) == 1))
print(f'{len(labels)} nodes, {len(edges)} directed edges ({n_drawn} drawn)')
