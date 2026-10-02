# -*- coding: utf-8 -*-
"""
fig_layer_model.py -- Schematic of the five-layer model (Fig. 2 of the paper).

Shows the layers as bands, the two switches S_a and S_b, and the path of a frame
of flow f on VLAN k with priority p, as defined in the paper (Eq. path_intra and
the inter-switch segment):

* between switches:  L1 s -> L2 s -> L3 k@S_a -> L4 p@S_a -> L4 trunk a->b
                     -> L4 trunk b->a -> L3 k@S_b -> L4 p@S_b -> L5 d -> L1 d
* within one switch: L4 p@S_a -> L5 d' -> L1 d'

The last edge L5 -> L1 delivers the frame to the subscriber and closes the path.

Outputs: results/paper/fig_layer_model.pdf (vector) and .png (600 dpi)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 7, 'mathtext.fontset': 'stix',
                     'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
INK, GREY, BAND = '#1f1f1d', '#8a8a86', '#f2f2ef'
INTER, INTRA = '#2a78d6', '#eb6834'          # validated categorical slots 1 and 2

fig, ax = plt.subplots(figsize=(3.5, 2.95))
X0, X1 = 0.0, 10.0
ax.set_xlim(-3.15, X1 + 0.25)
ax.set_ylim(-1.05, 4.55)
ax.axis('off')

Y = {1: 4.0, 2: 3.0, 3: 2.0, 4: 1.0, 5: 0.0}
names = {1: r'$\mathcal{L}_1$ Equipment', 2: r'$\mathcal{L}_2$ Ingress ports', 3: r'$\mathcal{L}_3$ VLANs',
         4: r'$\mathcal{L}_4$ Queues, trunk', 5: r'$\mathcal{L}_5$ Egress ports'}
for k, y in Y.items():
    ax.add_patch(FancyBboxPatch((X0 - 0.2, y - 0.3), X1 - X0 + 0.4, 0.6, boxstyle='round,pad=0,rounding_size=0.12',
                                fc=BAND, ec='none', zorder=0))
    ax.text(X0 - 0.35, y, names[k], ha='right', va='center', color=INK, fontsize=6.8)

# switches: dashed frames over L2-L5
for x0, x1, lab in ((0.25, 4.75, r'Switch $S_a$'), (5.25, 9.75, r'Switch $S_b$')):
    ax.add_patch(FancyBboxPatch((x0, -0.45), x1 - x0, 3.85, boxstyle='round,pad=0,rounding_size=0.15',
                                fc='none', ec=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1))
    ax.text((x0 + x1) / 2, -0.62, lab, ha='center', va='top', color=INK, fontsize=6.8)

P = {  # node positions
    's1': (2.0, 4), 'dp1': (0.8, 4), 'd1': (8.7, 4),
    's2': (2.0, 3),
    'k3a': (2.45, 2), 'k3b': (6.9, 2),
    'p4a': (2.45, 1), 'tab': (4.08, 1), 'tba': (6.05, 1), 'p4b': (7.6, 1),
    'dp5': (0.8, 0), 'd5': (8.7, 0),
}


def node(key, label, kind):
    x, y = P[key]
    if kind == 'dev':
        w = 0.17 * len(label.strip('$')) + 0.55
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.2), w, 0.4, boxstyle='round,pad=0,rounding_size=0.08',
                                    fc=INK, ec=INK, zorder=4))
        ax.text(x, y, label, ha='center', va='center', color='white', fontsize=7, zorder=5)
    elif kind == 'port':
        ax.plot(x, y, 'o', ms=4.5, mfc='white', mec=INK, mew=0.8, zorder=4)
        if label:
            ax.text(x + 0.22, y + 0.12, label, ha='left', va='bottom', fontsize=6.2, color=INK, zorder=5)
    elif kind == 'trunk':
        ax.add_patch(FancyBboxPatch((x - 0.55, y - 0.21), 1.1, 0.42, boxstyle='round,pad=0,rounding_size=0.18',
                                    fc='white', ec=INK, lw=0.8, zorder=4))
        ax.text(x, y, label, ha='center', va='center', fontsize=6.2, color=INK, zorder=5)
    else:
        ax.plot(x, y, 'o', ms=13, mfc='white', mec=INK, mew=0.8, zorder=4)
        ax.text(x, y, label, ha='center', va='center', fontsize=6.2, color=INK, zorder=5)


def edge(a, b, color, ls='-', lw=1.0, rad=0.0, sa=6, sb=6):
    ax.add_patch(FancyArrowPatch(P[a], P[b], arrowstyle='-|>,head_length=3,head_width=1.6',
                                 connectionstyle=f'arc3,rad={rad}', color=color, lw=lw, ls=ls,
                                 shrinkA=sa, shrinkB=sb, zorder=3))


# path between switches
path = [('s1', 's2', 9, 4), ('s2', 'k3a', 4, 7), ('k3a', 'p4a', 7, 7), ('p4a', 'tab', 7, 10.5), ('tab', 'tba', 10.5, 10.5),
        ('tba', 'k3b', 5, 7), ('k3b', 'p4b', 7, 7), ('p4b', 'd5', 7, 4)]
for a, b, sa, sb in path:
    edge(a, b, INTER, sa=sa, sb=sb)
# path within one switch
edge('p4a', 'dp5', INTRA, ls=(0, (4, 2)), sa=7, sb=4)
# delivery edges L5 -> L1 that close each path
edge('d5', 'd1', INTER, ls=(0, (1, 1.6)), sa=4, sb=8)
edge('dp5', 'dp1', INTRA, ls=(0, (1, 1.6)), sa=4, sb=8)

node('s1', '$s$', 'dev'); node('dp1', "$d'$", 'dev'); node('d1', '$d$', 'dev')
node('s2', '', 'port'); node('dp5', '', 'port'); node('d5', '', 'port')
node('k3a', '$k$', 'q'); node('k3b', '$k$', 'q')
node('p4a', '$p$', 'q'); node('p4b', '$p$', 'q')
node('tab', r'$a{\to}b$', 'trunk'); node('tba', r'$b{\to}a$', 'trunk')
ax.text(5.0, 0.62, 'trunk link', ha='center', va='top', fontsize=6.2, color=INK, zorder=5,
        bbox=dict(fc=BAND, ec='none', pad=0.6))

handles = [Line2D([], [], color=INTER, lw=1.2, label=r'Path $s\to d$ between switches'),
           Line2D([], [], color=INTRA, lw=1.2, ls=(0, (4, 2)), label=r"Path $s\to d'$ within $S_a$"),
           Line2D([], [], color=GREY, lw=1.2, ls=(0, (1, 1.6)), label=r'Delivery $\mathcal{L}_5\to\mathcal{L}_1$')]
ax.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False,
          fontsize=6, handlelength=2.2, columnspacing=0.9, handletextpad=0.5)

out = os.path.join('results', 'paper')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'fig_layer_model.pdf'))
fig.savefig(os.path.join(out, 'fig_layer_model.png'))
print('written')
