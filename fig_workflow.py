# -*- coding: utf-8 -*-
"""
fig_workflow.py -- Methodology workflow (Fig. 1 of the paper), drawn as a vector
figure so that the paper does not need TikZ (which conflicts with the IEEE
Access class).

Initialization: substation configuration -> offered-traffic tensor O ->
supra-adjacency matrix A and weights W. Evolution monitoring: protection event
-> burst-mode update of O -> recompute W(t) -> graph metrics -> decision "is the
traffic burst over?" (no: next burst snapshot; yes: IEC 61850 compliance).

Outputs: results/paper/fig_workflow.pdf (vector) and .png (600 dpi)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 7, 'mathtext.fontset': 'stix',
                     'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
INK, GREY, BAND = '#1f1f1d', '#8a8a86', '#f2f2ef'
BLUE, ORANGE = '#2a78d6', '#eb6834'

fig, ax = plt.subplots(figsize=(7.16, 1.95))
ax.set_xlim(-0.65, 15.95)
ax.set_ylim(-2.75, 1.25)
ax.set_aspect('equal')
ax.axis('off')

W, H = 1.55, 0.78
blocks = [  # key, x centre, label, width
    ('cfg', 0.6, 'Substation\nconfiguration', 1.85),
    ('ot', 2.55, 'Build tensor\n$\\mathcal{O}$', W),
    ('mat', 4.42, 'Build\n$\\tilde{\\mathbf{A}}$, $\\mathbf{W}$', W),
    ('event', 6.55, 'Protection\nevent', W),
    ('burst', 8.5, 'Burst-mode\nupdate of $\\mathcal{O}$', 1.75),
    ('wt', 10.45, 'Recompute\n$\\mathbf{W}(t)$', W),
    ('met', 12.35, 'Graph\nmetrics', W),
    ('comp', 14.4, 'IEC 61850\ncompliance', 1.65),
]
pos = {}
for key, x, lab, w in blocks:
    fc = BLUE if key == 'comp' else 'white'
    tc = 'white' if key == 'comp' else INK
    ax.add_patch(FancyBboxPatch((x - w / 2, -H / 2), w, H, boxstyle='round,pad=0,rounding_size=0.1',
                                fc=fc, ec=BLUE if key == 'comp' else INK, lw=0.7, zorder=3))
    ax.text(x, 0, lab, ha='center', va='center', fontsize=6.8, color=tc, zorder=4, linespacing=1.25)
    pos[key] = (x, w)


def arrow(p0, p1, color=INK, lw=0.8):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>,head_length=3,head_width=1.6', color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, zorder=2))


chain = ['cfg', 'ot', 'mat', 'event', 'burst', 'wt', 'met']
for a, b in zip(chain[:-1], chain[1:]):
    arrow((pos[a][0] + pos[a][1] / 2, 0), (pos[b][0] - pos[b][1] / 2, 0))

# decision below the graph metrics
xd, yd, dw, dh = pos['met'][0], -1.75, 1.05, 0.62
ax.add_patch(Polygon([(xd, yd + dh), (xd + dw, yd), (xd, yd - dh), (xd - dw, yd)], closed=True,
                     fc='#fbe3d8', ec=ORANGE, lw=0.8, zorder=3))
ax.text(xd, yd, 'Is the traffic\nburst over?', ha='center', va='center', fontsize=6.3, zorder=4, linespacing=1.2)
arrow((xd, -H / 2), (xd, yd + dh))
# no: next burst snapshot (back to the burst-mode update)
xb = pos['burst'][0]
ax.plot([xd - dw, xb], [yd, yd], color=INK, lw=0.8, zorder=2)
arrow((xb, yd), (xb, -H / 2))
ax.text(xd - dw - 0.15, yd + 0.1, 'No', ha='right', va='bottom', fontsize=6.4)
ax.text((xb + xd - dw) / 2 - 0.3, yd - 0.12, 'next burst snapshot', ha='center', va='top', fontsize=6.2,
        color='#55554f')
# yes: compliance assessment
xc = pos['comp'][0]
ax.plot([xd + dw, xc], [yd, yd], color=INK, lw=0.8, zorder=2)
arrow((xc, yd), (xc, -H / 2))
ax.text(xd + dw + 0.15, yd + 0.1, 'Yes', ha='left', va='bottom', fontsize=6.4)

# dashed stage frames
for x0, x1, y0, lab in ((-0.5, 5.35, -0.75, 'Initialization'), (5.6, 15.4, -2.55, 'Evolution monitoring')):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, 0.75 - y0 + 0.15, boxstyle='round,pad=0,rounding_size=0.15',
                                fc='none', ec=GREY, lw=0.7, ls=(0, (3, 2)), zorder=1))
    ax.text((x0 + x1) / 2 if lab == 'Initialization' else x1 - 0.1, 0.98, lab, ha='center' if lab == 'Initialization' else 'right',
            va='bottom', fontsize=6.8, color=INK)

out = os.path.join('results', 'paper')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'fig_workflow.pdf'))
fig.savefig(os.path.join(out, 'fig_workflow.png'))
print('written')
