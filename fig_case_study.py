# -*- coding: utf-8 -*-
"""
fig_case_study.py -- Case-study architecture (Fig. 3 of the paper).

(a) Base unit (N = 1): single-line diagram of the bay with a busbar voltage
    transformer (VT1), disconnector, CT1, circuit breaker, CT2 and a line voltage
    transformer (VT2). Each CT has two cores, one per merging unit, and both VTs
    feed both merging units (MU1B1, MU2B1), which also control the breaker. The
    merging units, protection IEDs, busbar protection (BBP), SMC and GPS are on
    the process-bus switch SW1; PP1B1L, PP2B1L and BBP are also on the station bus.
(b) Modular expansion to N bays: up to five bays every device is on SW1; for
    N >= 6 bays 1..floor(N/2) stay on SW1 with BBP, SMC and GPS, the others move
    to SW2, joined by the trunk. The BBP link and the trunk are the links upgraded
    to 1 Gbps in the upgrade scenarios.

Outputs: results/paper/fig_case_study.pdf (vector) and .png (600 dpi)
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 7, 'mathtext.fontset': 'stix',
                     'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
INK, GREY, BAND = '#1f1f1d', '#8a8a86', '#f2f2ef'
PB, SB = '#2a78d6', '#eb6834'          # process bus, station bus (validated slots 1, 2)

fig, ax = plt.subplots(figsize=(7.16, 3.6))
ax.set_xlim(0, 24.4)
ax.set_ylim(-1.5, 12.75)
ax.set_aspect('equal')
ax.axis('off')
for x0, w in ((0.0, 9.6), (10.0, 14.4)):
    ax.add_patch(FancyBboxPatch((x0, -0.15), w, 12.85, boxstyle='round,pad=0,rounding_size=0.3',
                                fc=BAND, ec='none', zorder=0))


def box(x0, y0, w, h, text, fc='white', ec=INK, tc=INK, fs=6.4, lw=0.7, z=4):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                                fc=fc, ec=ec, lw=lw, zorder=z))
    ax.text(x0 + w / 2, y0 + h / 2, text, ha='center', va='center', fontsize=fs, color=tc, zorder=z + 1)


def line(xs, ys, color=INK, lw=0.8, ls='-', z=2):
    ax.add_line(Line2D(xs, ys, color=color, lw=lw, ls=ls, zorder=z, solid_capstyle='butt'))


def arrow(p0, p1, color=GREY, lw=0.6, z=2):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle='-|>,head_length=2.4,head_width=1.2', color=color,
                                 lw=lw, shrinkA=0, shrinkB=0, zorder=z))


def dot(x, y, color=GREY):
    ax.plot(x, y, 'o', ms=2.2, color=color, zorder=3)


def core(x, y):
    ax.add_patch(Circle((x, y), 0.24, fc='none', ec=INK, lw=0.7, zorder=3))


def vt(x, y):
    for dy in (0.17, -0.17):
        ax.add_patch(Circle((x, y + dy), 0.26, fc='none', ec=INK, lw=0.7, zorder=3))


# =============================================================== (a) base unit
ax.text(4.8, 12.35, r'(a) Base unit ($N=1$)', ha='center', va='center', fontsize=7.5)
fx = 4.6
line([0.6, 8.9], [11.3, 11.3], lw=2.6)
ax.text(8.9, 11.6, 'Busbar', ha='right', va='bottom', fontsize=6.2)
# VT1 on the busbar
line([1.8, 1.8], [11.3, 10.85])
vt(1.8, 10.52)
ax.text(2.2, 10.52, 'VT1', ha='left', va='center', fontsize=6.4)
# disconnector, CT1, CB, CT2, VT2, line
line([fx, fx], [11.3, 10.85]); line([fx, fx + 0.38], [10.85, 10.42]); line([fx, fx], [10.44, 9.99])
core(fx, 9.75); core(fx, 9.25)
ax.text(fx + 0.38, 9.75, 'CT1', ha='left', va='center', fontsize=6.4)
line([fx, fx], [9.01, 8.65])
box(fx - 0.5, 7.95, 1.0, 0.7, 'CB')
line([fx, fx], [7.95, 7.39])
core(fx, 7.15); core(fx, 6.65)
ax.text(fx + 0.38, 7.15, 'CT2', ha='left', va='center', fontsize=6.4)
line([fx, fx], [6.41, 6.05])
line([fx, fx + 0.9], [6.05, 6.05])
vt(fx + 0.9, 5.72)
ax.text(fx + 1.25, 5.72, 'VT2', ha='left', va='center', fontsize=6.4)
arrow((fx, 6.05), (fx, 5.15), color=INK, lw=0.8)
ax.text(fx - 0.15, 5.4, 'Line', ha='right', va='center', fontsize=6.2)

# merging units
box(0.5, 3.6, 1.9, 0.65, 'MU1B1', fc=INK, tc='white')
box(6.8, 3.6, 1.9, 0.65, 'MU2B1', fc=INK, tc='white')
# secondary wiring: first core of each CT to MU1, second core to MU2, both VTs to both MUs
line([fx - 0.24, 1.2], [9.75, 9.75], color=GREY, lw=0.6); arrow((1.2, 9.75), (1.2, 4.25))
line([fx - 0.24, 1.6], [7.15, 7.15], color=GREY, lw=0.6); arrow((1.6, 7.15), (1.6, 4.25))
line([fx + 0.24, 7.9], [9.25, 9.25], color=GREY, lw=0.6); arrow((7.9, 9.25), (7.9, 4.25))
line([fx + 0.24, 7.5], [6.65, 6.65], color=GREY, lw=0.6); arrow((7.5, 6.65), (7.5, 4.25))
line([1.8, 1.8], [10.2, 10.05], color=GREY, lw=0.6); dot(1.8, 10.05)
line([0.8, 8.3], [10.05, 10.05], color=GREY, lw=0.6, z=1)
arrow((0.8, 10.05), (0.8, 4.25)); arrow((8.3, 10.05), (8.3, 4.25))
line([fx + 0.9, fx + 0.9], [5.4, 4.8], color=GREY, lw=0.6); dot(fx + 0.9, 4.8)
line([2.0, 7.1], [4.8, 4.8], color=GREY, lw=0.6)
arrow((2.0, 4.8), (2.0, 4.25)); arrow((7.1, 4.8), (7.1, 4.25))
# breaker control (dashed)
line([2.25, 2.25, fx - 0.5], [4.25, 8.3, 8.3], lw=0.6, ls=(0, (3, 1.6)))
line([6.95, 6.95, fx + 0.5], [4.25, 8.3, 8.3], lw=0.6, ls=(0, (3, 1.6)))

# process bus switch, devices and station bus
box(0.5, 2.3, 8.2, 0.55, 'SW1 (process bus)', fc=PB, ec=PB, tc='white')
for x in (1.45, 7.75):
    line([x, x], [3.6, 2.85], color=PB, lw=1.1)
w, g = 1.52, 0.15
for i, d in enumerate(['PP1B1L', 'PP2B1L', 'BBP', 'SMC', 'GPS']):
    xd = 0.5 + i * (w + g)
    box(xd, 1.05, w, 0.55, d, fs=5.6)
    line([xd + w / 2, xd + w / 2], [1.6, 2.3], color=PB, lw=1.1)
    if i < 3:
        line([xd + w / 2, xd + w / 2], [1.05, 0.35], color=SB, lw=1.1)
line([0.5, 4.95], [0.35, 0.35], color=SB, lw=2.2)
ax.text(5.15, 0.35, 'Station bus', ha='left', va='center', fontsize=6.2)

# =============================================================== (b) modular expansion
ax.text(17.2, 12.35, r'(b) Modular expansion ($N=1,\ldots,10$)', ha='center', va='center', fontsize=7.5)


def bay(x0, label, n):
    w, h = 2.05, 3.0
    ax.add_patch(FancyBboxPatch((x0, 7.4), w, h, boxstyle='round,pad=0,rounding_size=0.15',
                                fc='white', ec=INK, lw=0.7, zorder=3))
    ax.text(x0 + w / 2, 10.7, label, ha='center', va='bottom', fontsize=6.4)
    for i, d in enumerate([f'MU1B{n}', f'MU2B{n}', f'PP1B{n}L', f'PP2B{n}L']):
        ax.text(x0 + w / 2, 9.95 - 0.68 * i, d, ha='center', va='center', fontsize=5.9, zorder=4)
    line([x0 + w / 2, x0 + w / 2], [7.4, 6.45], color=PB, lw=1.1)


bay(10.4, 'Bay 1', '1')
bay(12.65, 'Bay 2', '2')
ax.text(15.1, 8.9, r'$\cdots$', ha='center', va='center', fontsize=9)
bay(15.55, r'Bay $k=\lfloor N/2\rfloor$', '$k$')
bay(18.55, r'Bay $k{+}1$', '$k{+}1$')
ax.text(21.0, 8.9, r'$\cdots$', ha='center', va='center', fontsize=9)
bay(21.45, r'Bay $N$', '$N$')
box(10.4, 5.75, 7.2, 0.7, 'SW1', fc=PB, ec=PB, tc='white', fs=6.8)
box(18.55, 5.75, 4.95, 0.7, 'SW2', fc=PB, ec=PB, tc='white', fs=6.8)
line([17.6, 18.55], [6.1, 6.1], lw=2.6, z=5)
ax.text(18.07, 5.55, 'trunk', ha='center', va='top', fontsize=6.2)
for i, d in enumerate(['BBP', 'SMC', 'GPS']):
    xd = 10.9 + i * 2.2
    box(xd, 3.7, 1.6, 0.6, d, fs=6.2)
    if d == 'BBP':
        line([xd + 0.8, xd + 0.8], [4.3, 5.75], lw=2.6, z=5)
    else:
        line([xd + 0.8, xd + 0.8], [4.3, 5.75], color=PB, lw=1.1)
ax.text(10.4, 2.7, r'$N\leq 5$: every device on SW1 ($4N+3$ ports).', ha='left', va='center', fontsize=6.4)
ax.text(10.4, 1.9, r'$N\geq 6$: bays $1,\ldots,\lfloor N/2\rfloor$ on SW1 with BBP, SMC and GPS;',
        ha='left', va='center', fontsize=6.4)
ax.text(10.4, 1.2, 'the remaining bays on SW2, joined by the trunk.', ha='left', va='center', fontsize=6.4)
ax.text(10.4, 0.4, 'Station bus not drawn.', ha='left', va='center', fontsize=6.0, color='#55554f')

# legend
h = [Line2D([], [], color=PB, lw=1.4, label='Process-bus link (100 Mbps)'),
     Line2D([], [], color=INK, lw=2.6, label='Upgraded to 1 Gbps (BBP link, trunk)'),
     Line2D([], [], color=SB, lw=2.0, label='Station bus'),
     Line2D([], [], color=GREY, lw=0.8, label='CT/VT secondary wiring'),
     Line2D([], [], color=INK, lw=0.8, ls=(0, (3, 1.6)), label='Breaker control')]
ax.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, -0.005), ncol=5, frameon=False, fontsize=6.2,
          handlelength=2.2, columnspacing=1.1, handletextpad=0.5)

out = os.path.join('results', 'paper')
os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, 'fig_case_study.pdf'))
fig.savefig(os.path.join(out, 'fig_case_study.png'))
print('written')
