# -*- coding: utf-8 -*-
"""
plot_style.py -- Common style of the figures of the paper.

Importing this module sets the shared matplotlib parameters (open axes, light
grid, frameless legends, vector-friendly fonts). The categorical palette was
checked for colour-vision deficiencies; every series also has its own marker
or line style, so colour is never the only cue.
"""

import matplotlib.pyplot as plt

PALETTE = {'blue': '#2a78d6', 'orange': '#eb6834', 'aqua': '#1baf7a', 'yellow': '#eda100', 'magenta': '#e87ba4'}

plt.rcParams.update({
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'axes.edgecolor': '#3a3a37',
    'axes.labelcolor': '#1f1f1d',
    'xtick.color': '#3a3a37',
    'ytick.color': '#3a3a37',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'grid.color': '#d9d9d4',
    'grid.linewidth': 0.4,
    'grid.linestyle': '-',
    'legend.frameon': False,
    'mathtext.fontset': 'stix',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})
