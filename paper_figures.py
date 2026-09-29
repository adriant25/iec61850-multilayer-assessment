# -*- coding: utf-8 -*-
"""
paper_figures.py -- Publication figures for the journal paper.

Reads the analytical sweep (results/Global_Report.xlsx, written by
simulation.py) and the DES reference table (des_reference.csv, written by
extract_des_reference.py) and writes IEEE-sized figures to results/paper/:

  fig_structural.png     lambda_2 (log), global efficiency E, load Gini vs N
  fig_sv_validation.png  SV end-to-end delay and PLR: analytical vs DES
  fig_critical_nodes.png steady-state BC ranking and vulnerability index V
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import T_MAX, TRAFFIC_SPECS, CAPACITY_DEFAULT

OUT = os.path.join('results', 'paper')
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.7, 'lines.linewidth': 1.2, 'lines.markersize': 4,
    'grid.linewidth': 0.35, 'grid.alpha': 0.5, 'grid.linestyle': '--',
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

# Color-blind-safe palette (Wong 2011)
STYLE = {
    'base':     dict(color='#0072B2', marker='o', ls='-',  label='Base (all links 100 Mbps)'),
    'bbp_only': dict(color='#E69F00', marker='^', ls='-.', label='BBP link 1 Gbps'),
    'upgraded': dict(color='#009E73', marker='s', ls='--', label='Trunk + BBP link 1 Gbps'),
}
ORDER = ['base', 'bbp_only', 'upgraded']

# Stability boundary of the BBP egress port: 2 SV streams per bay
SV_BPS_PER_BAY = 2 * TRAFFIC_SPECS['SV']['size'] * 8 * TRAFFIC_SPECS['SV']['freq']
N_STAR = CAPACITY_DEFAULT / SV_BPS_PER_BAY

report = pd.read_excel(os.path.join('results', 'Global_Report.xlsx'), sheet_name=None)
raw = report['Raw_Data']
per_n = (raw.groupby(['N_Bays', 'Scenario'])
            [['Fiedler_Lambda2', 'Global_Efficiency', 'Load_Gini',
              'E2E_Avg_SV_us', 'E2E_PLR_Avg_SV']]
            .mean().reset_index())


def by_scenario(sc):
    return per_n[per_n.Scenario == sc].sort_values('N_Bays')


def mark_n_star(ax):
    ax.axvline(N_STAR, color='0.4', lw=0.8, ls=':')


# ---------------------------------------------------------------------------
# Fig. structural indicators
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.2), gridspec_kw={'wspace': 0.32})
panels = [('Fiedler_Lambda2', r'$\lambda_2$ (conductance-weighted)', True, '(a) Algebraic connectivity'),
          ('Global_Efficiency', r'$E$ (ms$^{-1}$)', False, '(b) Global efficiency'),
          ('Load_Gini', r'Load Gini $G$', False, '(c) Load Gini coefficient')]
for ax, (col, ylab, logy, title) in zip(axes, panels):
    for sc in ORDER:
        d = by_scenario(sc)
        st = STYLE[sc]
        ax.plot(d.N_Bays, d[col], color=st['color'], marker=st['marker'],
                ls=st['ls'], label=st['label'])
    ax.axvline(5.5, color='0.6', lw=0.8, ls='--')
    mark_n_star(ax)
    if logy:
        ax.set_yscale('log')
    ax.set_xticks(range(1, 11))
    ax.set_xlabel('Number of bays $N$')
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.grid(True, which='both')
axes[0].text(5.6, axes[0].get_ylim()[1] * 0.5, 'trunk\nadded', fontsize=6.5, color='0.35')
axes[2].text(N_STAR + 0.1, 0.80, r'$N^*$', fontsize=7, color='0.35')
axes[1].legend(loc='lower left', frameon=True)
fig.savefig(os.path.join(OUT, 'fig_structural.png'))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig. SV validation: analytical vs DES
# ---------------------------------------------------------------------------
des = pd.read_csv('des_reference.csv')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={'wspace': 0.28})
for sc in ORDER:
    d = by_scenario(sc)
    st = STYLE[sc]
    ax1.plot(d.N_Bays, d.E2E_Avg_SV_us, color=st['color'], ls=st['ls'],
             label=f"Analytical: {st['label']}")
    ax2.plot(d.N_Bays, d.E2E_PLR_Avg_SV, color=st['color'], ls=st['ls'],
             label=f"Analytical: {st['label']}")
for sc, mk in (('base', 'o'), ('upgraded', 's')):
    d = des[des.Scenario == sc].sort_values('N_Bays')
    st = STYLE[sc]
    ax1.plot(d.N_Bays, d.SV_mean_us, ls='none', marker=mk, mfc='white',
             mec=st['color'], mew=1.0, ms=5, label=f"DES: {st['label']}")
    ax2.plot(d.N_Bays, d.SV_PLR_pct, ls='none', marker=mk, mfc='white',
             mec=st['color'], mew=1.0, ms=5, label=f"DES: {st['label']}")
ax1.axhline(T_MAX * 1e6, color='#D55E00', lw=1.0, ls=':', label='IEC 61850 limit (3 ms)')
for ax in (ax1, ax2):
    mark_n_star(ax)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel('Number of bays $N$')
    ax.grid(True, which='both')
ax1.set_yscale('log')
ax1.set_ylabel(r'Mean SV end-to-end delay ($\mu$s)')
ax1.set_title('(a) SV delay')
ax1.text(N_STAR + 0.1, 40, r'$N^*=%.2f$' % N_STAR, fontsize=7, color='0.35')
ax2.set_ylabel('Mean SV packet loss ratio (%)')
ax2.set_title('(b) SV packet loss (7 s window)')
ax1.legend(loc='upper left', fontsize=6, frameon=True)
fig.savefig(os.path.join(OUT, 'fig_sv_validation.png'))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig. critical nodes (N = 10): BC ranking and vulnerability
# ---------------------------------------------------------------------------
N_FOCUS = 10


def short(lbl: str) -> str:
    return (lbl.replace('L4_Priority_', 'Queue ').replace('_Prio_', ' P')
               .replace('L4_Trunk_', 'Trunk ').replace('_to_', '$\\to$')
               .replace('L5_OutPort_', 'Egress ').replace('L1_Equipment_', 'Dev. ')
               .replace('L3_VLAN_', 'VLAN '))


steady = raw[(raw.N_Bays == N_FOCUS) & (raw.Scenario == 'base') & (raw.Snapshot_Time_ms == 0)]
bc = steady.set_index('Node_Label').Centrality.sort_values(ascending=False).head(8)

vul = report['Vulnerability'].drop_duplicates(['N_Bays', 'Scenario', 'Node_Label'])
vul = vul[vul.N_Bays == N_FOCUS]
top_v = (vul[vul.Scenario == 'base'].sort_values('Vulnerability_Index', ascending=False)
         .Node_Label.head(6).tolist())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw={'wspace': 0.55})
ax1.barh([short(l) for l in bc.index[::-1]], bc.values[::-1], color='#0072B2', edgecolor='black', lw=0.4)
ax1.set_xlabel('Normalized betweenness centrality')
ax1.set_title(f'(a) BC, $N={N_FOCUS}$, base, steady state')
ax1.grid(True, axis='x')

y = np.arange(len(top_v))
h = 0.26
for i, sc in enumerate(ORDER):
    vals = [vul[(vul.Scenario == sc) & (vul.Node_Label == l)].Vulnerability_Index.max() for l in top_v]
    ax2.barh(y[::-1] + (1 - i) * h, np.nan_to_num(vals), height=h, color=STYLE[sc]['color'],
             edgecolor='black', lw=0.4, label=STYLE[sc]['label'])
ax2.set_yticks(y[::-1])
ax2.set_yticklabels([short(l) for l in top_v])
ax2.set_xlabel('Node vulnerability index $V$')
ax2.set_title(f'(b) Vulnerability, $N={N_FOCUS}$')
ax2.grid(True, axis='x')
ax2.legend(loc='lower right', fontsize=6, frameon=True)
fig.savefig(os.path.join(OUT, 'fig_critical_nodes.png'))
plt.close(fig)

# ---------------------------------------------------------------------------
# Fig. DES time profile of an SV flow to the BBP vs fluid model
# ---------------------------------------------------------------------------
from analysis import fluid_overload  # noqa: E402
from config import PORT_BUFFER_BYTES  # noqa: E402

SIM = r"C:\Users\adria\OneDrive - Universidad de los andes\Uniandes\2025-1\Simulación"
TRACES = {
    9:  os.path.join(SIM, r'CASO9BAHIAS\100\Results\Detailed_Plots\Flow_Plots'
                          r'\SV_MU1B2_5_BBP\delay_data_SV_MU1B2_5_BBP.xlsx'),
    10: os.path.join(SIM, r'CASO10BAHIAS\Results100\Detailed_Plots\Detailed_Plots\Flow_Plots'
                          r'\SV_MU1B2_5_BBP\delay_data_SV_MU1B2_5_BBP.xlsx'),
}
T_EVENT = 3.0
fig, ax = plt.subplots(figsize=(3.5, 2.4))
for n, color in ((9, '#0072B2'), (10, '#D55E00')):
    tr = pd.read_excel(TRACES[n])
    idx = pd.to_numeric(tr['Packet_Index'].astype(str).str.extract(r'(\d+)')[0])
    t = idx / TRAFFIC_SPECS['SV']['freq']
    dly = pd.to_numeric(tr['Total_Delay_ms'], errors='coerce')
    ax.plot(t[::20], dly[::20], color=color, lw=0.9, label=f'DES, $N={n}$')
    rho = n * SV_BPS_PER_BAY / CAPACITY_DEFAULT
    tt = np.linspace(0, 7, 200)
    cap_ms = 8 * PORT_BUFFER_BYTES / CAPACITY_DEFAULT * 1e3
    ax.plot(tt, np.minimum((rho - 1) * tt * 1e3, cap_ms), color=color, lw=1.0, ls='--',
            label=f'Fluid model, $\\rho={rho:.3f}$')
ax.axvline(T_EVENT, color='0.3', lw=0.8, ls=':')
ax.text(T_EVENT + 0.08, 20, '50BF event', fontsize=6.5, color='0.3')
ax.axhline(T_MAX * 1e3, color='0.5', lw=0.6)
ax.set_xlabel('Simulation time (s)')
ax.set_ylabel('SV delay MU1B2$\\to$BBP (ms)')
ax.set_xlim(0, 7)
ax.grid(True)
ax.legend(loc='upper left', fontsize=6, frameon=True)
fig.savefig(os.path.join(OUT, 'fig_des_timeline.png'))
plt.close(fig)

print(f'N* = {N_STAR:.3f}; figures written to {OUT}')
