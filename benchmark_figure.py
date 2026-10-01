# -*- coding: utf-8 -*-
"""
benchmark_figure.py -- Proposed model vs network calculus vs DES, N = 1..10.

(a) Base scenario: network-calculus bound of the worst SV flow to the BBP
    (steady state and all-bay burst peak), DES largest and mean SV delay,
    and the proposed (mean) model, with the 0.6 ms network budget and the
    3 ms transfer-time limit.
(b) Ratio of the network-calculus bound to the largest delay observed in the
    DES for every compliant configuration.

Inputs: results/nc_bound.csv (nc_bound.py), des_reference.csv,
        analytic_vs_des.csv (compare_des.py)
Output: results/paper/fig_benchmark.png, results/benchmark_table.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

nc = pd.read_csv(os.path.join('results', 'nc_bound.csv'))
des = pd.read_csv('des_reference.csv')
an = pd.read_csv('analytic_vs_des.csv')

b = nc[nc.Scenario == 'base'].pivot(index='N_Bays', columns='Snapshot', values='NC_bound_us')
d = des[des.Scenario == 'base'].set_index('N_Bays')
a = an[an.Scenario == 'base'].set_index('N')
tab = pd.DataFrame({'NC_steady_us': b['steady'], 'NC_burst_us': b['burst'],
                    'DES_max_us': d.SV_max_us, 'DES_mean_us': d.SV_mean_us,
                    'Proposed_mean_us': a.SV_an})
tab['NC_over_DES_max'] = tab.NC_burst_us / tab.DES_max_us
up = nc[nc.Scenario == 'upgraded'].pivot(index='N_Bays', columns='Snapshot', values='NC_bound_us')
dup = des[des.Scenario == 'upgraded'].set_index('N_Bays')
tab_up = pd.DataFrame({'NC_burst_us': up['burst'], 'DES_max_us': dup.SV_max_us}).dropna()
tab_up['NC_over_DES_max'] = tab_up.NC_burst_us / tab_up.DES_max_us
tab.to_csv(os.path.join('results', 'benchmark_table.csv'), float_format='%.2f')
print(tab.round(2).to_string())
print(tab_up.round(2).to_string())

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 8, 'legend.fontsize': 6.5, 'lines.markersize': 4,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6), gridspec_kw={'wspace': 0.3, 'width_ratios': [1.6, 1]})
n = tab.index.values
fin = np.isfinite(tab.NC_burst_us)
ax1.semilogy(n[fin], tab.NC_burst_us[fin], 'v-', color='#D55E00', label='Network calculus bound (burst)')
ax1.semilogy(n[fin], tab.NC_steady_us[fin], 'v--', color='#D55E00', mfc='none', label='Network calculus bound (steady)')
ax1.semilogy(n, tab.DES_max_us, 's-', color='0.35', label='DES, largest frame delay')
ax1.semilogy(n, tab.DES_mean_us, 'o-', color='#0072B2', mfc='none', label='DES, mean')
ax1.semilogy(n, tab.Proposed_mean_us, 'x--', color='#009E73', label='Proposed model, mean')
for y, lab in ((600, 'network budget 0.6 ms'), (3000, 'IEC 61850 limit 3 ms')):
    ax1.axhline(y, color='k', lw=0.7, ls=':')
    ax1.text(0.7, y * 1.12, lab, fontsize=6.5)
ax1.annotate(r'NC bound $\to\infty$ for $N\geq9$ ($\rho\geq1$)', (6.2, 1.1e4), fontsize=6.5,
             color='#D55E00', ha='center')
ax1.set_xticks(range(1, 11))
ax1.set_xlabel('Number of bays $N$')
ax1.set_ylabel(r'SV delay towards the BBP ($\mu$s)')
ax1.set_title('(a) Base design (100 Mbps)')
ax1.legend(frameon=False, loc='upper left', fontsize=6)
ax1.set_ylim(15, 1e6)

ok = tab[fin]
ax2.plot(ok.index, ok.NC_over_DES_max, 'v-', color='#D55E00', label='Base, burst bound')
ax2.plot(tab_up.index, tab_up.NC_over_DES_max, '^', color='#CC79A7', label='Trunk + BBP 1 Gbps')
ax2.set_xticks(range(1, 11))
ax2.set_xlabel('Number of bays $N$')
ax2.set_ylabel('NC bound / DES largest delay')
ax2.set_title('(b) Pessimism of the bound')
ax2.set_ylim(0, 10)
ax2.legend(frameon=False, loc='lower right')
for ax in (ax1, ax2):
    ax.grid(True, ls='--', lw=0.35, alpha=0.5)
os.makedirs(os.path.join('results', 'paper'), exist_ok=True)
fig.savefig(os.path.join('results', 'paper', 'fig_benchmark.png'))
print('figure written')
