# -*- coding: utf-8 -*-
"""Theory-versus-DES plots for the verification cases (validate_des.py, validate_extra.py)."""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
rd = lambda f: pd.read_csv(os.path.join(OUT, f))
s = rd('des_validation_summary.csv')
ov = rd('des_validation_overload.csv')
pr = rd('des_validation_priority_reps.csv')
cc = rd('des_validation_ccdf.csv')
ls = rd('des_validation_loss_series.csv')
lo = rd('des_validation_loss.csv')
mx = rd('des_validation_mixed.csv')
S_US = 149 * 8 / 100.0                     # service time of one SV frame [us]

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                     'font.size': 8, 'legend.fontsize': 6, 'lines.markersize': 4,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})
fig, ax = plt.subplots(2, 4, figsize=(7.16, 4.4), gridspec_kw={'wspace': 0.6, 'hspace': 0.65})
ax = ax.ravel()

# (a) single queue means
r = np.linspace(0.05, 0.92, 200)
for case, th, c, mk, lab in (('A_M/D/1', lambda x: x * S_US / (2 * (1 - x)), '#0072B2', 'o', 'M/D/1'),
                             ('A_M/M/1', lambda x: x * S_US / (1 - x), '#D55E00', 's', 'M/M/1')):
    x = s[s.Case == case]
    ax[0].plot(r, th(r), color=c, lw=1, label=f'{lab} theory')
    ax[0].plot(x.rho, x.DES_wait_us, mk, color=c, mfc='none', label=f'{lab} DES')
ax[0].set(xlabel=r'Utilization $\rho$', ylabel=r'Mean wait ($\mu$s)', title='(a) Single queue')

# (b) N*D/D/1 means
x = s[s.Case == 'B_N*D/D/1']
ax[1].plot(x.rho, x.Theory_wait_us, '-', color='#009E73', lw=1, label=r'N$\cdot$D/D/1 theory')
ax[1].errorbar(x.rho, x.DES_wait_us, yerr=x.DES_ci95_us, fmt='o', color='#009E73', mfc='none',
               capsize=2, label='DES (95% CI)')
ax[1].set(xlabel=r'Utilization $\rho$ ($K$ SV streams)', ylabel=r'Mean wait ($\mu$s)',
          title='(b) Periodic SV streams')

# (c) strict priority with replications
for name, c, mk in (('high', '#CC79A7', '^'), ('low', '#0072B2', 'v')):
    x = pr[pr.Class == name]
    ax[2].plot(x.rho, x.Theory_wait_us, '-', color=c, lw=1, label=f'{name} prio. Cobham')
    ax[2].errorbar(x.rho, x.DES_wait_us, yerr=x.DES_ci95_us, fmt=mk, color=c, mfc='none', capsize=2,
                   label=f'{name} prio. DES')
ax[2].set(xlabel=r'Total utilization $\rho$', ylabel=r'Mean wait ($\mu$s)', title='(c) Strict priority')

# (d) overload growth
ax[3].plot(ov.t_gen * 1e3, ov.DES_wait_us / 1e3, '.', color='0.5', ms=1, label='DES')
ax[3].plot(ov.t_gen * 1e3, ov.Fluid_wait_us / 1e3, '-', color='k', lw=1, label=r'Fluid $(\rho-1)t$')
ax[3].set(xlabel='Time (ms)', ylabel='Wait (ms)', title=r'(d) Overload, $\rho=1.14$')

# (e) N*D/D/1 distribution
for m, c in (('N*D/D/1 K=12', '#E69F00'), ('N*D/D/1 K=16', '#009E73')):
    x = cc[(cc.Model == m) & (cc.x_service_units > 0)]
    ax[4].semilogy(x.x_service_units, x.Theory_ccdf.clip(lower=1e-6), '-', color=c, lw=1,
                   label=f'{m[-4:]} Benes tail')
    ax[4].semilogy(x.x_service_units, x.DES_ccdf.clip(lower=1e-6), 'o', color=c, mfc='none', ms=2.5,
                   label=f'{m[-4:]} DES')
ax[4].set(xlabel='Wait $x$ (service times)', ylabel=r'$P(W>x)$', title='(e) SV wait distribution',
          ylim=(1e-4, 1.2))

# (f) M/D/1 and M/M/1 distributions
for m, c in (('M/D/1 rho=0.8', '#0072B2'), ('M/M/1 rho=0.8', '#D55E00')):
    x = cc[(cc.Model == m) & (cc.x_service_units > 0)]
    ax[5].semilogy(x.x_service_units, x.Theory_ccdf.clip(lower=1e-6), '-', color=c, lw=1,
                   label=f'{m[:5]} theory')
    ax[5].semilogy(x.x_service_units, x.DES_ccdf.clip(lower=1e-6), 'o', color=c, mfc='none', ms=2.5,
                   label=f'{m[:5]} DES')
ax[5].set(xlabel='Wait $x$ (service times)', ylabel=r'$P(W>x)$', title=r'(f) Distribution, $\rho=0.8$',
          ylim=(1e-3, 1.2))

# (g) finite buffer: fill and plateau
for k, c in ((20, '#0072B2'), (24, '#D55E00')):
    x = ls[ls.K == k]
    row = lo[lo.K == k].iloc[0]
    ax[6].plot(x.t_gen * 1e3, x.w_egr * 1e3, '.', color=c, ms=1,
               label=rf'DES $\rho$={row.rho:.2f}, loss {100 * row.Loss_ratio_DES:.1f}%')
    tf = row.Fill_time_fluid_ms
    tt = np.linspace(0, 400, 400)
    ax[6].plot(tt, np.minimum((row.rho - 1) * tt, row.Plateau_wait_fluid_ms), '-', color='k', lw=0.8)
ax[6].plot([], [], '-', color='k', lw=0.8, label='Fluid (12.6%, 27.2% loss)')
ax[6].set(xlabel='Time (ms)', ylabel='Wait (ms)', title='(g) Finite buffer (100 kB)')

# (h) mixed traffic: DES vs analytic model
classes = ['P6 Poisson', 'P4 SV periodic', 'P1 Poisson']
xs = np.arange(len(classes))
for i, (rh, c) in enumerate(((0.05, '#0072B2'), (0.2, '#D55E00'))):
    x = mx[mx.rho_P6 == rh].set_index('Class').loc[classes]
    off = (i - 0.5) * 0.3
    ax[7].bar(xs + off, x.Theory_wait_us, width=0.28, color=c, alpha=0.35,
              label=rf'Model, $\rho_{{P6}}$={rh}')
    ax[7].errorbar(xs + off, x.DES_wait_us, yerr=x.DES_ci95_us, fmt='o', color=c, mfc='none',
                   capsize=2, label=rf'DES, $\rho_{{P6}}$={rh}')
ax[7].set_xticks(xs, ['P6\nPoisson', 'P4 SV\nperiodic', 'P1\nPoisson'])
ax[7].set(ylabel=r'Mean wait ($\mu$s)', title='(h) Mixed traffic')

locs = ['upper left'] * 5 + ['upper right', 'center right', 'upper left']
for a, loc in zip(ax, locs):
    a.grid(True, ls='--', lw=0.35, alpha=0.5)
    a.legend(frameon=False, loc=loc)
fig.savefig(os.path.join(OUT, 'fig_des_validation.png'))
print('figure written')
