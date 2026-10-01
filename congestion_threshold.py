# -*- coding: utf-8 -*-
"""
congestion_threshold.py -- Topological quantification of the bottleneck (congestion
threshold map) and its growth during the protection event.

Rate-weighted flow betweenness of node v (the analogue of the betweenness that
sets the congestion threshold in Zhao et al. 2005, lambda_c = C_max (N-1)/B_max):

    B_F(v) = sum over time-critical and background flows f with v in P_f of r_f

so that the utilization of a queue is rho_v = B_F(v) / C_v and the network is
stable while max_v B_F(v)/C_v < 1. B_F(v) grows with the number of bays N; the
critical size of each port, N*_v, solves B_F(v; N) = C_v (linear fit of the
sweep), the smallest N*_v locates the bottleneck and the ordering gives the
next ones. The event is added as the burst-peak load of (a) one bay and
(b) all bays, i.e. the critical size the port would have if the peak load
were sustained.

The GOOSE delay increase of the event is decomposed into the contention among
the frames released together (synchronized-release term) and the rest
(extra queuing from the burst load).

Outputs: results/bottleneck_map.csv, results/bottleneck_event.csv,
         results/paper/threshold_table.tex, results/paper/fig_threshold_map.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from config import GOOSE_VLAN_IDS, get_vlan_id
from model import (generate_devices, generate_demand_tensor, update_goose_demand,
                   build_topology, event_burst_vlans)
from analysis import (_port_cumulative_rho, _synchronized_burst_waits,
                      compute_weights, compute_e2e_metrics)

OUT = os.path.join('results', 'paper')
os.makedirs(OUT, exist_ok=True)
N_SWEEP = range(1, 31)
FIT_FROM = 6          # two-switch regime


def total_rho(D, sw, scenario):
    """{port: (rho over all classes, capacity)}."""
    return {p: v for (p, k), v in _port_cumulative_rho(D, sw, scenario).items() if k == 0}


def port_name(p: str) -> str:
    return (p.replace('L5_OutPort_', 'Egress ').replace('L4_Trunk_', 'Trunk ')
             .replace('_to_', '$\\to$'))


def sweep(scenario: str) -> pd.DataFrame:
    rows = []
    for n in N_SWEEP:
        devs, sw = generate_devices(n)
        D0 = generate_demand_tensor(n, devs, 1000.0)
        loads = {'steady': total_rho(D0, sw, scenario),
                 'single': total_rho(update_goose_demand(D0, n, 4.0, burst_bays=[n]), sw, scenario),
                 'all': total_rho(update_goose_demand(D0, n, 4.0), sw, scenario)}
        for p, (rho, cap) in loads['steady'].items():
            if p.startswith('L5_OutPort_') and p[len('L5_OutPort_'):] not in ('BBP', 'SMC'):
                continue                       # bay-device ports do not grow with N
            rows.append({'N_Bays': n, 'Port': p, 'C_Mbps': cap / 1e6,
                         'BF_Mbps': rho * cap / 1e6, 'Rho_steady': rho,
                         'Rho_single': loads['single'][p][0], 'Rho_all': loads['all'][p][0]})
    return pd.DataFrame(rows)


def critical_size(d: pd.DataFrame, col: str) -> tuple[float, float]:
    """(N* from linear fit over the two-switch regime, slope in rho per bay)."""
    x = d[d.N_Bays >= FIT_FROM]
    b, a = np.polyfit(x.N_Bays, x[col], 1)
    return ((1.0 - a) / b if b > 1e-9 else np.inf), b


def threshold_map(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, d in df.groupby('Port'):
        r = {'Port': p, 'C_Mbps': d.C_Mbps.iloc[0]}
        for tag in ('steady', 'single', 'all'):
            r[f'Nstar_{tag}'], r[f'Slope_{tag}'] = critical_size(d, f'Rho_{tag}')
        fail = d[d.Rho_steady >= 1.0]
        r['First_FAIL_N'] = int(fail.N_Bays.min()) if not fail.empty else np.nan
        r['BF_per_bay_Mbps'] = r['Slope_steady'] * r['C_Mbps']
        rows.append(r)
    return pd.DataFrame(rows).sort_values('Nstar_steady')


def event_decomposition(n: int, scenario: str = 'base') -> dict:
    """GOOSE event delay increase and its synchronized-release share."""
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    labels, A = build_topology(n, devs, D0, sw)
    out = {'N_Bays': n}
    for scope, bays in (('single', [n]), ('all', None)):
        vlans = set(event_burst_vlans(n, bays))
        res = {}
        for tag, dt in (('steady', 1000.0), ('peak', 4.0)):
            D = update_goose_demand(D0, n, dt, burst_bays=bays)
            W, J, PL, u = compute_weights(labels, A, D, sw, devs, dt, scenario)
            res[tag] = compute_e2e_metrics(labels, W, PL, D, sw, scenario, D_window=D0,
                                           synchronized_burst=(dt < 1000.0),
                                           burst_vlans=vlans, return_flows=True)
        D = update_goose_demand(D0, n, 4.0, burst_bays=bays)
        sync = _synchronized_burst_waits(D, sw, scenario, burst_vlans=vlans)
        ev = [(v, s, t) for (c, v, s, t, _, _) in res['peak']['flow_records']
              if c == 'GOOSE' and v in vlans]
        sync_mean = np.mean([sync.get(k, 0.0) for k in ev]) * 1e6
        d_ss = res['steady']['E2E_Avg_GOOSE_Event_us']
        d_pk = res['peak']['E2E_Avg_GOOSE_Event_us']
        out.update({f'{scope}_GOOSE_steady_us': d_ss, f'{scope}_GOOSE_peak_us': d_pk,
                    f'{scope}_increase_us': d_pk - d_ss, f'{scope}_sync_us': sync_mean,
                    f'{scope}_sync_share': sync_mean / (d_pk - d_ss) if d_pk > d_ss else np.nan})
    return out


if __name__ == '__main__':
    maps, sweeps = [], []
    for streams, scenario in ((2, 'base'), (4, 'base'), (4, 'bbp_only')):
        config.BBP_SV_STREAMS_PER_BAY = streams
        df = sweep(scenario)
        df['Streams'], df['Scenario'] = streams, scenario
        sweeps.append(df)
        m = threshold_map(df)
        m['Streams'], m['Scenario'] = streams, scenario
        maps.append(m)
        print(f'--- {streams} streams/bay, {scenario}')
        print(m.round(3).to_string(index=False))
    config.BBP_SV_STREAMS_PER_BAY = 2
    sw_df, mp = pd.concat(sweeps), pd.concat(maps)
    sw_df.to_csv(os.path.join('results', 'bottleneck_sweep.csv'), index=False, float_format='%.6f')
    mp.to_csv(os.path.join('results', 'bottleneck_map.csv'), index=False, float_format='%.4f')

    ev = pd.DataFrame([event_decomposition(n) for n in (1, 5, 8, 9, 10)])
    ev.to_csv(os.path.join('results', 'bottleneck_event.csv'), index=False, float_format='%.4f')
    print(ev.round(3).to_string(index=False))

    # ---- table ----
    label = {(2, 'base'): '2, 100 Mbps', (4, 'base'): '4, 100 Mbps',
             (4, 'bbp_only'): '4, BBP link 1 Gbps'}
    lines = [r'\begin{table}[!t]', r'\centering',
             r'\caption{Congestion threshold map: rate-weighted flow betweenness per bay and critical size $N^*_v$ of each port whose load grows with $N$, in steady state and if the burst-peak load of a one-bay or an all-bay event were sustained. The smallest $N^*_v$ locates the bottleneck; the next ones give the order in which ports saturate.}',
             r'\label{tab:thresholds}', r'\resizebox{\columnwidth}{!}{',
             r'\begin{tabular}{llcccc}', r'\toprule',
             r'SV streams to BBP, & Port & $\Delta B_F$ per bay & \multicolumn{3}{c}{$N^*_v$} \\',
             r'\cmidrule(lr){4-6}',
             r'links & & (Mbps) & Steady & One-bay peak & All-bay peak \\', r'\midrule']
    for key in ((2, 'base'), (4, 'base'), (4, 'bbp_only')):
        m = mp[(mp.Streams == key[0]) & (mp.Scenario == key[1])]
        m = m[m.Nstar_steady < 60].head(3)
        for i, (_, r) in enumerate(m.iterrows()):
            lines.append(f'{label[key] if i == 0 else ""} & {port_name(r.Port)} & '
                         f'{r.BF_per_bay_Mbps:.2f} & {r.Nstar_steady:.2f} & '
                         f'{r.Nstar_single:.2f} & {r.Nstar_all:.2f} \\\\')
        if key != (4, 'bbp_only'):
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    with open(os.path.join(OUT, 'threshold_table.tex'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ---- figure ----
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 8, 'legend.fontsize': 6.5, 'lines.markersize': 3.5,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={'wspace': 0.25}, sharey=True)
    colors = {'L5_OutPort_BBP': '#D55E00', 'L4_Trunk_SW2_to_SW1': '#0072B2'}
    for ax, key, title in ((axes[0], (2, 'base'), '(a) 2 SV streams per bay, 100 Mbps'),
                           (axes[1], (4, 'bbp_only'), '(b) 4 SV streams per bay, BBP link 1 Gbps')):
        d = sw_df[(sw_df.Streams == key[0]) & (sw_df.Scenario == key[1])]
        m = mp[(mp.Streams == key[0]) & (mp.Scenario == key[1])].set_index('Port')
        for p, c in colors.items():
            x = d[d.Port == p].sort_values('N_Bays')
            ax.fill_between(x.N_Bays, x.Rho_steady, x.Rho_all, color=c, alpha=0.2, lw=0)
            ax.plot(x.N_Bays, x.Rho_steady, color=c, marker='o',
                    label=port_name(p).replace('$\\to$', r'$\to$'))
            ns = m.loc[p, 'Nstar_steady']
            if ns < 18:
                ax.axvline(ns, color=c, lw=0.8, ls=':')
                ax.annotate(f'$N^*$={ns:.2f}', (ns, 0.05), textcoords='offset points', rotation=90,
                            xytext=(-9, 0), color=c, fontsize=7)
        ax.fill_between([], [], [], color='0.6', alpha=0.3, label='Steady to all-bay burst peak')
        ax.axhline(1.0, color='k', lw=0.8, ls='--')
        ax.set_xlim(0.5, 18.5)
        ax.set_xticks(range(1, 19))
        ax.set_ylim(0, 1.6)
        ax.set_xlabel('Number of bays $N$')
        ax.set_title(title)
        ax.grid(True, ls='--', lw=0.35, alpha=0.5)
    axes[0].set_ylabel(r'Utilization $\rho_v = B_F(v)/C_v$')
    axes[0].legend(frameon=False, loc='upper left')
    axes[1].legend(frameon=False, loc='upper left')
    fig.savefig(os.path.join(OUT, 'fig_threshold_map.png'))
    print('figure written')
