# -*- coding: utf-8 -*-
"""
maintainability.py -- Availability and maintainability of the time-critical
flows on the multilayer graph (RAM extension).

Every node of a flow path is mapped to the physical component that hosts it:
  L1_Equipment_X             -> device X (IED, MU, BBP, SMC)
  L2_InPort_X, L5_OutPort_X  -> access link of X (fibre between X and its switch)
  L3 VLAN@SWk, L4_Priority_SWk -> switch SWk
  L4_Trunk_*                 -> inter-switch trunk
so that the component set C_f of every SV/GOOSE flow f is read from the graph.

With the steady-state unavailability U_c = MTTR / (MTTF + MTTR) of each
component (independent failures):
  structural flow availability  A_f = prod_{c in C_f} (1 - U_c)        (single LAN)
  PRP (two identical LANs)      A_f = prod_dev (1-U) * [1 - (1 - prod_net (1-U))^2]
  combined (RAM) availability   A_F = (1/|F|) sum_f 1[D_f <= T_req] * A_f
  component flow importance     I_c = |{f : c in C_f}| / |F|
  downtime share                S_c = U_c I_c / sum_k U_k I_k
  MTTR sensitivity              dT_down / dMTTR_c = I_c / MTTF_c   [min/yr per h]
  maintenance exposure          max over network components of I_c: the share
                                of protection flows interrupted while the most
                                critical network element is out for service.

D_f is the worst of the steady-state and event (first burst) delays.

Outputs: results/maintainability.csv, results/maintainability_components.csv,
         results/paper/maintainability_table.tex, results/paper/fig_maintainability.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plot_style  # noqa: E402,F401  (common style of the paper figures)

from config import T_MAX
from model import generate_devices, generate_demand_tensor, update_goose_demand, build_topology
from analysis import compute_weights, compute_e2e_metrics

OUT = os.path.join('results', 'paper')
os.makedirs(OUT, exist_ok=True)

HOURS_PER_YEAR = 8760.0
MIN_PER_YEAR = 525600.0
# MTTF [years]: IED, MU, time source 150 y and Ethernet switch 50 y
# (Kanabar & Sidhu 2009, as used by Mathebula & Saha 2022); optical fibre 300 y
# (Frei et al., US 8,886,475). MTTR 8 h (Kanabar & Sidhu 2009).
MTTF_Y = {'Device': 150.0, 'Switch': 50.0, 'Link': 300.0, 'Trunk': 300.0}
MTTR_BASE_H = 8.0
MTTR_SWEEP_H = (8.0, 24.0, 72.0)
NETWORK = ('Switch', 'Link', 'Trunk')


def component(node: str) -> tuple[str, str]:
    if node.startswith('L1_Equipment_'):
        return 'Device', node[len('L1_Equipment_'):]
    if node.startswith('L2_InPort_'):
        return 'Link', node[len('L2_InPort_'):]
    if node.startswith('L5_OutPort_'):
        return 'Link', node[len('L5_OutPort_'):]
    if node.startswith('L4_Trunk_'):
        return 'Trunk', 'SW1-SW2'
    if node.startswith('L4_Priority_'):
        return 'Switch', node.split('_')[2]
    return 'Switch', node.rsplit('_', 1)[1]          # L3 VLAN@switch


def unavail(kind: str, mttr_h: float) -> float:
    mttf_h = MTTF_Y[kind] * HOURS_PER_YEAR
    return mttr_h / (mttf_h + mttr_h)


def flows_for(n: int, scenario: str) -> list[dict]:
    """Time-critical flows with their component sets and worst delay."""
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    labels, A = build_topology(n, devs, D0, sw)
    worst = {}
    for dt in (1000.0, 4.0):
        D = update_goose_demand(D0, n, dt)
        W, J, PL, u = compute_weights(labels, A, D, sw, devs, dt, scenario)
        e2e = compute_e2e_metrics(labels, W, PL, D, sw, scenario, D_window=D0,
                                  synchronized_burst=(dt < 1000.0), return_flows=True)
        for cls, vlan, src, dst, path, d in e2e['flow_records']:
            key = (vlan, src, dst)
            if key not in worst or d > worst[key]['delay_s']:
                worst[key] = {'cls': cls, 'delay_s': d,
                              'comps': frozenset(component(p) for p in path)}
    return list(worst.values())


def availability(flows: list[dict], mttr_h: float, prp: bool) -> tuple[float, float]:
    """(mean structural flow availability, mean combined timing x structural)."""
    a_s, a_f = [], []
    for f in flows:
        dev = np.prod([1 - unavail(k, mttr_h) for k, _ in f['comps'] if k == 'Device'])
        net = np.prod([1 - unavail(k, mttr_h) for k, _ in f['comps'] if k in NETWORK])
        a = dev * (1 - (1 - net) ** 2) if prp else dev * net
        a_s.append(a)
        a_f.append(a * (f['delay_s'] <= T_MAX))
    return float(np.mean(a_s)), float(np.mean(a_f))


def importance(flows: list[dict]) -> dict:
    cnt = {}
    for f in flows:
        for c in f['comps']:
            cnt[c] = cnt.get(c, 0) + 1
    return {c: v / len(flows) for c, v in cnt.items()}


def label(c: tuple[str, str]) -> str:
    kind, name = c
    return {'Device': f'{name}', 'Link': f'{name} access link',
            'Switch': name, 'Trunk': 'Trunk SW1--SW2'}[kind]


if __name__ == '__main__':
    rows, comp_rows = [], []
    for scenario in ('base', 'bbp_only'):
        for n in range(1, 11):
            flows = flows_for(n, scenario)
            I = importance(flows)
            net = {c: v for c, v in I.items() if c[0] in NETWORK}
            top_net = max(net, key=net.get)
            r = {'N_Bays': n, 'Scenario': scenario, 'Flows': len(flows),
                 'Timing_OK': float(np.mean([f['delay_s'] <= T_MAX for f in flows])),
                 'Exposure': net[top_net], 'Exposure_Node': label(top_net)}
            for mttr in MTTR_SWEEP_H:
                for prp in (False, True):
                    a_s, a_f = availability(flows, mttr, prp)
                    tag = f'{"PRP" if prp else "Single"}_{int(mttr)}h'
                    r[f'Down_min_yr_{tag}'] = (1 - a_s) * MIN_PER_YEAR
                    r[f'A_F_{tag}'] = a_f
            rows.append(r)
            U = {c: unavail(c[0], MTTR_BASE_H) for c in I}
            tot = sum(U[c] * I[c] for c in I)
            for c, i in I.items():
                comp_rows.append({'N_Bays': n, 'Scenario': scenario, 'Kind': c[0],
                                  'Component': label(c), 'MTTF_y': MTTF_Y[c[0]],
                                  'Importance': i, 'Downtime_Share': U[c] * i / tot,
                                  'MTTR_Sens_min_yr_per_h':
                                      i / (MTTF_Y[c[0]] * HOURS_PER_YEAR) * MIN_PER_YEAR})
            print(scenario, n, len(flows), round(r['Timing_OK'], 3),
                  round(r['Down_min_yr_Single_8h'], 2), round(r['Down_min_yr_PRP_8h'], 2),
                  r['Exposure_Node'], round(r['Exposure'], 3))
    df = pd.DataFrame(rows)
    dc = pd.DataFrame(comp_rows)
    df.to_csv(os.path.join('results', 'maintainability.csv'), index=False, float_format='%.8g')
    dc.to_csv(os.path.join('results', 'maintainability_components.csv'), index=False,
              float_format='%.8g')

    # ---- table: components ranked by downtime share, N = 10, BBP link upgraded ----
    d = dc[(dc.N_Bays == 10) & (dc.Scenario == 'bbp_only')].copy()

    def group(row):
        name = row.Component
        if row.Kind in ('Switch', 'Trunk') or name.startswith('BBP'):
            return name
        if row.Kind == 'Link':
            return 'Access link of a bay device (each)'
        return ('Protection IED (each)' if name.startswith('PP') else
                'Merging unit (each)' if name.startswith('MU') else name)
    d['Group'] = d.apply(group, axis=1)
    g = d.groupby('Group').agg(Kind=('Kind', 'first'), MTTF_y=('MTTF_y', 'first'),
                               Importance=('Importance', 'max'),
                               Share=('Downtime_Share', 'max'),
                               Sens=('MTTR_Sens_min_yr_per_h', 'max'),
                               Count=('Component', 'count'))
    g = g.sort_values('Share', ascending=False)
    g = g[~g.index.isin(['SMC', 'SMC access link'])]
    r10 = df[(df.N_Bays == 10) & (df.Scenario == 'bbp_only')].iloc[0]
    lines = [r'\begin{table}[!t]', r'\centering',
             r'\caption{Graph-based maintainability ranking of the components for $N=10$ '
             r'(BBP link at 1~Gbps, single LAN, MTTR~$=8$~h). $I_c$: share of the time-critical '
             r'flows whose path uses the component; $S_c$: share of the expected flow downtime; '
             r'$\partial T/\partial\mathrm{MTTR}$: added downtime per hour of repair time. '
             r'For repeated bay components the largest value is shown.}',
             r'\label{tab:maintainability}', r'\resizebox{\columnwidth}{!}{',
             r'\begin{tabular}{lcccc}', r'\toprule',
             r'Component & MTTF & $I_c$ & $S_c$ & $\partial T/\partial\mathrm{MTTR}$ \\',
             r' & (y) & (\%) & (\%) & (min/yr per h) \\', r'\midrule']
    for name, x in g.iterrows():
        lines.append(f'{name} & {x.MTTF_y:.0f} & {100*x.Importance:.1f} & '
                     f'{100*x.Share:.1f} & {x.Sens:.4f} \\\\')
    lines += [r'\midrule',
              r'\multicolumn{5}{l}{Mean flow downtime: single LAN '
              f'{r10["Down_min_yr_Single_8h"]:.1f}~min/yr (MTTR 8~h), '
              f'{r10["Down_min_yr_Single_72h"]:.0f}~min/yr (72~h)' r'} \\',
              r'\multicolumn{5}{l}{\phantom{Mean flow downtime:} PRP '
              f'{r10["Down_min_yr_PRP_8h"]:.1f}~min/yr (8~h), '
              f'{r10["Down_min_yr_PRP_72h"]:.1f}~min/yr (72~h)' r'} \\',
              r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    with open(os.path.join(OUT, 'maintainability_table.tex'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ---- figure ----
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 8, 'legend.fontsize': 6.5, 'lines.markersize': 4,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={'wspace': 0.3})
    b = df[df.Scenario == 'bbp_only'].sort_values('N_Bays')
    for tag, color, mk, ls, lab in (('Single_8h', '#2a78d6', 'o', '-', 'Single LAN, MTTR 8 h'),
                                    ('Single_72h', '#2a78d6', 's', '--', 'Single LAN, MTTR 72 h'),
                                    ('PRP_8h', '#1baf7a', '^', '-', 'PRP, MTTR 8 h'),
                                    ('PRP_72h', '#1baf7a', 'D', '--', 'PRP, MTTR 72 h')):
        ax1.plot(b.N_Bays, b[f'Down_min_yr_{tag}'], color=color, marker=mk, ls=ls, label=lab)
    ax1.set_yscale('log')
    ax1.set_ylabel('Mean flow downtime (min/yr)')
    ax1.set_title('(a) Structural downtime of the time-critical flows')
    ax1.legend(frameon=False, ncol=2, loc='upper center')
    ax1.set_ylim(3, 1500)
    c = dc[dc.Scenario == 'bbp_only']
    for comp, color, mk, ls in (('SW1', '#eb6834', 'o', '-'), ('SW2', '#e87ba4', 's', '-.'),
                                ('Trunk SW1--SW2', '#eda100', 'v', ':'),
                                ('BBP access link', '#2a78d6', 'D', '--')):
        s = c[c.Component == comp].set_index('N_Bays').Importance.reindex(range(1, 11))
        ax2.plot(s.index, 100 * s, color=color, marker=mk, ls=ls,
                 label={'Trunk SW1--SW2': 'Trunk', 'BBP access link': 'BBP link'}.get(comp, comp))
    ax2.plot(b.N_Bays, 0 * b.N_Bays, color='#1baf7a', marker='^', ls='--',
             label='PRP: any')
    ax2.set_ylabel('Flows interrupted (%)')
    ax2.set_ylim(-5, 150)
    ax2.set_yticks(range(0, 101, 20))
    ax2.set_title('(b) Exposure during maintenance of one element')
    ax2.legend(frameon=False, loc='upper center', ncol=3, columnspacing=0.8,
               title='Element out of service (single LAN unless noted)', title_fontsize=6.5)
    for ax in (ax1, ax2):
        ax.set_xticks(range(1, 11))
        ax.set_xlabel('Number of bays $N$')
        ax.grid(True, ls='--', lw=0.35, alpha=0.5)
    fig.savefig(os.path.join(OUT, 'fig_maintainability.png'))
    fig.savefig(os.path.join(OUT, 'fig_maintainability.pdf'))
    print(df[['N_Bays', 'Scenario', 'Timing_OK', 'A_F_Single_8h', 'A_F_PRP_8h',
              'Down_min_yr_Single_8h', 'Down_min_yr_PRP_8h', 'Exposure', 'Exposure_Node']]
          .to_string())
