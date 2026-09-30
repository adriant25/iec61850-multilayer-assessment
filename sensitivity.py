# -*- coding: utf-8 -*-
"""
sensitivity.py -- Design sensitivity study with flow-constrained graph indicators.

Variants of the busbar-protection subscription (config.BBP_SV_STREAMS_PER_BAY):
  2 streams per bay (case study) and 4 streams per bay (e.g. main + check zone),
each under the three link scenarios (base, bbp_only, upgraded), N = 1..10.

For every configuration the steady-state and event (first burst) snapshots are
evaluated. Recorded per configuration:
  - IEC compliance of all SV and GOOSE flows (both snapshots)
  - the most loaded port and its utilization (cumulative, steady state)
  - flow-constrained efficiency E_F and the node with the largest queuing-delay
    centrality (QDC) -- the bottleneck located by the graph
  - shortest-path global efficiency E (for comparison)

Outputs: results/sensitivity.csv, results/paper/sensitivity_table.tex,
         results/paper/fig_flow_indicators.png
"""

import os
import sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import config
from config import T_MAX
from model import (generate_devices, generate_demand_tensor, update_goose_demand,
                   build_topology)
from analysis import (compute_weights, compute_e2e_metrics, analyze_network,
                      _port_cumulative_rho, SCENARIOS)

OUT = os.path.join('results', 'paper')
os.makedirs(OUT, exist_ok=True)

SCEN_LABEL = {'base': 'Base (100 Mbps)', 'bbp_only': 'BBP link 1 Gbps',
              'upgraded': 'Trunk + BBP link 1 Gbps'}
ORDER = ('base', 'bbp_only', 'upgraded')


def short(node: str) -> str:
    return (node.replace('L5_OutPort_', 'Egress ').replace('L4_Trunk_', 'Trunk ')
                .replace('_to_', r'$\to$').replace('L4_Priority_', 'Queue ')
                .replace('L1_Equipment_', 'Publisher '))


def evaluate(n: int, scenario: str) -> dict:
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    labels, A = build_topology(n, devs, D0, sw)
    res = {}
    for tag, dt in (('steady', 1000.0), ('event', 4.0)):
        D = update_goose_demand(D0, n, dt)
        W, J, PL, u = compute_weights(labels, A, D, sw, devs, dt, scenario)
        e2e = compute_e2e_metrics(labels, W, PL, D, sw, scenario, D_window=D0,
                                  synchronized_burst=(dt < 1000.0))
        res[tag] = e2e
        if tag == 'steady':
            res['E_shortest'] = analyze_network(n, labels, W, J, PL, u)['Global_Efficiency']
    rho = {p: v[0] for (p, k), v in _port_cumulative_rho(D0, sw, scenario).items() if k == 0}
    top_port = max(rho, key=rho.get)
    worst = max(max(res[t]['E2E_Max_SV_us'], res[t]['E2E_Max_GOOSE_us']) for t in ('steady', 'event'))
    return {
        'N_Bays': n, 'Scenario': scenario,
        'Compliant': worst < T_MAX * 1e6,
        'Worst_Delay_us': worst,
        'Top_Port': top_port, 'Top_Port_Rho': rho[top_port],
        'E_Flow': res['steady']['Flow_Efficiency'],
        'E_Shortest': res['E_shortest'],
        'QDC_Top_Node': res['steady']['QDC_Top_Node'],
        'QDC_Top_Share': res['steady']['QDC_Top_Share'],
        'SV_Mean_us': res['steady']['E2E_Avg_SV_us'],
    }


if __name__ == '__main__':
    csv_path = os.path.join('results', 'sensitivity.csv')
    rows = []
    for streams in ((2, 4) if '--plot' not in sys.argv else ()):
        config.BBP_SV_STREAMS_PER_BAY = streams
        for scenario in SCENARIOS:
            for n in range(1, 11):
                r = evaluate(n, scenario)
                r['BBP_Streams_per_Bay'] = streams
                rows.append(r)
                print(streams, scenario, n, r['Compliant'], r['Top_Port'],
                      round(r['Top_Port_Rho'], 3), r['QDC_Top_Node'], round(r['QDC_Top_Share'], 3),
                      round(r['E_Flow'], 2))
    config.BBP_SV_STREAMS_PER_BAY = 2
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, float_format='%.5f')
    else:
        df = pd.read_csv(csv_path)

    # ---- summary table: first non-compliant N and the bottleneck located ----
    lines = [r'\begin{table}[!t]', r'\centering',
             r'\caption{Design sensitivity: first non-compliant size and bottleneck located by the queuing-delay centrality (QDC) for two BBP subscription patterns and three link scenarios ($N\le10$).}',
             r'\label{tab:sensitivity}', r'\resizebox{\columnwidth}{!}{',
             r'\begin{tabular}{llccl}', r'\toprule',
             r'SV streams & Scenario & First & Port $\rho$ & Bottleneck (max QDC) \\',
             r'to BBP / bay & & FAIL $N$ & at FAIL & at first FAIL \\', r'\midrule']
    for streams in (2, 4):
        for scenario in ORDER:
            d = df[(df.BBP_Streams_per_Bay == streams) & (df.Scenario == scenario)].sort_values('N_Bays')
            fail = d[~d.Compliant]
            if fail.empty:
                last = d.iloc[-1]
                lines.append(f'{streams} & {SCEN_LABEL[scenario]} & -- & '
                             f'{last.Top_Port_Rho:.2f} ($N=10$) & -- \\\\')
            else:
                f = fail.iloc[0]
                lines.append(f'{streams} & {SCEN_LABEL[scenario]} & {int(f.N_Bays)} & '
                             f'{f.Top_Port_Rho:.2f} & {short(f.QDC_Top_Node)} \\\\')
        if streams == 2:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    with open(os.path.join(OUT, 'sensitivity_table.tex'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ---- figure: flow-constrained indicators ----
    plt.rcParams.update({'font.family': 'serif',
                         'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
                         'font.size': 8, 'legend.fontsize': 6.5, 'lines.markersize': 4,
                         'savefig.dpi': 300, 'savefig.bbox': 'tight'})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.5), gridspec_kw={'wspace': 0.28})
    cases = [(2, 'base', '#0072B2', 'o', '-'), (2, 'bbp_only', '#E69F00', '^', '-.'),
             (4, 'base', '#CC79A7', 's', ':'), (4, 'bbp_only', '#009E73', 'D', '--')]
    for streams, scenario, color, mk, ls in cases:
        d = df[(df.BBP_Streams_per_Bay == streams) & (df.Scenario == scenario)].sort_values('N_Bays')
        # QDC share of the port that becomes the bottleneck of this configuration
        port = d[~d.Compliant].QDC_Top_Node.iloc[0] if (~d.Compliant).any() else d.Top_Port.iloc[-1]
        share = []
        for n in d.N_Bays:
            share.append(d[d.N_Bays == n].QDC_Top_Share.iloc[0]
                         if d[d.N_Bays == n].QDC_Top_Node.iloc[0] == port else float('nan'))
        ax1.plot(d.N_Bays, share, color=color, marker=mk, ls=ls,
                 label=f'{streams} str./bay, {SCEN_LABEL[scenario]}: {short(port)}')
        ax2.plot(d.N_Bays, d.E_Flow / d.E_Flow.iloc[0], color=color, marker=mk, ls=ls)
    d = df[(df.BBP_Streams_per_Bay == 2) & (df.Scenario == 'base')].sort_values('N_Bays')
    ax2.plot(d.N_Bays, d.E_Shortest / d.E_Shortest.iloc[0], color='0.4', ls='--', lw=0.9,
             label=r'Shortest-path $E$, 2 str./bay, base')
    ax1.set_ylabel('QDC share of the bottleneck node')
    ax1.set_title('(a) Bottleneck located by QDC')
    ax2.set_ylabel(r'Normalized efficiency ($N=1$ = 1)')
    ax2.set_title(r'(b) Flow-constrained efficiency $E_F$')
    for ax in (ax1, ax2):
        ax.set_xticks(range(1, 11))
        ax.set_xlabel('Number of bays $N$')
        ax.grid(True, ls='--', lw=0.35, alpha=0.5)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='upper center', ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(OUT, 'fig_flow_indicators.png'))
    print('figure written')
