# -*- coding: utf-8 -*-
"""
event_scope.py -- Effect of the scope of the 50BF event on message timing.

Two event scopes are compared for N = 1..10 (base scenario):
  single : breaker failure in one bay (bay N): its line protections and merging
           units, plus the BBP bus-trip broadcast, enter burst mode;
  all    : the event is injected simultaneously in every bay (worst case, the
           scenario simulated in the DES).
For each scope the steady-state and burst-peak (T1 = 4 ms) snapshots give the
mean/max end-to-end delay of the GOOSE flows of the event, the SV delay and the
instantaneous BBP-port utilization.

Outputs: results/event_scope.csv, results/paper/event_scope_table.tex
"""

import os
import pandas as pd

from model import (generate_devices, generate_demand_tensor, update_goose_demand,
                   build_topology, event_burst_vlans)
from analysis import compute_weights, compute_e2e_metrics, _port_cumulative_rho

SCENARIO = 'base'
OUT = os.path.join('results', 'paper')
os.makedirs(OUT, exist_ok=True)


def evaluate(n: int, scope: str) -> dict:
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    labels, A = build_topology(n, devs, D0, sw)
    bays = None if scope == 'all' else [n]
    vlans = set(event_burst_vlans(n, bays))
    row = {'N_Bays': n, 'Scope': scope}
    for tag, dt in (('steady', 1000.0), ('peak', 4.0)):
        D = update_goose_demand(D0, n, dt, burst_bays=bays)
        W, J, PL, u = compute_weights(labels, A, D, sw, devs, dt, SCENARIO)
        m = compute_e2e_metrics(labels, W, PL, D, sw, SCENARIO, D_window=D0,
                                synchronized_burst=(dt < 1000.0), burst_vlans=vlans)
        rho = _port_cumulative_rho(D, sw, SCENARIO)[('L5_OutPort_BBP', 0)][0]
        row.update({f'GOOSE_Event_Avg_{tag}': m['E2E_Avg_GOOSE_Event_us'],
                    f'GOOSE_Event_Max_{tag}': m['E2E_Max_GOOSE_Event_us'],
                    f'SV_Avg_{tag}': m['E2E_Avg_SV_us'],
                    f'SV_Max_{tag}': m['E2E_Max_SV_us'],
                    f'Rho_BBP_{tag}': rho})
    return row


if __name__ == '__main__':
    rows = [evaluate(n, scope) for n in range(1, 11) for scope in ('single', 'all')]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join('results', 'event_scope.csv'), index=False, float_format='%.4f')
    pd.set_option('display.width', 220)
    print(df.round(2).to_string(index=False))

    sel = [5, 8, 10]
    lines = [r'\begin{table}[!t]', r'\centering',
             r'\caption{Effect of the event scope on the end-to-end delay of the 50BF-cascade GOOSE flows (base scenario): breaker failure in one bay versus simultaneous event in all bays.}',
             r'\label{tab:event_scope}', r'\resizebox{\columnwidth}{!}{',
             r'\begin{tabular}{lccccc}', r'\toprule',
             r'$N$ & Scope & GOOSE mean & GOOSE mean & GOOSE max & $\rho_{BBP}$ \\',
             r' & & steady ($\mu$s) & peak ($\mu$s) & peak ($\mu$s) & peak \\', r'\midrule']
    for n in sel:
        for scope, label in (('single', 'One bay'), ('all', 'All bays')):
            r = df[(df.N_Bays == n) & (df.Scope == scope)].iloc[0]
            lines.append(f'{n} & {label} & {r.GOOSE_Event_Avg_steady:.1f} & {r.GOOSE_Event_Avg_peak:.1f} & '
                         f'{r.GOOSE_Event_Max_peak:.1f} & {r.Rho_BBP_peak:.3f} \\\\')
        if n != sel[-1]:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    with open(os.path.join(OUT, 'event_scope_table.tex'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
