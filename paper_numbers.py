# -*- coding: utf-8 -*-
"""
paper_numbers.py -- Every case-dependent number quoted in the text of the paper,
computed from the model and the DES outputs (on-wire frame sizes).

Output: results/paper_numbers.txt
"""

import os
import numpy as np
import pandas as pd

from config import TRAFFIC_SPECS, T_MAX, PORT_BUFFER_BYTES, OBS_WINDOW_S
from model import generate_devices, generate_demand_tensor, update_goose_demand, build_topology
from analysis import (compute_weights, compute_e2e_metrics, _port_cumulative_rho, fluid_overload,
                      ndd1_mean_wait, port_capacity, SV_PERIOD_S)
from extract_des_reference import DES_RUNS, DES_REPLICATES

out = []
P = lambda *a: out.append(' '.join(str(x) for x in a))
S_SV = TRAFFIC_SPECS['SV']['size']
s = S_SV * 8 / 100e6
P('SV on wire', S_SV, 'B; service', round(s * 1e6, 3), 'us; per-bay BBP load',
  round(2 * S_SV * 8 * 4800 / 1e6, 4), 'Mbps; N* =', round(100e6 / (2 * S_SV * 8 * 4800), 4))

# ---- port utilizations
for n in range(1, 11):
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    r = {p: v[0] for (p, k), v in _port_cumulative_rho(D0, sw, 'base').items() if k == 0}
    other = max(v for p, v in r.items() if p != 'L5_OutPort_BBP' and not p.startswith('L4_Trunk'))
    trunk = max((v for p, v in r.items() if p.startswith('L4_Trunk')), default=0)
    hi = _port_cumulative_rho(D0, sw, 'base')[('L5_OutPort_BBP', 5)][0]
    Dp = update_goose_demand(D0, n, 4.0)
    hi_peak = _port_cumulative_rho(Dp, sw, 'base')[('L5_OutPort_BBP', 5)][0]
    P(f'N={n}: rho_BBP={r["L5_OutPort_BBP"]:.4f} other_max={other:.3f} trunk_max={trunk:.3f}'
      f' load_above_SV steady={hi:.4f} allbay_peak={hi_peak:.4f}')

# ---- BBP queue: N*D/D/1 vs M/D/1 at the last stable size
for n in (7,):
    k = 2 * n
    w_nd = ndd1_mean_wait(k - 1, SV_PERIOD_S, s)
    rho = k * s / SV_PERIOD_S
    w_md1 = rho * s / (2 * (1 - rho))
    an = pd.read_csv('analytic_vs_des.csv').set_index(['N', 'Scenario'])
    sv_an = an.loc[(n, 'base'), 'SV_an']
    n_flows, n_bbp = 6 * n, 2 * n
    sv_md1 = sv_an + (w_md1 - w_nd) * 1e6 * n_bbp / n_flows
    P(f'N={n}: rho={rho:.4f} NDD1 wait={w_nd*1e6:.2f} us, M/D/1 wait={w_md1*1e6:.2f} us,'
      f' mean SV model={sv_an:.2f}, with M/D/1={sv_md1:.2f} (+{100*(sv_md1/sv_an-1):.1f}% vs model)')
    des = pd.read_csv('des_reference.csv').set_index(['N_Bays', 'Scenario'])
    P(f'   vs DES mean {des.loc[(n, "base"), "SV_mean_us"]:.2f}: M/D/1 +{100*(sv_md1/des.loc[(n, "base"), "SV_mean_us"]-1):.1f}%')

# ---- fluid vs DES for the BBP-bound SV flows beyond the boundary
for n in (8, 9, 10):
    rho = 2 * n * s / SV_PERIOD_S + 0.0
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    rho = _port_cumulative_rho(D0, sw, 'base')[('L5_OutPort_BBP', 0)][0]
    mean, mx, loss = fluid_overload(rho, 100e6)
    t_fill = 8 * PORT_BUFFER_BYTES / (100e6 * (rho - 1))
    k = pd.read_excel(DES_RUNS[(n, 'base')], sheet_name='KPIs_per_Flow')
    b = k[(k.Traffic_Type == 'SV') & (k.Destination == 'BBP')]
    P(f'N={n}: rho={rho:.4f} t_fill={t_fill:.2f} s fluid wait mean={mean*1e3:.1f} ms max={mx*1e3:.1f} ms '
      f'loss={100*loss:.2f}% | DES BBP-bound mean={b.Average_Delay_us.mean()/1e3:.1f} ms '
      f'max={b.Max_Delay_us.max()/1e3:.1f} ms loss={100*b.PLR_Total.mean():.2f}% (n_flows={len(b)})')

# ---- analytical SV means of the scenarios
g = pd.read_excel(os.path.join('results', 'Global_Report.xlsx'), sheet_name='Raw_Data')
cols = ['Fiedler_Lambda2', 'Global_Efficiency', 'Load_Gini', 'Flow_Efficiency', 'QDC_Top_Node',
        'QDC_Top_Share', 'E2E_Avg_SV_us', 'E2E_Avg_GOOSE_Event_us', 'E2E_Max_GOOSE_Event_us']
for scen in ('base', 'bbp_only', 'upgraded'):
    x = g[(g.Scenario == scen) & (g.Snapshot_Time_ms == 0)].set_index('N_Bays')[cols]
    P('== steady', scen)
    P(x.round(4).to_string())
pk = g[(g.Scenario == 'base') & (g.Delta_T_ms == 4.0)].groupby('N_Bays')[['E2E_Avg_GOOSE_Event_us', 'E2E_Max_GOOSE_Event_us', 'E2E_Avg_SV_us', 'E2E_Max_SV_us', 'Load_Gini']].max()
P('== base burst peak (dt=4 ms)'); P(pk.round(3).to_string())
v = pd.read_excel(os.path.join('results', 'Metrics_By_Scenario.xlsx'), sheet_name='Vulnerability_Base')
for n in (5, 6, 7, 8, 10):
    x = v[v.N_Bays == n].drop_duplicates('Node_Label').sort_values('Vulnerability_Index', ascending=False).head(3)
    P(f'N={n} top V:', [(a, round(b, 3)) for a, b in zip(x.Node_Label, x.Vulnerability_Index)])
    x = v[v.N_Bays == n].drop_duplicates('Node_Label').sort_values('Betweenness', ascending=False)
    P(f'N={n} top BC:', [(a, round(b, 3)) for a, b in zip(x.Node_Label.head(3), x.Betweenness.head(3))],
      'BBP egress in list:', 'L5_OutPort_BBP' in set(x.Node_Label))

# ---- DES maxima of the compliant configurations
des = pd.read_csv('des_reference.csv')
ok = des[des.SV_mean_us < 3000]
P('DES largest frame delay among compliant configs: SV', ok.SV_max_us.max().round(1),
  'GOOSE', ok.GOOSE_max_us.max().round(1))
P(des[['N_Bays', 'Scenario', 'SV_mean_us', 'SV_max_us', 'GOOSE_mean_us', 'GOOSE_max_us', 'SV_PLR_pct']].round(2).to_string())

open(os.path.join('results', 'paper_numbers.txt'), 'w', encoding='utf-8').write('\n'.join(out) + '\n')
print('\n'.join(out))
