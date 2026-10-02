# -*- coding: utf-8 -*-
"""
single_bay_des.py -- Single-bay 50BF event: analytical model versus DES.

DES runs (N = 7, base, 7 s, event at 3 s only in bay 7, seeds 42, 1, 2; on-wire
frame sizes) are stored in CASO7BAHIAS/SingleBay7_wire/seed_<k>/KPIs_PB.xlsx. The compared GOOSE
flows are the trip and breaker-status flows of the failed bay (publishers
PP1B7L, PP2B7L, MU1B7, MU2B7); the analytical value is the window average
f_b * D_burst + (1 - f_b) * D_steady used for the all-bay comparison.

Output: results/single_bay_des.csv
"""

import glob
import os
import numpy as np
import pandas as pd
from scipy import stats

from extract_des_reference import _case
from compare_des import F_BURST
from model import generate_devices, generate_demand_tensor, update_goose_demand, build_topology, event_burst_vlans
from analysis import compute_weights, compute_e2e_metrics

N, BAY = 7, 7
PUB = {f'PP1B{BAY}L', f'PP2B{BAY}L', f'MU1B{BAY}', f'MU2B{BAY}'}


def analytic():
    devs, sw = generate_devices(N)
    D0 = generate_demand_tensor(N, devs, 1000.0)
    labels, A = build_topology(N, devs, D0, sw)
    vl = set(event_burst_vlans(N, [BAY]))
    out = {}
    for tag, dt in (('steady', 1000.0), ('burst', 4.0)):
        D = update_goose_demand(D0, N, dt, burst_bays=[BAY])
        W, J, PL, u = compute_weights(labels, A, D, sw, devs, dt, 'base')
        m = compute_e2e_metrics(labels, W, PL, D, sw, 'base', D_window=D0, synchronized_burst=(dt < 1000),
                                burst_vlans=vl, return_flows=True)
        f = [d for (c, v, s, t, _, d) in m['flow_records'] if c == 'GOOSE' and s in PUB and v in vl]
        out[tag] = np.mean(f) * 1e6
        out[f'SV_{tag}'] = m['E2E_Avg_SV_us']
    out['window'] = F_BURST * out['burst'] + (1 - F_BURST) * out['steady']
    return out


def des():
    rows = []
    for p in sorted(glob.glob(os.path.join(_case(N), 'SingleBay7_wire', 'seed_*', 'KPIs_PB.xlsx'))):
        k = pd.read_excel(p, sheet_name='KPIs_per_Flow')
        g = k[(k.Traffic_Type == 'GOOSE') & k.Source.astype(str).isin(PUB)]
        g = g[g.Destination.astype(str).str.contains(f'B{BAY}') | (g.Destination == 'BBP')]
        sv = k[k.Traffic_Type == 'SV']
        rows.append({'seed': os.path.basename(os.path.dirname(p)), 'GOOSE_flows': len(g),
                     'GOOSE_mean_us': g.Average_Delay_us.mean(), 'GOOSE_max_us': g.Max_Delay_us.max(),
                     'SV_mean_us': sv.Average_Delay_us.mean()})
    return pd.DataFrame(rows)


if __name__ == '__main__':
    a = analytic()
    d = des()
    print(a)
    print(d.round(2).to_string(index=False))
    ci = stats.t.ppf(0.975, len(d) - 1) * d.GOOSE_mean_us.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
    res = {'Analytic_GOOSE_steady_us': a['steady'], 'Analytic_GOOSE_burst_us': a['burst'],
           'Analytic_GOOSE_window_us': a['window'], 'DES_GOOSE_mean_us': d.GOOSE_mean_us.mean(),
           'DES_GOOSE_ci95_us': ci, 'DES_GOOSE_max_us': d.GOOSE_max_us.max(),
           'Analytic_SV_steady_us': a['SV_steady'], 'DES_SV_mean_us': d.SV_mean_us.mean(),
           'Replications': len(d)}
    res['GOOSE_diff_pct'] = 100 * (res['Analytic_GOOSE_window_us'] / res['DES_GOOSE_mean_us'] - 1)
    print(res)
    pd.DataFrame([res]).to_csv(os.path.join('results', 'single_bay_des.csv'), index=False, float_format='%.3f')
