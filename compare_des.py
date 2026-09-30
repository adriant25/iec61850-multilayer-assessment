# -*- coding: utf-8 -*-
"""
compare_des.py -- Analytical end-to-end delays vs the DES reference table.

SV: steady-state snapshot vs DES mean over SV flows (the DES mean is dominated
by steady-state traffic).
GOOSE (50BF-cascade flows): the DES averages the frames received in the 7 s
run, of which a fraction f_b belongs to the burst (the rest are 1 s
heartbeats). The analytical window average is therefore
    f_b * D_burst + (1 - f_b) * D_steady,
with f_b = n_burst / (n_burst + (T_obs - T_burst) / T0) from the burst timeline.

Usage:  python compare_des.py   (needs des_reference.csv)
"""

import pandas as pd

from config import OBS_WINDOW_S
from model import (generate_devices, generate_demand_tensor, update_goose_demand,
                   build_topology, generate_goose_burst_timeline)
from analysis import compute_weights, compute_e2e_metrics

_, deltas = generate_goose_burst_timeline()
burst = [d for d in deltas if d < 1000.0]
F_BURST = len(burst) / (len(burst) + (OBS_WINDOW_S - sum(burst) / 1e3) / 1.0)


def analytic(n, scenario):
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    labels, A = build_topology(n, devs, D0, sw)
    out = {}
    for tag, dt in (('steady', 1000.0), ('burst', 4.0)):
        D = update_goose_demand(D0, n, dt)
        W, J, PL, u = compute_weights(labels, A, D, sw, devs, dt, scenario)
        out[tag] = compute_e2e_metrics(labels, W, PL, D, sw, scenario, D_window=D0,
                                       synchronized_burst=(dt < 1000.0))
    return out


if __name__ == '__main__':
    des = pd.read_csv('des_reference.csv')
    rows = []
    for _, r in des.iterrows():
        a = analytic(int(r.N_Bays), r.Scenario)
        sv = a['steady']['E2E_Avg_SV_us']
        go = F_BURST * a['burst']['E2E_Avg_GOOSE_us'] + (1 - F_BURST) * a['steady']['E2E_Avg_GOOSE_us']
        rows.append({'N': int(r.N_Bays), 'Scenario': r.Scenario,
                     'SV_an': sv, 'SV_des': r.SV_mean_us, 'SV_diff_%': 100 * (sv / r.SV_mean_us - 1),
                     'GO_an_steady': a['steady']['E2E_Avg_GOOSE_us'],
                     'GO_an_burst': a['burst']['E2E_Avg_GOOSE_us'],
                     'GO_an_window': go, 'GO_des': r.GOOSE_mean_us,
                     'GO_diff_%': 100 * (go / r.GOOSE_mean_us - 1),
                     'SV_PLR_an': a['steady']['E2E_PLR_Avg_SV'], 'SV_PLR_des': r.SV_PLR_pct})
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 220)
    print(f'f_burst = {F_BURST:.3f}')
    print(df.round(2).to_string(index=False))
    df.to_csv('analytic_vs_des.csv', index=False, float_format='%.4f')
