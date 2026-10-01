# -*- coding: utf-8 -*-
"""
robustness.py -- Additional sensitivity and scaling studies (review round).

1. GOOSE minimum retransmission interval T1 (1, 2, 4, 8 ms): burst-peak load
   of the BBP port, critical size if the peak were sustained, and delay of the
   cascade GOOSE flows (one-bay and all-bay events).
2. Link capacity of the BBP port and of the trunk: maximum number of bays
   N*(C) from the rate-weighted flow betweenness (congestion threshold map).
3. Computational scaling of the analytical model: graph size and run time
   for N = 10 ... 50 bays (two-switch topology kept; the purpose is the cost
   of the evaluation, not the realism of very large designs).

Outputs: results/robust_t1.csv, results/robust_capacity.csv,
         results/robust_scaling.csv
"""

import os
import time
import numpy as np
import pandas as pd

from model import (generate_devices, generate_demand_tensor, update_goose_demand,
                   build_topology, event_burst_vlans)
from analysis import (compute_weights, compute_e2e_metrics, analyze_network,
                      _port_cumulative_rho)

OUT = 'results'


def rho_port(D, sw, scen, port='L5_OutPort_BBP'):
    return _port_cumulative_rho(D, sw, scen)[(port, 0)][0]


def t1_sweep():
    rows = []
    for t1 in (1.0, 2.0, 4.0, 8.0):
        # critical size of the BBP port with the sustained burst-peak load
        slopes = {}
        for scope in ('single', 'all'):
            xs, ys = [], []
            for n in range(6, 13):
                devs, sw = generate_devices(n)
                D0 = generate_demand_tensor(n, devs, 1000.0)
                D = update_goose_demand(D0, n, t1, burst_bays=[n] if scope == 'single' else None)
                xs.append(n)
                ys.append(rho_port(D, sw, 'base'))
            b, a = np.polyfit(xs, ys, 1)
            slopes[scope] = (1 - a) / b
        for n in (8, 10):
            devs, sw = generate_devices(n)
            D0 = generate_demand_tensor(n, devs, 1000.0)
            labels, A = build_topology(n, devs, D0, sw)
            r = {'T1_ms': t1, 'N_Bays': n, 'Nstar_single_peak': slopes['single'],
                 'Nstar_all_peak': slopes['all'], 'Rho_BBP_steady': rho_port(D0, sw, 'base')}
            for scope, bays in (('single', [n]), ('all', None)):
                D = update_goose_demand(D0, n, t1, burst_bays=bays)
                W, J, PL, u = compute_weights(labels, A, D, sw, devs, t1, 'base')
                m = compute_e2e_metrics(labels, W, PL, D, sw, 'base', D_window=D0,
                                        synchronized_burst=True,
                                        burst_vlans=set(event_burst_vlans(n, bays)))
                r.update({f'Rho_BBP_peak_{scope}': rho_port(D, sw, 'base'),
                          f'GOOSE_event_mean_{scope}_us': m['E2E_Avg_GOOSE_Event_us'],
                          f'GOOSE_event_max_{scope}_us': m['E2E_Max_GOOSE_Event_us'],
                          f'SV_mean_peak_{scope}_us': m['E2E_Avg_SV_us']})
            rows.append(r)
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'robust_t1.csv'), index=False, float_format='%.4f')


def capacity_sweep():
    """N*(C) from the per-bay rate-weighted flow betweenness of the two growing ports."""
    loads = {}
    for p in ('L5_OutPort_BBP', 'L4_Trunk_SW2_to_SW1'):
        xs, ys = [], []
        for n in range(6, 21):
            devs, sw = generate_devices(n)
            D0 = generate_demand_tensor(n, devs, 1000.0)
            rho, cap = _port_cumulative_rho(D0, sw, 'base')[(p, 0)]
            xs.append(n)
            ys.append(rho * cap / 1e6)          # B_F in Mbps
        loads[p] = np.polyfit(xs, ys, 1)       # (slope, intercept) Mbps per bay
    rows = []
    for c in (100, 200, 500, 1000, 2500, 10000):
        nb = (c - loads['L5_OutPort_BBP'][1]) / loads['L5_OutPort_BBP'][0]
        nt = (c - loads['L4_Trunk_SW2_to_SW1'][1]) / loads['L4_Trunk_SW2_to_SW1'][0]
        rows.append({'C_Mbps': c, 'Nstar_BBP_port': nb, 'Nstar_trunk': nt,
                     'BF_per_bay_BBP_Mbps': loads['L5_OutPort_BBP'][0],
                     'BF_per_bay_trunk_Mbps': loads['L4_Trunk_SW2_to_SW1'][0]})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'robust_capacity.csv'), index=False, float_format='%.4f')


def scaling():
    rows = []
    for n in (10, 20, 30, 40, 50):
        t0 = time.perf_counter()
        devs, sw = generate_devices(n)
        D0 = generate_demand_tensor(n, devs, 1000.0)
        labels, A = build_topology(n, devs, D0, sw)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        for dt in (1000.0, 4.0):
            D = update_goose_demand(D0, n, dt)
            W, J, PL, u = compute_weights(labels, A, D, sw, devs, dt, 'base')
            compute_e2e_metrics(labels, W, PL, D, sw, 'base', D_window=D0,
                                synchronized_burst=(dt < 1000.0))
        t_flow = (time.perf_counter() - t0) / 2
        t0 = time.perf_counter()
        W, J, PL, u = compute_weights(labels, A, D0, sw, devs, 1000.0, 'base')
        analyze_network(n, labels, W, J, PL, u)
        t_struct = time.perf_counter() - t0
        n_flows = sum(int((df > 0).values.sum()) for df in D0.values())
        rows.append({'N_Bays': n, 'Nodes': len(labels), 'Edges': int(A.sum()), 'Flows': n_flows,
                     'Build_s': t_build, 'Delay_and_flow_indicators_s_per_snapshot': t_flow,
                     'Structural_indicators_s': t_struct})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'robust_scaling.csv'), index=False, float_format='%.4f')


if __name__ == '__main__':
    import sys
    {'t1': t1_sweep, 'cap': capacity_sweep, 'scale': scaling}[sys.argv[1]]()
