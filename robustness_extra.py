# -*- coding: utf-8 -*-
"""
robustness_extra.py -- Two further sensitivity checks (review round).

1. Port buffer size B (64 kB, 256 kB, 1 MB, 4 MB): time to fill the BBP port
   buffer, delay plateau 8B/C and mean SV loss of the BBP-bound flows over the
   7 s window (fluid model), for the overloaded sizes N = 8 and N = 10. The
   stability boundary N* does not depend on B.
2. Switch fabric capacity: largest fabric utilization of the model (sum of the
   loads entering a switch over C_fabric) in the base and burst snapshots, and
   the head-of-line wait it implies, for the 68 Gbps of the case study and for
   a fabric of only 1 Gbps.

Outputs: results/robust_buffer.csv, results/robust_fabric.csv
"""

import os
import numpy as np
import pandas as pd

import config
from config import T_PROC_SWITCH, OBS_WINDOW_S
from model import generate_devices, generate_demand_tensor, update_goose_demand, build_topology
from analysis import compute_weights, _port_cumulative_rho, fluid_overload, CAPACITY_DEFAULT

OUT = 'results'


def buffer_sweep():
    rows = []
    for n in (8, 10):
        devs, sw = generate_devices(n)
        D0 = generate_demand_tensor(n, devs, 1000.0)
        rho = _port_cumulative_rho(D0, sw, 'base')[('L5_OutPort_BBP', 0)][0]
        for kb in (64, 256, 1000, 4000):
            b = kb * 1000.0
            t_fill = 8 * b / (CAPACITY_DEFAULT * (rho - 1))
            plateau = 8 * b / CAPACITY_DEFAULT
            loss = (rho - 1) / rho * max(OBS_WINDOW_S - t_fill, 0) / OBS_WINDOW_S
            rows.append({'N_Bays': n, 'Rho_BBP': rho, 'Buffer_kB': kb, 't_fill_s': t_fill,
                         'Plateau_ms': plateau * 1e3, 'Loss_BBP_flows_pct': 100 * loss,
                         'Plateau_above_3ms': plateau > 3e-3})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'robust_buffer.csv'), index=False, float_format='%.4f')
    print(df.round(3).to_string(index=False))


def fabric_sweep():
    rows = []
    for n in (5, 10):
        devs, sw = generate_devices(n)
        D0 = generate_demand_tensor(n, devs, 1000.0)
        labels, A = build_topology(n, devs, D0, sw)
        for tag, dt in (('steady', 1000.0), ('burst', 4.0)):
            D = update_goose_demand(D0, n, dt)
            for cap in (68000.0, 1000.0):
                config.FABRIC_CAPACITY = cap
                import analysis
                analysis.FABRIC_CAPACITY = cap
                W, J, PL, u = compute_weights(labels, A, D, sw, devs, dt, 'base')
                u_max = max(max(v.values()) for v in u.values() if v)
                hol = T_PROC_SWITCH * u_max / (2 * (1 - u_max)) if u_max < 1 else np.inf
                rows.append({'N_Bays': n, 'Snapshot': tag, 'C_fabric_Mbps': cap,
                             'U_fabric_max': u_max, 'HOL_wait_us': hol * 1e6})
    config.FABRIC_CAPACITY = 68000.0
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'robust_fabric.csv'), index=False, float_format='%.6f')
    print(df.round(4).to_string(index=False))


if __name__ == '__main__':
    buffer_sweep()
    fabric_sweep()
