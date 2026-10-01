# -*- coding: utf-8 -*-
"""
nc_bound.py -- Network-calculus worst-case delay bound of the SV flows towards
the busbar protection, as a baseline for the analytical (mean) model and the DES.

Per hop, every frame stream f (one VLAN and publisher; multicast counted once
per resource) has a token-bucket arrival curve b_f + r_f t, with b_f one frame
at the publisher and b_f + r_f * D_up after an upstream delay bound D_up.

* Publisher link (FIFO, rate C):           D = sum_f b_f / C
* Switch fabric (single server, one frame per T_proc, non-preemptive strict
  priority, counted in frames):           D = (B_H + 1 + B_S) T / (1 - R_H T)
* Egress / trunk port (rate C, non-preemptive strict priority):
      leftover rate-latency curve R = C - R_H, T = (B_H + L_low) / R
      D = T + B_S / R
  with B_H, R_H the burst and rate of the higher priorities, B_S the burst of
  the SV priority (FIFO within the class) and L_low the largest lower-priority
  frame (non-preemption).

The end-to-end bound is the sum of the hop bounds (pay-bursts-only-once is not
applied, so the bound is conservative). Two snapshots: steady state and the
burst peak of an all-bay 50BF event (GOOSE at 1/T1 with one frame of burst).

Output: results/nc_bound.csv
"""

import os
import numpy as np
import pandas as pd

from config import T_PROC_SWITCH, get_frame_specs, get_vlan_priority, get_vlan_id, SV_VLAN_IDS
from model import generate_devices, generate_demand_tensor, update_goose_demand
from analysis import trunk_capacity_for, port_capacity, CAPACITY_DEFAULT

OUT = 'results'


def streams(D, sw):
    """One entry per (VLAN, publisher): rate [frames/s], size [bits], prio, dsts."""
    out = []
    for vlan, df in D.items():
        prio = get_vlan_priority(vlan)
        for src in df.index[df.any(axis=1)]:
            row = df.loc[src]
            dsts = list(row.index[row > 0])
            bits = get_frame_specs(vlan, src=src)['size'] * 8
            bw = float(row[row > 0].iloc[0]) * 1e6          # bit/s (same for every destination)
            out.append({'vlan': vlan, 'src': src, 'prio': prio, 'bits': bits,
                        'rate': bw / bits, 'dsts': dsts, 'sv': get_vlan_id(vlan) in SV_VLAN_IDS})
    return out


def port_bound(members, me_prio, cap, d_up):
    """Strict-priority, non-preemptive port of capacity cap [bit/s]."""
    hi = [m for m in members if m['prio'] > me_prio]
    same = [m for m in members if m['prio'] == me_prio]
    lo = [m for m in members if m['prio'] < me_prio]
    r_h = sum(m['rate'] * m['bits'] for m in hi)
    b_h = sum(m['bits'] + m['rate'] * m['bits'] * d_up for m in hi)
    b_s = sum(m['bits'] + m['rate'] * m['bits'] * d_up for m in same)
    l_lo = max((m['bits'] for m in lo), default=0)
    r = cap - r_h
    return (b_h + l_lo) / r + b_s / r


def fabric_bound(members, me_prio, d_up):
    """Single fabric server, one frame per T_PROC, non-preemptive strict priority."""
    t = T_PROC_SWITCH
    hi = [m for m in members if m['prio'] > me_prio]
    same = [m for m in members if m['prio'] == me_prio]
    r_h = sum(m['rate'] for m in hi)
    b_h = sum(1 + m['rate'] * d_up for m in hi)
    b_s = sum(1 + m['rate'] * d_up for m in same)
    return (b_h + 1 + b_s) * t / (1 - r_h * t)


def sv_to_bbp_bound(n, scenario, dt):
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    D = update_goose_demand(D0, n, dt)
    st = streams(D, sw)
    target = next(s for s in st if s['sv'] and s['src'] == f'MU1B{n}' and 'BBP' in s['dsts'])
    src, prio = target['src'], target['prio']
    sa, sb = sw[src], sw['BBP']
    hops = []
    d = 0.0
    # publisher link: all frames of the device, FIFO
    pub = [s for s in st if s['src'] == src]
    h = sum(s['bits'] for s in pub) / CAPACITY_DEFAULT
    hops.append(('publisher link', h)); d += h

    def entering(swn):
        return [s for s in st if sw[s['src']] == swn or any(sw[x] == swn for x in s['dsts'])]

    h = fabric_bound(entering(sa), prio, d); hops.append((f'fabric {sa}', h)); d += h
    if sa != sb:
        crossing = [s for s in st if sw[s['src']] == sa and any(sw[x] == sb for x in s['dsts'])]
        h = port_bound(crossing, prio, trunk_capacity_for(scenario), d)
        hops.append((f'trunk {sa}->{sb}', h)); d += h
        h = fabric_bound(entering(sb), prio, d); hops.append((f'fabric {sb}', h)); d += h
    at_bbp = [s for s in st if 'BBP' in s['dsts']]
    cap = port_capacity('L5_OutPort_BBP', scenario)
    rho = sum(s['rate'] * s['bits'] for s in at_bbp) / cap
    h = port_bound(at_bbp, prio, cap, d) if rho < 1 else np.inf
    hops.append(('BBP egress', h)); d += h
    return d, hops, rho


if __name__ == '__main__':
    rows = []
    for scenario in ('base', 'upgraded'):
        for n in range(1, 11):
            for tag, dt in (('steady', 1000.0), ('burst', 4.0)):
                d, hops, rho = sv_to_bbp_bound(n, scenario, dt)
                rows.append({'Scenario': scenario, 'N_Bays': n, 'Snapshot': tag, 'Rho_BBP': rho,
                             'NC_bound_us': d * 1e6,
                             **{f'hop_{k}_us': v * 1e6 for k, v in hops}})
                print(scenario, n, tag, round(rho, 3), round(d * 1e6, 1),
                      [(k, round(v * 1e6, 1)) for k, v in hops], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'nc_bound.csv'), index=False, float_format='%.3f')
