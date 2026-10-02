# -*- coding: utf-8 -*-
"""
nc_panco.py -- Independent network-calculus bound of the SV flow towards the busbar
protection, computed with panco (A. Bouillard, https://github.com/anne-bou/panco,
static-priority module + FIFO analyses), as a check of nc_bound.py.

The network is generated from the same demand tensor as the analytical model:
one panco flow per (VLAN, publisher) stream, token bucket (one frame, rate),
priority class, maximum frame length (on-wire bits). Servers on, or upstream of,
the path of the flow of interest:

* publisher link of every device (rate C);
* switch fabric of SW1 and SW2: one frame per T_PROC is represented as a
  rate-latency server of rate L_ref / T_PROC with L_ref the on-wire SV frame;
  frames shorter than L_ref (PTP) are padded to L_ref, so every frame needs at
  least T_PROC (a valid upper bound of the serial fabric of the DES);
* trunk ports SW2->SW1 and SW1->SW2 (rate of the trunk);
* egress port towards the BBP.

Egress ports towards other devices are omitted: they are not on, nor upstream
of, the path of the flow of interest. A multicast stream that branches before
the BBP port is split into one flow per branch (counted twice on the shared
servers, which is conservative). Every server applies non-preemptive strict
priority (residual service curves of panco's SpNetwork) and FIFO inside a class;
the SV class is then analysed with TFA++, SFA and the polynomial LP method (PLP).

Setup (not versioned, see .gitignore):
  git clone https://github.com/anne-bou/panco tools/panco   (commit f035ccc)
  git -C tools/panco apply ../../panco_local_changes.patch   (only lets panco call a
      native lp_solve.exe instead of WSL; the analyses are unchanged)
  pip install -e tools/panco
  lp_solve 5.5.2.11 (lp_solve_5.5.2.11_exe_win64.zip, SourceForge) in tools/lp_solve;
  the executable is given by the environment variable PANCO_LPSOLVE.

Output: results/nc_panco.csv
"""

import os
import sys
import time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('PANCO_LPSOLVE', os.path.join(HERE, 'tools', 'lp_solve', 'lp_solve.exe'))

from panco.descriptor.curves import TokenBucket, RateLatency  # noqa: E402
from panco.descriptor.server import Server  # noqa: E402
from panco.staticpriorities.spFlow import SpFlow  # noqa: E402
from panco.staticpriorities.spNetwork import SpNetwork  # noqa: E402
from panco.fifo.tfaLP import TfaLP  # noqa: E402
from panco.fifo.sfaLP import SfaLP  # noqa: E402
from panco.fifo.fifoLP import FifoLP  # noqa: E402

from config import T_PROC_SWITCH, TRAFFIC_SPECS  # noqa: E402
from model import generate_devices, generate_demand_tensor, update_goose_demand  # noqa: E402
from analysis import trunk_capacity_for, port_capacity, CAPACITY_DEFAULT  # noqa: E402
from nc_bound import streams  # noqa: E402

L_REF = TRAFFIC_SPECS['SV']['size'] * 8          # on-wire SV frame [bit]
LP_DIR = os.path.join(HERE, 'results', 'panco_lp')


def build(n, scenario, dt):
    devs, sw = generate_devices(n)
    D0 = generate_demand_tensor(n, devs, 1000.0)
    st = streams(update_goose_demand(D0, n, dt), sw)
    switches = sorted(set(sw[s['src']] for s in st) | {sw['BBP']})
    sw_bbp = sw['BBP']

    servers, idx = [], {}

    def server(key, rate):
        if key not in idx:
            idx[key] = len(servers)
            servers.append(Server([RateLatency(rate, 0)], [TokenBucket(0, rate)]))
        return idx[key]

    r_fab = L_REF / T_PROC_SWITCH
    for s in switches:
        server(('fabric', s), r_fab)
    server(('bbp',), port_capacity('L5_OutPort_BBP', scenario))

    flows, foi = [], None
    for s in st:
        a = sw[s['src']]
        head = [server(('pub', s['src']), CAPACITY_DEFAULT), idx[('fabric', a)]]
        remote = sorted({sw[d] for d in s['dsts'] if d != 'BBP' and sw[d] != a})
        branches = []
        if 'BBP' in s['dsts']:
            if a == sw_bbp:
                branches.append(head + [idx[('bbp',)]])
            else:
                branches.append(head + [server(('trunk', a, sw_bbp), trunk_capacity_for(scenario)),
                                        idx[('fabric', sw_bbp)], idx[('bbp',)]])
        for b in remote:
            p = head + [server(('trunk', a, b), trunk_capacity_for(scenario)), idx[('fabric', b)]]
            if not any(q[:len(p)] == p for q in branches):
                branches.append(p)
        if not branches:
            branches.append(head)
        bits = max(s['bits'], L_REF)
        for p in branches:
            if s['sv'] and s['src'] == f'MU1B{n}' and 'BBP' in s['dsts'] and p[-1] == idx[('bbp',)]:
                foi = (len(flows), s['prio'])
            flows.append((s['prio'], p, bits, s['rate'] * s['bits']))

    # panco classes: 0 = highest priority
    prios = sorted({f[0] for f in flows}, reverse=True)
    cls = {p: k for k, p in enumerate(prios)}
    spflows = [SpFlow([TokenBucket(bits, rate)], path, bits, cls[pr]) for pr, path, bits, rate in flows]
    k_sv = cls[foi[1]]
    # index of the flow of interest inside its class network
    foi_in_class = sum(1 for i in range(foi[0]) if spflows[i].sp_class == k_sv)
    rho_bbp = sum(f[3] for f in flows if idx[('bbp',)] in f[1]) / port_capacity('L5_OutPort_BBP', scenario)
    return servers, spflows, k_sv, foi_in_class, rho_bbp


def analyse(n, scenario, dt):
    servers, spflows, k_sv, foi, rho = build(n, scenario, dt)
    os.makedirs(LP_DIR, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(LP_DIR)
    try:
        if rho >= 1:
            return rho, np.inf, np.inf, np.inf
        nets = SpNetwork(servers, spflows).equiv_network(False)
        net = nets[k_sv]
        tfa = TfaLP(net).delay(foi)
        sfa = SfaLP(net).delay(foi)
        plp = FifoLP(net, polynomial=True, sfa=True, tfa=True).delay(foi)
        return rho, tfa, sfa, plp
    finally:
        os.chdir(cwd)


if __name__ == '__main__':
    sizes = [int(a) for a in sys.argv[1:]] or list(range(1, 11))
    rows = []
    for scenario in ('base', 'upgraded'):
        for n in sizes:
            for tag, dt in (('steady', 1000.0), ('burst', 4.0)):
                t0 = time.perf_counter()
                rho, tfa, sfa, plp = analyse(n, scenario, dt)
                rows.append({'Scenario': scenario, 'N_Bays': n, 'Snapshot': tag, 'Rho_BBP': rho,
                             'TFA_us': tfa * 1e6, 'SFA_us': sfa * 1e6, 'PLP_us': plp * 1e6,
                             'Runtime_s': time.perf_counter() - t0})
                print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    out = os.path.join('results', 'nc_panco.csv' if len(sizes) == 10 else 'nc_panco_partial.csv')
    pd.DataFrame(rows).to_csv(out, index=False, float_format='%.3f')
