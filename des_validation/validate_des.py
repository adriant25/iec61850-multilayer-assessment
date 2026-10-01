# -*- coding: utf-8 -*-
"""
validate_des.py -- Verification of the packet-level DES against queueing theory.

The harness imports the simulator's own modules (simulation_core, io_handler)
and runs its main loop, unchanged, on a minimal topology: K sources and one
sink on a single switch. Only the traffic generation is replaced by synthetic
arrival processes with a known analytical solution. Each arrived packet keeps
the timestamps added by the simulator,

    p[8] generation, p[9] arrival at the switch input port,
    p[10] entry into the egress queue, p[11] arrival at the sink,

so the waiting time of each queue is measured separately.

Cases
  A  Poisson source on its own link: M/D/1 (fixed size) and M/M/1
     (exponential size) waiting time, utilization 0.1-0.9.
  B  K periodic SV streams (4800 Hz, 149 B) with random phases converging on
     one egress port: N*D/D/1 mean wait (analytic model of the paper).
  C  Two priority classes of Poisson traffic at one egress port:
     non-preemptive priority (Cobham) waiting time per class.
  D  Overloaded egress port (rho > 1): the waiting time grows as (rho-1) t
     (fluid model of the paper).
  E  Zero load: end-to-end delay = 2 serializations + switch processing.

Usage:  python validate_des.py   (set DES_SIMULATOR_DIR to the Simulador folder
        if it is not in the default location)
Outputs: results/*.csv, results/fig_des_validation.png
"""

import os
import sys
import types
import time
import numpy as np
import pandas as pd

SIM_DIR = os.environ.get('DES_SIMULATOR_DIR') or os.path.join(
    os.path.expanduser('~'), 'OneDrive - Universidad de los andes', 'Uniandes', '2025-1',
    'Simulación', 'Simulador')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

# GUI-only dependencies of io_handler are not needed: stub them.
qt = types.ModuleType('PyQt5')
qtw = types.ModuleType('PyQt5.QtWidgets')
qtw.QApplication = qtw.QFileDialog = object
qt.QtWidgets = qtw
sys.modules.setdefault('PyQt5', qt)
sys.modules.setdefault('PyQt5.QtWidgets', qtw)
sys.path.insert(0, SIM_DIR)
import io_handler as io            # noqa: E402  (simulator module)
import simulation_core as core     # noqa: E402  (simulator module)
# N*D/D/1 mean wait exactly as used by the analytic model of the paper (the
# simulator has its own analysis.py, so the analytic module is loaded by path)
import importlib.util  # noqa: E402
_ANA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_ANA_DIR)                       # for its config.py import
_spec = importlib.util.spec_from_file_location('analytic_model', os.path.join(_ANA_DIR, 'analysis.py'))
_ana = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ana)
ndd1_mean_wait = _ana.ndd1_mean_wait

# Same parameters as the simulator (main.py)
PORT_BUFFER = 4_000_000
T_PROC = 2e-6
C = 100.0                           # link speed [Mbps]
S_BYTES = 149                       # SV frame
S = S_BYTES * 8 / (C * 1e6)         # service time of one frame [s]
SINK = 'SINK'
VLAN = 10


def run_des(packets: dict, t_sim: float, link_mbps: float = C, src_mbps: float | None = None,
            buffer_bytes: float = PORT_BUFFER, return_lost: bool = False):
    """Run the simulator's main loop (PB bus only, no GOOSE logic) on a star of
    len(packets) sources and one sink. packets = {source: [(t_gen, size, prio), ...]}.
    Returns the list of packets arrived at the sink (and, with return_lost, the
    packets dropped at the sink's egress queue)."""
    sources = list(packets)
    eq = sources + [SINK]
    switch_data = {'SW1': pd.DataFrame({'EQUIPMENT': eq, 'PORT': list(range(1, len(eq) + 1)),
                                        'LINK (Mbps)': [src_mbps or link_mbps] * (len(eq) - 1) + [link_mbps],
                                        'TYPE': ['IED'] * len(eq)})}
    vlan_tables = {'SW1': pd.DataFrame({'VLAN ID': [VLAN], 'PORTS': [str(len(eq))],
                                        'PRIORITY': [4]})}
    eqp = io.initialize_dict_by_equipment(switch_data, [])
    arrived = io.initialize_dict_by_equipment(switch_data, [])
    in_pkts = io.initialize_dict_by_equipment_and_priority(switch_data, [])
    in_log = io.initialize_dict_by_equipment(switch_data, [[0] * 9])
    out_log = io.initialize_dict_by_equipment(switch_data, [[0] * 9])
    out_pkts = io.initialize_dict_by_equipment_and_priority(switch_data, [])
    lost_in = io.initialize_dict_by_equipment(switch_data, [])
    lost_out = io.initialize_dict_by_equipment(switch_data, [])
    free = io.initialize_dict_by_equipment(switch_data, [0, 0])
    th_in = io.initialize_dict_by_equipment(switch_data, [[0] * 4])
    th_out = io.initialize_dict_by_equipment(switch_data, [[0] * 4])
    loop_drop = io.initialize_dict_by_equipment(switch_data, [])
    d_in = io.initialize_dict_by_equipment(switch_data, [])
    d_out = io.initialize_dict_by_equipment(switch_data, [])
    buf_in = io.initialize_dict_by_equipment(switch_data, 0)
    buf_out = io.initialize_dict_by_equipment(switch_data, 0)
    speed, sw_log, sw_names, info = io.prepare_bus_data(switch_data)

    for src, plist in packets.items():
        eqp[src] = [(size, VLAN, prio, src, 'SV', 1, ['Cre_' + src], k, t)
                    for k, (t, size, prio) in enumerate(sorted(plist))]

    delta_t = (10 * 8) / (speed * 1e6)            # simulator time step
    sw_free = [0]
    now = delta_t
    while now <= t_sim:
        core.process_output_port_overflow(out_pkts, lost_out, buffer_bytes, buf_out, out_log, now)
        core.process_output_queues(out_pkts, free, info, th_out, arrived, d_out, th_in, d_in,
                                   in_pkts, in_log, out_log, sw_log, sw_names, loop_drop, now, buf_out)
        ready = core.process_input_queues(in_pkts, lost_in, buffer_bytes, info, 1, now, buf_in, in_log)
        core.process_switch_forwarding(ready, in_pkts, out_pkts, in_log, out_log, sw_log, sw_free,
                                       info, sw_names, vlan_tables, now, T_PROC, buf_out, buf_in)
        core.process_equipment_to_port(eqp, free, info, sw_names, th_in, in_pkts, in_log, sw_log,
                                       d_in, now, buf_in)
        now += delta_t
    if return_lost:
        return arrived[SINK], lost_out[SINK]
    return arrived[SINK]


def waits(arr, warmup=0.0):
    a = np.array([[p[0], p[2], p[8], p[9], p[10], p[11]] for p in arr], dtype=float)
    a = a[a[:, 2] >= warmup]
    s = a[:, 0] * 8 / (C * 1e6)
    return pd.DataFrame({'size': a[:, 0], 'prio': a[:, 1], 't_gen': a[:, 2],
                         'w_src': a[:, 3] - a[:, 2] - s,          # wait on the source link
                         'w_egr': a[:, 5] - a[:, 4] - s,          # wait in the egress queue
                         'e2e': a[:, 5] - a[:, 2]})


# ---------------------------------------------------------------- theory
def md1(rho, s):  return rho * s / (2 * (1 - rho))
def mm1(rho, s):  return rho * s / (1 - rho)


def cobham(lams, s, prios):
    """Non-preemptive priority M/D/1 waits, prios descending order = service order."""
    R = sum(l * s * s / 2 for l in lams)
    out, sigma_prev = {}, 0.0
    for l, p in sorted(zip(lams, prios), key=lambda z: -z[1]):
        sigma = sigma_prev + l * s
        out[p] = R / ((1 - sigma_prev) * (1 - sigma))
        sigma_prev = sigma
    return out


# ---------------------------------------------------------------- cases
def case_a(rng, t_sim=3.0):
    rows = []
    for dist in ('D', 'M'):
        for rho in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            lam = rho / S
            n = rng.poisson(lam * t_sim)
            t = np.sort(rng.uniform(0, t_sim, n))
            size = np.full(n, S_BYTES, float) if dist == 'D' else rng.exponential(S_BYTES, n)
            arr = run_des({'SRC1': list(zip(t, size, [4] * n))}, t_sim + 0.05)
            w = waits(arr, warmup=0.05)
            th = md1(rho, S) if dist == 'D' else mm1(rho, S)
            rows.append({'Case': f'A_M/{dist}/1', 'rho': rho, 'Samples': len(w),
                         'DES_wait_us': w.w_src.mean() * 1e6, 'Theory_wait_us': th * 1e6})
            print(rows[-1])
    return pd.DataFrame(rows)


def case_b(rng, reps=60, t_sim=0.004):
    period = 1 / 4800
    rows = []
    for k in (2, 4, 6, 8, 10, 12, 14, 16):
        ws = []
        for _ in range(reps):
            pk = {}
            for j in range(k):
                t0 = rng.uniform(0, period)
                pk[f'MU{j + 1}'] = [(t, S_BYTES, 4) for t in np.arange(t0, t_sim, period)]
            w = waits(run_des(pk, t_sim + 0.001), warmup=period)
            ws.append(w.w_egr.mean())
        rows.append({'Case': 'B_N*D/D/1', 'K': k, 'rho': k * S / period,
                     'DES_wait_us': np.mean(ws) * 1e6,
                     'DES_ci95_us': 1.96 * np.std(ws, ddof=1) / np.sqrt(reps) * 1e6,
                     'Theory_wait_us': ndd1_mean_wait(k - 1, period, S) * 1e6})
        print(rows[-1])
    return pd.DataFrame(rows)


def case_c(rng, k=10, t_sim=1.0, share_high=0.3):
    rows = []
    for rho in (0.3, 0.5, 0.7, 0.85):
        lam = rho / S
        pk = {}
        for j in range(k):
            n = rng.poisson(lam / k * t_sim)
            t = np.sort(rng.uniform(0, t_sim, n))
            pr = np.where(rng.uniform(size=n) < share_high, 6, 4)
            pk[f'SRC{j + 1}'] = list(zip(t, [S_BYTES] * n, pr))
        w = waits(run_des(pk, t_sim + 0.05), warmup=0.05)
        th = cobham([lam * share_high, lam * (1 - share_high)], S, [6, 4])
        for p, name in ((6, 'high'), (4, 'low')):
            rows.append({'Case': f'C_priority_{name}', 'rho': rho, 'Samples': int((w.prio == p).sum()),
                         'DES_wait_us': w[w.prio == p].w_egr.mean() * 1e6,
                         'Theory_wait_us': th[p] * 1e6})
            print(rows[-1])
    return pd.DataFrame(rows)


def case_d(rng, k=20, t_sim=0.15):
    period = 1 / 4800
    pk = {f'MU{j + 1}': [(t, S_BYTES, 4) for t in np.arange(rng.uniform(0, period), t_sim, period)]
          for j in range(k)}
    w = waits(run_des(pk, t_sim + 0.02))
    rho = k * S / period
    w['Fluid_wait_us'] = (rho - 1) * w.t_gen * 1e6
    w['DES_wait_us'] = w.w_egr * 1e6
    slope = np.polyfit(w.t_gen, w.w_egr, 1)[0]
    print({'Case': 'D_overload', 'rho': rho, 'DES_slope': slope, 'Fluid_slope': rho - 1})
    return w[['t_gen', 'DES_wait_us', 'Fluid_wait_us']], rho, slope


def case_e():
    arr = run_des({'SRC1': [(1e-4, S_BYTES, 4)]}, 0.001)
    e2e = arr[0][11] - arr[0][8]
    th = 2 * S + T_PROC
    print({'Case': 'E_zero_load', 'DES_us': e2e * 1e6, 'Theory_us': th * 1e6})
    return pd.DataFrame([{'Case': 'E_zero_load', 'DES_e2e_us': e2e * 1e6, 'Theory_e2e_us': th * 1e6}])


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    t0 = time.time()
    e = case_e()
    a = case_a(rng)
    b = case_b(rng)
    c = case_c(rng)
    d, rho_d, slope_d = case_d(rng)
    summary = pd.concat([a, b, c, e], ignore_index=True)
    summary['Error_pct'] = 100 * (summary.DES_wait_us - summary.Theory_wait_us) / summary.Theory_wait_us
    summary.to_csv(os.path.join(OUT, 'des_validation_summary.csv'), index=False, float_format='%.4f')
    d.to_csv(os.path.join(OUT, 'des_validation_overload.csv'), index=False, float_format='%.4f')
    with open(os.path.join(OUT, 'des_validation_overload_slope.txt'), 'w') as fh:
        fh.write(f'rho={rho_d:.4f} DES_slope={slope_d:.5f} fluid_slope={rho_d - 1:.5f}\n')
    print(f'done in {time.time() - t0:.0f} s')
