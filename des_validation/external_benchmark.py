# -*- coding: utf-8 -*-
"""
external_benchmark.py -- Benchmark against published OPNET results
(Mekkanen et al., Procedia Computer Science 36 (2014) 72-79).

Published setup: K IEC 61850-9-2LE merging units (4000 frames/s, 126-byte
payload plus header; 144-byte Ethernet frame, i.e. 164 B on the wire with
preamble and inter-frame gap) publish SV to one IED through one Ethernet
switch; all MUs start synchronously (aligned phases); 100 Mb/s and 10 Mb/s.
Published results: 1 MU 26.66 us; about 6 us more per MU; 19 MUs 144.91 us;
20-23 MUs: latency grows without bound, 0.0958, 0.1960, 0.2712, 0.3629 s
(2 s run); 10 Mb/s: 1 MU 250.56 us, 2 MUs grows to 0.72237 s in 15 s.

Predictions with the expressions of the paper (no parameter fitted):
  service s = 8 L / C, fixed delay 2 s + T_proc (store-and-forward, one switch)
  aligned streams: mean wait (K-1) s / 2   (Section V, robustness)
  stability: K* = C / (8 L f)              (stability criterion)
  overload: wait at time t = (rho - 1) t   (fluid model)
and the simulator core run on the same configuration.

Output: results/external_benchmark.csv
"""

import os
import numpy as np
import pandas as pd

import validate_des as v

F = 4000.0
L_WIRE = 164            # bytes on the wire (144-byte frame + 20 B preamble/IFG)
L_FRAME = L_WIRE - 20   # bytes passed to the simulator: it does not add preamble/IFG
T_PROC = v.T_PROC       # 2 us, the value of the case study (not fitted)

OPNET_100 = {1: 26.6613, 19: 144.914583}
OPNET_OVER = {20: 0.0958, 21: 0.1960, 22: 0.2712, 23: 0.3629}       # s, 2 s run


def predict_mean(k, c):
    s = 8 * L_WIRE / c
    return (2 * s + T_PROC + (k - 1) * s / 2) * 1e6


def des_mean(k, c, t_sim):
    """Aligned streams; frame sized so that the serialization equals the wire size.
    Frames are generated during [0, t_sim]; the run continues until the queue has
    drained, and the buffer is large because the published runs report no loss."""
    v.C = c / 1e6
    period = 1 / F
    pk = {f'MU{j + 1}': [(t, L_WIRE, 4) for t in np.arange(1e-5, t_sim, period)] for j in range(k)}
    rho = k * 8 * L_WIRE * F / c
    t_end = t_sim + max(rho - 1, 0) * t_sim * 1.05 + 0.002
    arr = v.run_des(pk, t_end, link_mbps=c / 1e6, buffer_bytes=1e12)
    a = np.array([[p[8], p[11]] for p in arr])
    return a, (a[:, 1] - a[:, 0])


if __name__ == '__main__':
    rows = []
    c = 100e6
    k_star = c / (8 * L_WIRE * F)
    print('K* =', k_star)
    for k in (1, 5, 10, 15, 19):
        a, d = des_mean(k, c, 0.01)
        rows.append({'Case': '100 Mb/s stable', 'MUs': k, 'OPNET_us': OPNET_100.get(k, np.nan),
                     'Predicted_us': predict_mean(k, c), 'DES_us': d.mean() * 1e6})
        print(rows[-1], flush=True)
    for k, t_opnet in OPNET_OVER.items():
        rho = k * 8 * L_WIRE * F / c
        a, d = des_mean(k, c, 2.0)
        last = d[a[:, 0] > 1.99].max()
        rows.append({'Case': '100 Mb/s overload (2 s)', 'MUs': k, 'rho': rho, 'OPNET_us': t_opnet * 1e6,
                     'Predicted_us': (rho - 1) * 2.0 * 1e6, 'DES_us': last * 1e6})
        print(rows[-1], flush=True)
    c = 10e6
    a, d = des_mean(1, c, 0.05)
    rows.append({'Case': '10 Mb/s stable', 'MUs': 1, 'OPNET_us': 250.5581,
                 'Predicted_us': predict_mean(1, c), 'DES_us': d.mean() * 1e6})
    print(rows[-1], flush=True)
    rho = 2 * 8 * L_WIRE * F / c
    a, d = des_mean(2, c, 15.0)
    rows.append({'Case': '10 Mb/s overload (15 s)', 'MUs': 2, 'rho': rho, 'OPNET_us': 0.72237e6,
                 'Predicted_us': (rho - 1) * 15.0 * 1e6, 'DES_us': d[a[:, 0] > 14.99].max() * 1e6})
    print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df['Pred_err_pct'] = 100 * (df.Predicted_us / df.OPNET_us - 1)
    df['DES_err_pct'] = 100 * (df.DES_us / df.OPNET_us - 1)
    df.to_csv(os.path.join(v.OUT, 'external_benchmark.csv'), index=False, float_format='%.3f')
    print(df.round(2).to_string(index=False))
