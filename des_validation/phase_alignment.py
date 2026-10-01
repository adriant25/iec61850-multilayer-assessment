# -*- coding: utf-8 -*-
"""
phase_alignment.py -- Sensitivity of the SV queuing delay to phase alignment
(PTP-synchronized merging units), with the simulator's own core.

K periodic SV streams (4800 Hz, 149 B) converge on one 100 Mbps egress port,
as the BBP port with 2K/2 bays. The phase of stream i is drawn uniformly in
[0, delta*T]: delta = 1 is the independent random-phase case of the paper
(N*D/D/1), delta = 0 is perfect alignment (all frames released together).
Each delta is averaged over random phase realizations.

Reference values:
  random phases  -> N*D/D/1 mean wait (analytic model of the paper)
  full alignment -> mean (K-1)s/2 and maximum (K-1)s

Output: results/des_phase_alignment.csv
"""

import os
import numpy as np
import pandas as pd

import validate_des as v

PERIOD = 1 / 4800
KS = globals().get("KS_OVERRIDE", (8, 12, 16))   # streams
OUT = v.OUT


def run(k, delta, rng, t_sim=0.003):
    pk = {}
    for j in range(k):
        t0 = rng.uniform(0, delta * PERIOD) if delta > 0 else 0.0
        pk[f'MU{j + 1}'] = [(t, v.S_BYTES, 4) for t in np.arange(t0 + 1e-5, t_sim, PERIOD)]
    w = v.waits(v.run_des(pk, t_sim + 0.001), warmup=PERIOD)
    return w.w_egr.mean(), w.w_egr.max()


if __name__ == '__main__':
    rng = np.random.default_rng(21)
    rows = []
    for k in KS:
        for delta in (0.0, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
            reps = 1 if delta == 0 else 40
            res = np.array([run(k, delta, rng) for _ in range(reps)])
            rows.append({'K_streams': k, 'rho': k * v.S / PERIOD, 'delta': delta,
                         'DES_mean_wait_us': res[:, 0].mean() * 1e6,
                         'DES_max_wait_us': res[:, 1].max() * 1e6,
                         'NDD1_mean_us': v.ndd1_mean_wait(k - 1, PERIOD, v.S) * 1e6,
                         'Aligned_mean_us': (k - 1) * v.S / 2 * 1e6,
                         'Aligned_max_us': (k - 1) * v.S * 1e6})
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'des_phase_alignment.csv'), index=False, float_format='%.4f')
