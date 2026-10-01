# -*- coding: utf-8 -*-
"""
validate_extra.py -- Additional DES verification cases (same harness as validate_des.py).

  F  Waiting-time distributions, not only means:
       - N*D/D/1 complementary distribution P(W > x) versus the Benes tail
         used in the analytic model (K = 12 and 16 SV streams);
       - M/D/1 (Erlang's exact distribution) and M/M/1 (rho e^{-(1-rho)t/s}).
  G  Buffer overflow and loss: overloaded egress port with a finite buffer;
     fill time, plateau wait 8B/C and loss ratio (rho-1)/rho of the fluid model.
  H  Mixed traffic as in the paper: periodic SV streams (P4) together with
     Poisson higher-priority (P6) and lower-priority (P1) traffic on one egress
     port, compared with the analytic egress model (N*D/D/1 + Cobham).
  I  Non-preemptive priority (Cobham) with independent replications and 95 %
     confidence intervals.

Usage:  python validate_extra.py {F|G|H|I}
"""

import os
import sys
from math import comb, factorial, floor
import numpy as np
import pandas as pd

import validate_des as v

S, S_BYTES, C = v.S, v.S_BYTES, v.C
PERIOD = 1 / 4800
OUT = v.OUT


# ------------------------------------------------------------ theory helpers
def ndd1_ccdf(x, n_other, D):
    """Benes / Roberts-Virtamo tail Q(x) = P(V > x), x in service units
    (same expression as the analytic model)."""
    out = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(x):
        tot = 0.0
        for k in range(int(np.floor(xi)) + 1, n_other + 1):
            a = (k - xi) / D
            if a >= 1.0:
                continue
            tot += comb(n_other, k) * a ** k * (1 - a) ** (n_other - k) * (D - n_other + xi) / (D - k + xi)
        out[i] = tot
    return out


def md1_cdf(t, rho, s):
    """Erlang's exact M/D/1 waiting-time distribution P(W <= t)."""
    lam = rho / s
    m = floor(t / s)
    return (1 - rho) * sum((lam * (k * s - t)) ** k / factorial(k) * np.exp(-lam * (k * s - t))
                           for k in range(m + 1))


def empirical_ccdf(w, grid):
    w = np.asarray(w, dtype=float)
    w = np.sort(np.where(np.abs(w) < 1e-6, 0.0, w))   # floating-point residue of zero waits
    return 1.0 - np.searchsorted(w, grid, side='right') / len(w)


# ------------------------------------------------------------ cases
def case_f():
    rng = np.random.default_rng(11)
    rows = []
    # N*D/D/1
    for k in (12, 16):
        ws = []
        for _ in range(300):
            pk = {f'MU{j + 1}': [(t, S_BYTES, 4) for t in np.arange(rng.uniform(0, PERIOD), 0.003, PERIOD)]
                  for j in range(k)}
            ws.append(v.waits(v.run_des(pk, 0.004), warmup=PERIOD).w_egr.values / S)
        w = np.concatenate(ws)
        grid = np.linspace(0, k - 1, 60)
        th = ndd1_ccdf(grid, k - 1, PERIOD / S)
        for g, e, t in zip(grid, empirical_ccdf(w, grid), th):
            rows.append({'Model': f'N*D/D/1 K={k}', 'x_service_units': g, 'DES_ccdf': e, 'Theory_ccdf': t,
                         'Samples': len(w)})
        print('ND/D/1', k, len(w), flush=True)
    # M/D/1 and M/M/1 on a single Poisson source link
    for dist, rho in (('D', 0.5), ('D', 0.8), ('M', 0.5), ('M', 0.8)):
        lam, t_sim = rho / S, 3.0
        n = rng.poisson(lam * t_sim)
        t = np.sort(rng.uniform(0, t_sim, n))
        size = np.full(n, S_BYTES, float) if dist == 'D' else rng.exponential(S_BYTES, n)
        w = v.waits(v.run_des({'SRC1': list(zip(t, size, [4] * n))}, t_sim + 0.05), warmup=0.05).w_src.values / S
        grid = np.linspace(0, 12 if rho == 0.5 else 25, 60)
        if dist == 'D':
            th = np.array([1 - md1_cdf(g * S, rho, S) for g in grid])
        else:
            th = rho * np.exp(-(1 - rho) * grid)
        for g, e, tt in zip(grid, empirical_ccdf(w, grid), th):
            rows.append({'Model': f'M/{dist}/1 rho={rho}', 'x_service_units': g, 'DES_ccdf': e,
                         'Theory_ccdf': tt, 'Samples': len(w)})
        print('M/', dist, rho, len(w), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, 'des_validation_ccdf.csv'), index=False, float_format='%.6g')
    # Kolmogorov-type distance between the curves
    print(df.groupby('Model').apply(lambda d: (d.DES_ccdf - d.Theory_ccdf).abs().max()).round(4))


def case_g():
    rng = np.random.default_rng(12)
    buffer_b = 100_000
    rows, series = [], []
    for k in (20, 24):
        t_sim = 0.4
        pk = {f'MU{j + 1}': [(t, S_BYTES, 4) for t in np.arange(rng.uniform(0, PERIOD), t_sim, PERIOD)]
              for j in range(k)}
        arr, lost = v.run_des(pk, t_sim + 0.02, buffer_bytes=buffer_b, return_lost=True)
        rho = k * S / PERIOD
        cap = C * 1e6
        t_fill = 8 * buffer_b / (cap * (rho - 1))
        w = v.waits(arr)
        t_lost = np.array([p[8] for p in lost])
        t_start = t_fill + 0.02
        gen_after = sum(1 for src in pk for (t, _, _) in pk[src] if t_start <= t <= t_sim - 0.02)
        lost_after = int(((t_lost >= t_start) & (t_lost <= t_sim - 0.02)).sum())
        plateau = w[(w.t_gen >= t_start) & (w.t_gen <= t_sim - 0.02)].w_egr
        first_loss = t_lost.min() if t_lost.size else np.nan
        rows.append({'K': k, 'rho': rho, 'Buffer_B': buffer_b,
                     'Fill_time_DES_ms': first_loss * 1e3, 'Fill_time_fluid_ms': t_fill * 1e3,
                     'Loss_ratio_DES': lost_after / gen_after, 'Loss_ratio_fluid': (rho - 1) / rho,
                     'Plateau_wait_DES_ms': plateau.mean() * 1e3, 'Plateau_wait_fluid_ms': 8 * buffer_b / cap * 1e3})
        print(rows[-1], flush=True)
        ws = w[['t_gen', 'w_egr']].copy()
        ws['K'] = k
        series.append(ws.iloc[::5])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, 'des_validation_loss.csv'), index=False, float_format='%.5f')
    pd.concat(series).to_csv(os.path.join(OUT, 'des_validation_loss_series.csv'), index=False, float_format='%.6g')


def egress_theory(n_sv, rho_h, rho_l, s_h, s_l):
    """Analytic egress model of the paper for P6 (Poisson), P4 (SV) and P1 (Poisson)."""
    cap = C * 1e6
    stats = {p: {'bps': 0.0, 'ssf': 0.0, 'sf': 0.0, 'n_sv': 0, 'bps_sv': 0.0} for p in range(8)}
    for p, rho, sb in ((6, rho_h, s_h), (1, rho_l, s_l)):
        f = rho * cap / (sb * 8)
        stats[p].update(bps=rho * cap, ssf=f * sb * sb, sf=f * sb)
    f_sv = n_sv / PERIOD
    stats[4].update(bps=f_sv * S_BYTES * 8, ssf=f_sv * S_BYTES ** 2, sf=f_sv * S_BYTES,
                    n_sv=n_sv, bps_sv=f_sv * S_BYTES * 8)
    return {p: v._ana.egress_wait(stats, p, cap) for p in (6, 4, 1)}


def case_h():
    rng = np.random.default_rng(13)
    n_sv, s_h, s_l = 8, 171, 149
    rows = []
    for rho_h, rho_l in ((0.05, 0.10), (0.10, 0.20), (0.20, 0.20)):
        reps, t_sim = 12, 0.06
        acc = {6: [], 4: [], 1: []}
        for _ in range(reps):
            pk = {f'MU{j + 1}': [(t, S_BYTES, 4) for t in np.arange(rng.uniform(0, PERIOD), t_sim, PERIOD)]
                  for j in range(n_sv)}
            for name, rho, sb, prio in (('H', rho_h, s_h, 6), ('L', rho_l, s_l, 1)):
                lam = rho * C * 1e6 / (sb * 8)
                for j in range(4):
                    n = rng.poisson(lam / 4 * t_sim)
                    pk[f'{name}{j + 1}'] = list(zip(np.sort(rng.uniform(0, t_sim, n)), [sb] * n, [prio] * n))
            w = v.waits(v.run_des(pk, t_sim + 0.005, src_mbps=1000.0), warmup=0.005)
            for p in acc:
                acc[p].append(w[w.prio == p].w_egr.mean())
        th = egress_theory(n_sv, rho_h, rho_l, s_h, s_l)
        for p, name in ((6, 'P6 Poisson'), (4, 'P4 SV periodic'), (1, 'P1 Poisson')):
            a = np.array(acc[p])
            rows.append({'rho_SV': n_sv * S / PERIOD, 'rho_P6': rho_h, 'rho_P1': rho_l, 'Class': name,
                         'DES_wait_us': a.mean() * 1e6,
                         'DES_ci95_us': 1.96 * a.std(ddof=1) / np.sqrt(reps) * 1e6,
                         'Theory_wait_us': th[p] * 1e6})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df['Error_pct'] = 100 * (df.DES_wait_us - df.Theory_wait_us) / df.Theory_wait_us
    df.to_csv(os.path.join(OUT, 'des_validation_mixed.csv'), index=False, float_format='%.4f')


def case_i():
    rng = np.random.default_rng(14)
    k, share, reps, t_sim = 10, 0.3, 6, 0.25
    rows = []
    for rho in (0.5, 0.7, 0.85):
        lam = rho / S
        acc = {6: [], 4: []}
        for _ in range(reps):
            pk = {}
            for j in range(k):
                n = rng.poisson(lam / k * t_sim)
                t = np.sort(rng.uniform(0, t_sim, n))
                pr = np.where(rng.uniform(size=n) < share, 6, 4)
                pk[f'SRC{j + 1}'] = list(zip(t, [S_BYTES] * n, pr))
            w = v.waits(v.run_des(pk, t_sim + 0.01, src_mbps=1000.0), warmup=0.01)
            for p in acc:
                acc[p].append(w[w.prio == p].w_egr.mean())
        th = v.cobham([lam * share, lam * (1 - share)], S, [6, 4])
        for p, name in ((6, 'high'), (4, 'low')):
            a = np.array(acc[p])
            rows.append({'Class': name, 'rho': rho, 'DES_wait_us': a.mean() * 1e6,
                         'DES_ci95_us': 1.96 * a.std(ddof=1) / np.sqrt(reps) * 1e6,
                         'Theory_wait_us': th[p] * 1e6})
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df['Error_pct'] = 100 * (df.DES_wait_us - df.Theory_wait_us) / df.Theory_wait_us
    df.to_csv(os.path.join(OUT, 'des_validation_priority_reps.csv'), index=False, float_format='%.4f')


if __name__ == '__main__':
    {'F': case_f, 'G': case_g, 'H': case_h, 'I': case_i}[sys.argv[1]]()
