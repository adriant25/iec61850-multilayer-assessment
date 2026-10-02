# -*- coding: utf-8 -*-
"""
des_replicates.py -- Statistical uncertainty of the DES reference values.

1. Across-flow confidence interval of every DES run used in the paper: the
   reported DES mean is the unweighted mean of the per-flow averages, so its
   95 % CI follows from the spread of those averages (t distribution).
2. Across-seed confidence interval for N = 8 and N = 9 (100 Mbps): independent
   replications of the full 7 s run with different random seeds (SV phases,
   GOOSE heartbeat offsets), stored in
       CASO8BAHIAS/Replicas/seed_<k>/KPIs_PB.xlsx
       CASO9BAHIAS/100_fix_replicas/seed_<k>/KPIs_PB.xlsx
   together with the original seed-42 runs used in the paper.

Outputs: results/des_flow_ci.csv, results/des_seed_replicates.csv,
         results/des_seed_summary.csv
"""

import glob
import os
import re
import numpy as np
import pandas as pd
from scipy import stats

from extract_des_reference import DES_RUNS, SIM, _case, summarize, _same_bay

OUT = 'results'


def flow_values(path: str) -> tuple[pd.Series, pd.Series]:
    k = pd.read_excel(path, sheet_name='KPIs_per_Flow')
    sv = k[k.Traffic_Type == 'SV'].Average_Delay_us
    g = k[k.Traffic_Type == 'GOOSE']
    src, dst = g.Source.astype(str), g.Destination.astype(str)
    trip = src.str.fullmatch(r'PP[12]B\d+L') & dst.str.fullmatch(r'PP[12]B\d+L|MU[12]B\d+')
    status = src.str.fullmatch(r'MU[12]B\d+') & dst.str.fullmatch(r'PP[12]B\d+L|BBP')
    go = g[trip | status]
    go = go[[d == 'BBP' or _same_bay(s, d) for s, d in zip(go.Source, go.Destination)]]
    return sv, go.Average_Delay_us


def ci95(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def replicate_paths() -> dict:
    """Seed replications (on-wire frame sizes) of the two sizes next to the boundary."""
    reps = {}
    for n in (7, 8):
        reps[(n, 42)] = os.path.join(_case(n), 'Results_wire', 'KPIs_PB.xlsx')
        for p in glob.glob(os.path.join(_case(n), 'Replicas_wire', 'seed_*', 'KPIs_PB.xlsx')):
            reps[(n, int(re.search(r'seed_(\d+)', p).group(1)))] = p
    return reps


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for (n, scen), path in sorted(DES_RUNS.items()):
        if not os.path.exists(path):
            continue
        sv, go = flow_values(path)
        rows.append({'N_Bays': n, 'Scenario': scen, 'SV_flows': len(sv), 'SV_mean_us': sv.mean(),
                     'SV_ci95_us': ci95(sv), 'GOOSE_flows': len(go), 'GOOSE_mean_us': go.mean(),
                     'GOOSE_ci95_us': ci95(go)})
    fc = pd.DataFrame(rows)
    fc['SV_ci95_pct'] = 100 * fc.SV_ci95_us / fc.SV_mean_us
    fc.to_csv(os.path.join(OUT, 'des_flow_ci.csv'), index=False, float_format='%.4f')
    print(fc.round(3).to_string(index=False))

    reps = replicate_paths()
    rr = []
    for (n, seed), path in sorted(reps.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        s = summarize(path)
        rr.append({'N_Bays': n, 'Seed': seed, **s, 'Source': os.path.relpath(path, SIM)})
    rr = pd.DataFrame(rr)
    rr.to_csv(os.path.join(OUT, 'des_seed_replicates.csv'), index=False, float_format='%.4f')
    print(rr.drop(columns='Source').round(3).to_string(index=False))

    summ = []
    for n, d in rr.groupby('N_Bays'):
        summ.append({'N_Bays': n, 'Replications': len(d),
                     **{f'{c}_mean': d[c].mean() for c in ('SV_mean_us', 'SV_PLR_pct', 'GOOSE_mean_us')},
                     **{f'{c}_ci95': ci95(d[c]) for c in ('SV_mean_us', 'SV_PLR_pct', 'GOOSE_mean_us')},
                     **{f'{c}_min': d[c].min() for c in ('SV_mean_us',)},
                     **{f'{c}_max': d[c].max() for c in ('SV_mean_us',)}})
    sm = pd.DataFrame(summ)
    sm.to_csv(os.path.join(OUT, 'des_seed_summary.csv'), index=False, float_format='%.4f')
    print(sm.round(3).to_string(index=False))
