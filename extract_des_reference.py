# -*- coding: utf-8 -*-
"""
extract_des_reference.py -- Build des_reference.csv from the DES KPI workbooks.

The packet-level simulator writes one KPIs_PB.xlsx per run (sheet
'KPIs_per_Flow', one row per flow and destination). This script condenses the
runs used in the paper into one table, so the analytical-vs-DES comparison is
driven by recorded simulator output instead of hand-typed numbers.

Per (N, scenario) it stores, over SV flows (one row per flow/destination):
  SV_mean_us   : unweighted mean of per-flow average delay (paper convention)
  SV_max_us    : maximum per-flow maximum delay
  SV_PLR_pct   : mean per-flow packet loss ratio [%]
and over the 50BF-cascade GOOSE flows modelled analytically -- trip
(PPxBnL -> PPyBnL / MUxBn, same bay) and breaker status (MUxBn -> PPxBnL, same
bay, and MUxBn -> BBP):
  GOOSE_mean_us, GOOSE_max_us

Usage:  python extract_des_reference.py   (edit DES_RUNS if runs move)
"""

import os
import re
import pandas as pd

SIM = r"C:\Users\adria\OneDrive - Universidad de los andes\Uniandes\2025-1\Simulación"


def _case(n: int) -> str:
    return os.path.join(SIM, f"CASO{n}BAHIA{'' if n == 1 else 'S'}")


# (N, scenario) -> KPI workbook. All runs use frame sizes on the wire (frame +
# 20 B preamble/SFD and inter-frame gap, see make_des_run_wire.py) and the
# current simulator version; N=9 uses the corrected VLAN/burst input tables.
# The 1 Gbps runs (trunk + BBP link) are included once they exist.
DES_RUNS = {
    **{(n, 'base'): os.path.join(_case(n), 'Results_wire', 'KPIs_PB.xlsx') for n in range(1, 11)},
    (9, 'upgraded'):  os.path.join(_case(9), '1000_wire', 'KPIs_PB.xlsx'),
    (10, 'upgraded'): os.path.join(_case(10), 'Results1000_wire', 'KPIs_PB.xlsx'),
}

# Runs of the first version of the study (frame sizes without the 20 B
# physical overhead), kept for traceability only.
DES_RUNS_FRAME_ONLY = {
    **{(n, 'base'): os.path.join(_case(n), 'Results', 'KPIs_PB.xlsx') for n in range(2, 9)},
    (1, 'base'):      os.path.join(_case(1), 'Results_rerun', 'KPIs_PB.xlsx'),
    (9, 'base'):      os.path.join(_case(9), '100_fix', 'Results', 'KPIs_PB.xlsx'),
    (9, 'upgraded'):  os.path.join(_case(9), '1000_fix', 'Results', 'KPIs_PB.xlsx'),
    (10, 'base'):     os.path.join(_case(10), 'Results100', 'KPIs_PB.xlsx'),
    (10, 'upgraded'): os.path.join(_case(10), 'Results1000', 'KPIs_PB.xlsx'),
}


def _same_bay(src: str, dst: str) -> bool:
    a, b = re.search(r'B(\d+)', src), re.search(r'B(\d+)', dst)
    return bool(a and b and a.group(1) == b.group(1))


def summarize(path: str) -> dict:
    k = pd.read_excel(path, sheet_name='KPIs_per_Flow')
    sv = k[k.Traffic_Type == 'SV']
    g = k[k.Traffic_Type == 'GOOSE']
    src, dst = g.Source.astype(str), g.Destination.astype(str)
    trip = src.str.fullmatch(r'PP[12]B\d+L') & dst.str.fullmatch(r'PP[12]B\d+L|MU[12]B\d+')
    status = src.str.fullmatch(r'MU[12]B\d+') & dst.str.fullmatch(r'PP[12]B\d+L|BBP')
    go = g[trip | status]
    go = go[[d == 'BBP' or _same_bay(s, d) for s, d in zip(go.Source, go.Destination)]]
    return {
        'SV_flows':      len(sv),
        'SV_mean_us':    sv.Average_Delay_us.mean(),
        'SV_max_us':     sv.Max_Delay_us.max(),
        'SV_PLR_pct':    100 * sv.PLR_Total.mean(),
        'GOOSE_flows':   len(go),
        'GOOSE_mean_us': go.Average_Delay_us.mean(),
        'GOOSE_max_us':  go.Max_Delay_us.max(),
    }


# Independent replications (different seeds, current simulator version) of the
# two configurations next to the stability boundary; the reference value is the
# mean over replications (see des_replicates.py for the confidence intervals).
DES_REPLICATES = {
    (n, 'base'): [os.path.join(_case(n), 'Results_wire', 'KPIs_PB.xlsx')] +
                 [os.path.join(_case(n), 'Replicas_wire', f'seed_{k}', 'KPIs_PB.xlsx')
                  for k in (1, 2, 3, 4, 5)]
    for n in (7, 8)
}


def summarize_replicates(paths: list[str]) -> dict:
    runs = pd.DataFrame([summarize(p) for p in paths])
    out = runs.mean().to_dict()
    out.update({c: runs[c].max() for c in runs if c.endswith('_max_us')})
    out['Replications'] = len(paths)
    return out


if __name__ == '__main__':
    rows = []
    for (n, scenario), path in sorted(DES_RUNS.items()):
        reps = [p for p in DES_REPLICATES.get((n, scenario), []) if os.path.exists(p)]
        if reps:
            rows.append({'N_Bays': n, 'Scenario': scenario, **summarize_replicates(reps),
                         'Source': f'mean of {len(reps)} replications'})
            continue
        if not os.path.exists(path):
            print(f'missing: N={n} {scenario}: {path}')
            continue
        rows.append({'N_Bays': n, 'Scenario': scenario, **summarize(path), 'Replications': 1,
                     'Source': os.path.relpath(path, SIM)})
    df = pd.DataFrame(rows)
    df.to_csv('des_reference.csv', index=False, float_format='%.4f')
    print(df.round(3).to_string(index=False))
