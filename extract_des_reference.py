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


# (N, scenario) -> KPI workbook. N=1 and N=9 point to re-runs with the current
# simulator version (N=9 with the corrected VLAN/burst input tables).
DES_RUNS = {
    **{(n, 'base'): os.path.join(_case(n), 'Results', 'KPIs_PB.xlsx') for n in range(2, 9)},
    (1, 'base'):      os.path.join(_case(1), 'Results_rerun', 'KPIs_PB.xlsx'),
    (9, 'base'):      os.path.join(_case(9), '100_fix', 'Results', 'KPIs_PB.xlsx'),
    # Falls back to the original '1000' run until the corrected re-run exists
    # (the input-table errors of N=9 affect GOOSE statistics, not SV).
    (9, 'upgraded'):  next(p for p in (
        os.path.join(_case(9), '1000_fix', 'Results', 'KPIs_PB.xlsx'),
        os.path.join(_case(9), '1000', 'Results', 'KPIs_PB.xlsx')) if os.path.exists(p)),
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


if __name__ == '__main__':
    rows = []
    for (n, scenario), path in sorted(DES_RUNS.items()):
        if not os.path.exists(path):
            print(f'missing: N={n} {scenario}: {path}')
            continue
        rows.append({'N_Bays': n, 'Scenario': scenario, **summarize(path),
                     'Source': os.path.relpath(path, SIM)})
    df = pd.DataFrame(rows)
    df.to_csv('des_reference.csv', index=False, float_format='%.4f')
    print(df.round(3).to_string(index=False))
