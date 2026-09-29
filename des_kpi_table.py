# -*- coding: utf-8 -*-
"""
des_kpi_table.py -- LaTeX table of DES KPIs for representative configurations.

Statistics are taken over flows (one row per flow and destination in the
'KPIs_per_Flow' sheet): per-flow average delay (min / mean / max over flows),
maximum packet delay, and per-flow packet loss ratio (mean / max over flows).
Writes results/paper/des_kpi_table.tex.
"""

import os
import pandas as pd

from extract_des_reference import DES_RUNS

COLUMNS = [((1, 'base'), '1 Bay'), ((5, 'base'), '5 Bays'), ((9, 'base'), '9 Bays'),
           ((9, 'upgraded'), '9 Bays (1 Gbps)'), ((10, 'base'), '10 Bays'),
           ((10, 'upgraded'), '10 Bays (1 Gbps)')]


def fmt(x: float) -> str:
    if x == 0:
        return '0'
    if abs(x) >= 1e4:
        return f'{x:,.0f}'.replace(',', '\\,')
    if abs(x) >= 100:
        return f'{x:.1f}'
    return f'{x:.2f}'


def stats(path: str) -> dict:
    k = pd.read_excel(path, sheet_name='KPIs_per_Flow')
    out = {}
    for cls in ('SV', 'GOOSE'):
        f = k[k.Traffic_Type == cls]
        out[(cls, 'avg_min')] = f.Average_Delay_us.min()
        out[(cls, 'avg_mean')] = f.Average_Delay_us.mean()
        out[(cls, 'avg_max')] = f.Average_Delay_us.max()
        out[(cls, 'pkt_max')] = f.Max_Delay_us.max()
        out[(cls, 'plr_mean')] = 100 * f.PLR_Total.mean()
        out[(cls, 'plr_max')] = 100 * f.PLR_Total.max()
    return out


ROWS = [('SV', 'avg_min', r'Flow avg.\ delay, min ($\mu$s)'),
        ('SV', 'avg_mean', r'Flow avg.\ delay, mean ($\mu$s)'),
        ('SV', 'avg_max', r'Flow avg.\ delay, max ($\mu$s)'),
        ('SV', 'pkt_max', r'Packet delay, max ($\mu$s)'),
        ('SV', 'plr_mean', r'PLR, mean over flows (\%)'),
        ('SV', 'plr_max', r'PLR, worst flow (\%)'),
        ('GOOSE', 'avg_min', r'Flow avg.\ delay, min ($\mu$s)'),
        ('GOOSE', 'avg_mean', r'Flow avg.\ delay, mean ($\mu$s)'),
        ('GOOSE', 'avg_max', r'Flow avg.\ delay, max ($\mu$s)'),
        ('GOOSE', 'pkt_max', r'Packet delay, max ($\mu$s)'),
        ('GOOSE', 'plr_mean', r'PLR, mean over flows (\%)'),
        ('GOOSE', 'plr_max', r'PLR, worst flow (\%)')]

if __name__ == '__main__':
    data = {key: stats(DES_RUNS[key]) for key, _ in COLUMNS}
    lines = [r'\begin{table*}[!t]', r'\centering',
             r'\caption{DES KPIs of SV and GOOSE traffic (all flows, 7~s run) for representative configurations; '
             r'``1 Gbps'' denotes the trunk + BBP link upgrade.}',
             r'\label{tab:kpi_summary}', r'\resizebox{\textwidth}{!}{',
             r'\begin{tabular}{ll' + 'c' * len(COLUMNS) + '}', r'\toprule',
             'Class & KPI & ' + ' & '.join(f'\\textbf{{{h}}}' for _, h in COLUMNS) + r' \\',
             r'\midrule']
    prev = None
    for cls, key, label in ROWS:
        if prev and prev != cls:
            lines.append(r'\midrule')
        first = cls if prev != cls else ''
        prev = cls
        vals = ' & '.join(fmt(data[c][(cls, key)]) for c, _ in COLUMNS)
        lines.append(f'{first} & {label} & {vals} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table*}']
    os.makedirs(os.path.join('results', 'paper'), exist_ok=True)
    path = os.path.join('results', 'paper', 'des_kpi_table.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
