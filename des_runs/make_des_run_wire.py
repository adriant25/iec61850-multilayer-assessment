"""
Build an isolated DES run folder (make_des_run.py) whose frame sizes include
the 20 B Ethernet physical overhead (preamble/SFD 8 B + inter-frame gap 12 B):
the simulator serializes BYTES*8/C, so adding 20 B to every frame makes the
serialization equal to the occupation of the link. Only the copied inputs and
the copied traffic generator of the run folder are modified.

usage: make_des_run_wire.py <run_dir> <N> <seed> [fix] [link1000]
"""
import os
import re
import subprocess
import sys

import openpyxl

HERE = os.path.dirname(os.path.abspath(__file__))
OVERHEAD = 20

run_dir, n, seed = sys.argv[1], sys.argv[2], sys.argv[3]
fix = 'fix' in sys.argv[4:]
link1000 = 'link1000' in sys.argv[4:]
args = [sys.executable, os.path.join(HERE, 'make_des_run.py'), run_dir, n, '']
if fix:
    args.append('fix')
subprocess.run(args, check=True, stdout=subprocess.DEVNULL)

inp = os.path.join(run_dir, 'Inputs')
for name in ('sv_traffic.xlsx', 'goose_pb_traffic.xlsx', 'goose_sb_traffic.xlsx'):
    path = os.path.join(inp, name)
    wb = openpyxl.load_workbook(path)
    for ws in wb.worksheets:
        hdr = [c.value for c in ws[1]]
        if 'BYTES' not in hdr:
            continue
        col = hdr.index('BYTES') + 1
        for r in range(2, ws.max_row + 1):
            v = ws.cell(r, col).value
            if isinstance(v, (int, float)):
                ws.cell(r, col).value = v + OVERHEAD
    wb.save(path)

# Input-format normalization: the simulator splits VLAN port lists on ';'
# (the N=6 station-bus table used ','), so commas are replaced; sheets beyond
# the number of switches are reported (the simulator then floods that bus).
for name in ('pb_vlans.xlsx', 'sb_vlans.xlsx'):
    path = os.path.join(inp, name)
    wb = openpyxl.load_workbook(path)
    changed = 0
    for ws in wb.worksheets:
        hdr = [c.value for c in ws[1]]
        if 'PORTS' not in hdr:
            continue
        col = hdr.index('PORTS') + 1
        for r in range(2, ws.max_row + 1):
            v = ws.cell(r, col).value
            if isinstance(v, str) and ',' in v:
                ws.cell(r, col).value = v.replace(',', ';')
                changed += 1
    if changed:
        wb.save(path)
        print(f'{name}: {changed} port lists with "," normalized to ";"')

tg = os.path.join(run_dir, 'traffic_generator.py')
src = open(tg, encoding='utf-8').read()
old = 'message_sizes = {"Announce": 100, "Sync": 60, "Follow_Up": 58}'
assert old in src
src = src.replace(old, 'message_sizes = {"Announce": 120, "Sync": 80, "Follow_Up": 78}  # + 20 B wire overhead')
open(tg, 'w', encoding='utf-8').write(src)

if link1000:
    # trunk + BBP links at 1 Gbps (upgraded scenario of the paper)
    path = os.path.join(inp, 'pb_switches.xlsx')
    wb = openpyxl.load_workbook(path)
    for ws in wb.worksheets:
        hdr = [c.value for c in ws[1]]
        ce, cl, ct = hdr.index('EQUIPMENT') + 1, hdr.index('LINK (Mbps)') + 1, hdr.index('TYPE') + 1
        for r in range(2, ws.max_row + 1):
            eq = str(ws.cell(r, ce).value)
            typ = str(ws.cell(r, ct).value)
            if eq.startswith('BBP') or 'TRUNK' in eq.upper() or 'TRUNK' in typ.upper():
                ws.cell(r, cl).value = 1000
    wb.save(path)

with open(os.path.join(run_dir, 'answers.txt'), 'w') as fh:
    fh.write(f'7\n-1\n-1\n7\n7\n4800\nn\n{seed}\ny\nn\nn\nn\n')
print('built', run_dir, 'fix' if fix else '', 'link1000' if link1000 else '')
