#!/usr/bin/env python3
# Merges benchmark result files into a markdown table.
# usage: table.py <row|col> [--paper] [--eigen] label=file ...   (label ending in '*' is the ratio reference)
import csv, math, os, sys
here = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(here, '..')

def load(path):
    d = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        key = '%sx%sx%s' % (f[1], f[2], f[3])
        d[key] = (int(f[0]), float(f[4]))
    return d

def paper(fig, series):
    d = {}
    for r in csv.DictReader(open(os.path.join(root, 'results', 'paper_numbers.csv'))):
        if r['figure'] == fig and r['series'] == series:
            d[r['id']] = float(r['gflops'])
    return d

def main():
    order = sys.argv[1]
    args = sys.argv[2:]
    cols = []
    ref = None
    for a in args:
        if a == '--paper':
            cols.append(('paper MpGEMM', 'paper', paper('single%smajor' % order, 'MpGEMM')))
            cols.append(('paper Accel', 'paper', paper('single%smajor' % order, 'Accelerate')))
        elif a == '--paper-multi':
            cols.append(('paper MpGEMM 2x', 'paper', paper('multi%smajor' % order, 'MpGEMM')))
            cols.append(('paper Accel 2x', 'paper', paper('multi%smajor' % order, 'Accelerate')))
        elif a == '--paper-fp64':
            cols.append(('paper MpGEMM', 'paper', paper('singlerowmajor-fp64', 'MpGEMM-Serial')))
            cols.append(('paper Accel', 'paper', paper('singlerowmajor-fp64', 'Accelerate-Serial')))
        elif a == '--paper-fp64-par':
            cols.append(('paper MpGEMM 2x', 'paper', paper('singlerowmajor-fp64', 'MpGEMM-Parallel')))
            cols.append(('paper Accel par', 'paper', paper('singlerowmajor-fp64', 'Accelerate-Parallel')))
        elif a == '--eigen':
            e = {}
            for line in open(os.path.join(root, 'bench', 'baseline', 'eigen_accel_paper_shapes_2026-09-22.txt')):
                f = line.split()
                if len(f) == 4 and 'x' in f[0]:
                    e[f[0]] = float(f[3])
            cols.append(('Eigen SME (!3164)', 'shape', e))
        else:
            label, path = a.split('=', 1)
            if label.endswith('*'):
                label = label[:-1]
                ref = label
            cols.append((label, 'file', load(path)))
    keys = []
    for _, kind, d in cols:
        if kind == 'file':
            for k, (i, _) in d.items():
                if k not in [x[0] for x in keys]:
                    keys.append((k, i))
    hdr = ['ID', 'M x N x K'] + [c[0] for c in cols]
    if ref:
        hdr += ['%s / %s' % (ref, c[0]) for c in cols if c[0] != ref]
    print('| ' + ' | '.join(hdr) + ' |')
    print('|' + '---|' * len(hdr))
    geo = {c[0]: [] for c in cols}
    sq = {c[0]: [] for c in cols}
    for k, i in keys:
        vals = []
        for label, kind, d in cols:
            if kind == 'file':
                v = d.get(k, (0, None))[1]
            elif kind == 'paper':
                v = d.get(str(i)) if i > 0 else None
            else:
                v = d.get(k)
            vals.append(v)
        row = [str(i) if i else '', k.replace('x', ' x ')] + ['%.0f' % v if v else '-' for v in vals]
        if ref:
            rv = vals[[c[0] for c in cols].index(ref)]
            row += ['%.2f' % (rv / v) if (v and rv) else '-' for (c, v) in zip(cols, vals) if c[0] != ref]
        print('| ' + ' | '.join(row) + ' |')
        for c, v in zip(cols, vals):
            if v:
                (geo if i else sq)[c[0]].append(v)
    square = all(len(set(k.split('x'))) == 1 for k, i in keys if not i)
    for name, gd in (('geomean IDs 1-24', geo), ('geomean squares' if square else 'geomean', sq)):
        if any(gd.values()):
            georow(name, gd, cols, ref)

def georow(name, geo, cols, ref):
    g = [math.exp(sum(map(math.log, geo[c[0]])) / len(geo[c[0]])) if geo[c[0]] else None for c in cols]
    row = ['', name] + ['%.0f' % v if v else '-' for v in g]
    if ref:
        rv = g[[c[0] for c in cols].index(ref)]
        row += ['%.2f' % (rv / v) if (v and rv) else '-' for (c, v) in zip(cols, g) if c[0] != ref]
    print('| ' + ' | '.join(row) + ' |')

main()
