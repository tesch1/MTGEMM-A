#!/usr/bin/env python3
# Geomean GFLOPS per shape group for each ablation variant, and the ratio to the reference variant.
# usage: ablation.py <dir> <ref-label> label ...   (files <dir>/<label>_<row|col>.txt)
import math, os, sys
groups = [('M=64', range(1, 7)), ('M=128', range(7, 13)), ('M=4096', range(13, 19)), ('N=256', range(19, 25)),
          ('all 24', range(1, 25)), ('squares', [-s for s in (512, 1000, 1024, 2048, 3000, 4096)])]

def load(path):
    d = {}
    for line in open(path):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        d[int(f[0]) or -int(f[1])] = float(f[4])  # squares keyed by -M
    return d

def geo(v):
    return math.exp(sum(map(math.log, v)) / len(v))

d, ref, labels = sys.argv[1], sys.argv[2], sys.argv[3:]
for order in ('row', 'col'):
    base = load(os.path.join(d, '%s_%s.txt' % (ref, order)))
    print('\n%s-major (GFLOPS geomean; ratio to %s)\n' % (order, ref))
    print('| variant | ' + ' | '.join(g for g, _ in groups) + ' |')
    print('|---|' + '---|' * len(groups))
    for lab in [ref] + labels:
        p = os.path.join(d, '%s_%s.txt' % (lab, order))
        if not os.path.exists(p):
            continue
        r = load(p)
        cells = []
        for _, ids in groups:
            if not all(i in r and i in base for i in ids):
                cells.append('-')
                continue
            g, gb = geo([r[i] for i in ids]), geo([base[i] for i in ids])
            cells.append('%.0f (%.2f)' % (g, g / gb))
        print('| %s | %s |' % (lab, ' | '.join(cells)))
