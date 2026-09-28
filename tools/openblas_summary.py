#!/usr/bin/env python3
# Geometric-mean GFLOPS of results/openblas: OpenBLAS develop, OpenBLAS with the SME2 port, MTGEMM-A, Accelerate.
import math, os, sys
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', sys.argv[1] if len(sys.argv) > 1 else 'openblas')

def load(f):
    d = {}
    for line in open(os.path.join(RES, f + '.txt')):
        p = line.split()
        if len(p) >= 5 and p[1].isdigit():
            d[(int(p[1]), int(p[2]), int(p[3]))] = max(float(p[4]), 0.5)  # sub-1 GFLOPS values print as 0
    return d

def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v))

SQ = [(s, s, s) for s in (512, 1000, 1024, 2048, 3000, 4096)]
print('| shapes | order | precision, threads | Accelerate | OpenBLAS develop | OpenBLAS + port | MTGEMM-A | port / develop | port / MTGEMM-A |')
print('|---|---|---|---|---|---|---|---|---|')
for st in ('all', 'small', 'thin'):
    for tag, suf in (('fp32, one thread', ('', '', '', '')), ('fp32, default', ('_t0', '_mt', '_mt', '_t0')),
                     ('fp64, one thread', ('_f64',) * 4)):
        for o in ('row', 'col'):
            ac, ob, ps, mt = (load('%s_%s_%s%s' % (lib, st, o, s)) for lib, s in zip(('accel', 'ob', 'obs2', 'mt'), suf))
            groups = [("paper's 24", [k for k in ob if k not in SQ]), ('squares 512-4096', SQ)] if st == 'all' else \
                     [({'small': 'squares 4-384', 'thin': 'thin (15)'}[st], list(ob))]
            for g, ks in groups:
                a, b, c, d = (gm([x[k] for k in ks]) for x in (ac, ob, ps, mt))
                print('| %s | %s | %s | %.0f | %.0f | **%.0f** | %.0f | %.2f | %.2f |' % (g, o, tag, a, b, c, d, c / b, c / d))
