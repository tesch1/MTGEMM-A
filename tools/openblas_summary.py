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
ONLY_L3 = len(sys.argv) > 3 and sys.argv[3] == 'l3'  # usage: openblas_summary.py [dir] [n] [l3]
if not ONLY_L3:
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

# Level-3 routines other than GEMM (bench/run_openblas_l3.sh): n = 256, 1024, 4096.
def load_l3(f):
    d = {}
    for line in open(os.path.join(RES, f + '.txt')):
        p = line.split()
        if len(p) == 8 and p[0].isdigit():
            d[(int(p[0]), p[1])] = [float(x) for x in p[2:]]
    return d

if os.path.exists(os.path.join(RES, 'l3_obs2_1t.txt')):
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    r = {(lib, s): load_l3('l3_%s_%s' % (lib, s)) for lib in ('accel', 'ob', 'obs2') for s in ('1t', 'default')}
    print()
    print('| n = %d | one thread: Accelerate | OpenBLAS develop | OpenBLAS + port | default: Accelerate | OpenBLAS develop | OpenBLAS + port |' % n)
    print('|---|---|---|---|---|---|---|')
    for prec in ('f32', 'f64'):
        for i, name in enumerate(('GEMM', 'SYMM', 'SYRK', 'SYR2K', 'TRMM', 'TRSM')):
            v = [r[(lib, s)][(n, prec)][i] for s in ('1t', 'default') for lib in ('accel', 'ob', 'obs2')]
            print('| %s %s | %.0f | %.0f | **%.0f** | %.0f | %.0f | **%.0f** |' % ('fp32' if prec == 'f32' else 'fp64', name, *v))
