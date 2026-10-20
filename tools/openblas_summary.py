#!/usr/bin/env python3
# OpenBLAS develop in results/openblas: level-3 routines against Accelerate (bench/run_openblas_l3.sh) and small GEMMs
# per call against MTGEMM-A (bench/run_openblas_small.sh). GEMM on all shape sets: tools/compare_summary.py.
import collections, os, sys
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', sys.argv[1] if len(sys.argv) > 1 else 'openblas')
NAMES = ['GEMM', 'SYMM', 'SYRK', 'SYR2K', 'TRMM', 'TRSM']

def load_l3(f):
    d = {}
    for line in open(os.path.join(RES, f + '.txt')):
        p = line.split()
        if len(p) == 8 and p[0].isdigit():
            d[(int(p[0]), p[1])] = [float(x) for x in p[2:]]
    return d

a1, o1, ad, od = (load_l3('l3_%s_%s' % (lib, s)) for s in ('1t', 'default') for lib in ('accel', 'ob'))
for n in (256, 1024, 4096):
    print('\n| n = %d | one thread: Accelerate | OpenBLAS | OpenBLAS / Accelerate | default: Accelerate | OpenBLAS | '
          'OpenBLAS / Accelerate |\n|---|---|---|---|---|---|---|' % n)
    for prec in ('f32', 'f64'):
        for i, nm in enumerate(NAMES):
            k = (n, prec)
            print('| %s %s | %.0f | %.0f | %.2f | %.0f | %.0f | %.2f |' % ('fp32' if prec == 'f32' else 'fp64', nm, a1[k][i],
                  o1[k][i], o1[k][i] / a1[k][i], ad[k][i], od[k][i], od[k][i] / ad[k][i]))

def load_v(f):
    d = collections.defaultdict(list)
    for line in open(os.path.join(RES, f + '.txt')):
        p = line.split()
        if len(p) >= 3 and p[0] in ('f32', 'f64'):
            d[(p[0], p[1])].append(float(p[-1]))
    return d

a, o = load_v('l3v_accel_1t'), load_v('l3v_ob_1t')
print('\nEvery variant at n = 1024, one thread (GFLOPS, min-max):')
for k in sorted(o):
    print('  %s %s: OpenBLAS %.0f-%.0f, Accelerate %.0f-%.0f' % (k + (min(o[k]), max(o[k]), min(a[k]), max(a[k]))))

def load_small(f):
    d = {}
    for line in open(os.path.join(RES, f + '.txt')):
        p = line.split()
        if len(p) >= 5 and p[0].isdigit():
            d[int(p[0])] = float(p[1])
    return d

print('\n| order, precision | MTGEMM-A time / OpenBLAS time | sizes where OpenBLAS is faster | sizes below 0.9x |\n|---|---|---|---|')
for o in ('row', 'col'):
    for p in ('f32', 'f64'):
        ob, mt = load_small('small_ob_%s_%s' % (o, p)), load_small('small_mt_%s_%s' % (o, p))
        r = {n: mt[n] / ob[n] for n in ob}
        slow = ', '.join('%d^3 (%.2fx)' % (n, v) for n, v in sorted(r.items()) if v < 0.9)
        print('| %s-major %s | %.2fx-%.2fx | %d of %d | %s |' % ('row' if o == 'row' else 'column', 'fp32' if p == 'f32' else 'fp64',
              min(r.values()), max(r.values()), sum(v > 1 for v in r.values()), len(r), slow or '-'))
