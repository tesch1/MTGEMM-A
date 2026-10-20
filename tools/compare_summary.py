#!/usr/bin/env python3
# Geometric-mean GFLOPS of Accelerate, MTGEMM-A, Eigen and OpenBLAS (results/ext: one thread, results/regular: default
# threading; bench/run_compare.sh), the gain from the second SME unit, and grid cells below 0.9x of Accelerate.
import math, os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
LIBS = [('accel', 'Accelerate'), ('mt', 'MTGEMM-A'), ('eigen', 'Eigen'), ('openblas', 'OpenBLAS')]

def load(f):
    d = {}
    for line in open(os.path.join(RES, f + '.txt')):
        p = line.split()
        if len(p) >= 5 and p[0].isdigit() and p[1].isdigit():
            d[(int(p[0]), int(p[1]), int(p[2]), int(p[3]))] = max(float(p[4]), 0.5)
    return d

def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v))

def name(lib, folder, st, o, s):
    if folder == 'ext':
        return 'ext/%s_%s%s%s' % (lib, '' if st == 'all' else st + '_', o, s)
    return 'regular/%s_%s_%s%s' % (lib, st, o, s)

GROUPS = [('M = 64 (IDs 1-6)', 'all', range(1, 7)), ('M = 128 (IDs 7-12)', 'all', range(7, 13)),
          ('M = 4096 (IDs 13-18)', 'all', range(13, 19)), ('N = 256 (IDs 19-24)', 'all', range(19, 25)),
          ("paper's 24", 'all', range(1, 25)), ('squares 512-4096', 'all', [0]), ('squares 4-384', 'small', None),
          ('thin (15)', 'thin', None)]

for s, prec in (('', 'fp32'), ('_f64', 'fp64')):
    for o in ('col', 'row'):
        print('\n%s, %s-major' % (prec, o))
        print('| threading | workloads | ' + ' | '.join(l for _, l in LIBS) + ' | MTGEMM-A / Accelerate | Eigen / Accelerate'
              ' | OpenBLAS / Accelerate | Eigen / MTGEMM-A | OpenBLAS / MTGEMM-A |')
        print('|---' * 11 + '|')
        for th, folder in (('default', 'regular'), ('one thread', 'ext')):
            for g, st, ids in GROUPS:
                d = [load(name(lib, folder, st, o, s)) for lib, _ in LIBS]
                v = [gm([x for k, x in di.items() if ids is None or k[0] in ids]) for di in d]
                a, m, e, b = v
                print('| %s | %s | %.0f | %.0f | %.0f | %.0f | %.2f | %.2f | %.2f | %.2f | %.2f |'
                      % (th, g, a, m, e, b, m / a, e / a, b / a, e / m, b / m))

# Gain from the second SME unit over one thread, squares, column-major fp32.
print('\n| size | ' + ' | '.join(l for _, l in LIBS) + ' |\n|---' * 1 + '|---' * 4 + '|')
for n in (256, 384, 512, 1024, 4096):
    r = []
    for lib, _ in LIBS:
        one = {**load(name(lib, 'ext', 'all', 'col', '')), **load(name(lib, 'ext', 'small', 'col', ''))}
        reg = {**load(name(lib, 'regular', 'all', 'col', '')), **load(name(lib, 'regular', 'small', 'col', ''))}
        k = [k for k in one if k[1:] == (n, n, n)][0]
        r.append(reg[k] / one[k])
    print('| %d^3 | ' % n + ' | '.join('%.2f' % x for x in r) + ' |')

# Grid cells below 0.9x of Accelerate (484 per threading: 2 orders x 2 K x 121).
print()
for th, folder in (('one thread', 'ext'), ('default threading', 'regular')):
    for lib, label in LIBS[1:]:
        slow = 0
        for o in ('col', 'row'):
            for kk in (512, 4096):
                a, b = load('%s/accel_grid%d_%s' % (folder, kk, o)), load('%s/%s_grid%d_%s' % (folder, lib, kk, o))
                slow += sum(1 for k in a if k in b and b[k] / a[k] < 0.9)
        print('%s, %s: %d of 484 grid cells below 0.9x Accelerate' % (th, label, slow))
