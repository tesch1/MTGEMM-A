#!/usr/bin/env python3
# Geometric-mean GFLOPS of Accelerate, MTGEMM-A and Eigen (results/ext: one thread, results/regular: default threading).
import math, os
RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')

def load(f):
    d = {}
    for line in open(os.path.join(RES, f + '.txt')):
        p = line.split()
        if len(p) >= 5 and p[0].isdigit() and p[1].isdigit():
            d[(int(p[0]), int(p[1]), int(p[2]), int(p[3]))] = max(float(p[4]), 0.5)
    return d

def gm(v):
    return math.exp(sum(map(math.log, v)) / len(v))

GROUPS = [('M = 64 (IDs 1-6)', range(1, 7)), ('M = 128 (IDs 7-12)', range(7, 13)), ('M = 4096 (IDs 13-18)', range(13, 19)),
          ('N = 256 (IDs 19-24)', range(19, 25)), ('all 24', range(1, 25)), ('squares 512-4096', [0])]
FILES = {'default threading': ('regular/accel_all_%s', 'regular/mt_all_%s', 'regular/eigen_all_%s'),
         'one thread': ('ext/accel_%s', 'ext/mt_%s', 'ext/eigen_%s')}

for o in ('col', 'row'):
    print('\n%s-major' % o)
    print('| threading | workloads | Accelerate | MTGEMM-A | Eigen | Eigen / Accelerate | Eigen / MTGEMM-A | MTGEMM-A / Accelerate |')
    print('|---|---|---|---|---|---|---|---|')
    for th, files in FILES.items():
        ac, mt, ei = (load(f % o) for f in files)
        for g, ids in GROUPS:
            a, m, e = (gm([v for k, v in x.items() if k[0] in ids]) for x in (ac, mt, ei))
            print('| %s | %s | %.0f | %.0f | %.0f | %.2f | %.2f | %.2f |' % (th, g, a, m, e, e / a, e / m, m / a))

# Gain from the second SME unit over one thread, squares, column-major.
print('\n| size | Accelerate | MTGEMM-A | Eigen |\n|---|---|---|---|')
one = [load('ext/%s_col' % l) for l in ('accel', 'mt', 'eigen')] + [load('ext/%s_small_col' % l) for l in ('accel', 'mt', 'eigen')]
reg = [load('regular/%s_all_col' % l) for l in ('accel', 'mt', 'eigen')] + [load('regular/%s_small_col' % l) for l in ('accel', 'mt', 'eigen')]
for n in (256, 384, 512, 1024, 4096):
    r = []
    for i in range(3):
        k1 = [k for d in (one[i], one[i + 3]) for k in d if k[1:] == (n, n, n)]
        r.append(max((reg[i].get(k) or reg[i + 3].get(k)) for k in k1) / max((one[i].get(k) or one[i + 3].get(k)) for k in k1))
    print('| %d^3 | %.2f | %.2f | %.2f |' % ((n,) + tuple(r)))

# Grid cells below 0.9x of Accelerate.
for th, folder in (('one thread', 'ext'), ('default threading', 'regular')):
    for lib in ('mt', 'eigen'):
        slow = 0
        for o in ('col', 'row'):
            for kk in (512, 4096):
                a, b = load('%s/accel_grid%d_%s' % (folder, kk, o)), load('%s/%s_grid%d_%s' % (folder, lib, kk, o))
                slow += sum(1 for k in a if k in b and b[k] / a[k] < 0.9)
        print('%s, %s: %d of 484 grid cells below 0.9x Accelerate' % (th, lib, slow))
