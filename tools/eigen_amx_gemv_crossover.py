# AMX/NEON time ratio of GEMV sweeps and scores of crossover rules in bytes:
# rows sizeof >= R, S <= rows cols sizeof < X (X: no upper bound when 0).
# usage: tools/eigen_amx_gemv_crossover.py [--quiet] <neon.txt> <amx_all.txt> [<neon2.txt> <amx_all2.txt> ...]
# A file whose name contains f64 holds double results.
import math, sys

args = sys.argv[1:]
quiet = args and args[0] == "--quiet"
if quiet: args = args[1:]

def load(f):
    d = {}
    for l in open(f):
        if l.startswith("#") or not l.strip(): continue
        t = l.split()
        d[(int(t[0]), int(t[1]))] = float(t[2])
    return d

pairs = [(load(args[i]), load(args[i + 1]), args[i + 1]) for i in range(0, len(args), 2)]
size = {name: 8 if "f64" in name else 4 for _, _, name in pairs}
if not quiet:
    for neon, amx, name in pairs:
        rows = sorted({r for r, _ in neon}); cols = sorted({c for _, c in neon})
        print(f"\n{name}: NEON time / AMX time (>1: AMX faster), rows down, cols across")
        print("      " + "".join(f"{c:6d}" for c in cols))
        for r in rows:
            print(f"{r:6d}" + "".join(f"{neon[(r, c)] / amx[(r, c)]:6.2f}" if (r, c) in neon else "     -" for c in cols))
res = []
for R in (0, 256, 384, 512, 768, 1024, 2048):
    for S in (0, 16384, 32768, 65536, 131072, 262144):
        for X in (0, 32 << 20, 48 << 20, 64 << 20):
            tot, worst, n = 0.0, 1.0, 0
            for neon, amx, name in pairs:
                es = size[name]
                for k, tn in neon.items():
                    b = k[0] * k[1] * es
                    use = k[0] * es >= R and b >= S and (X == 0 or b < X)
                    chosen = amx[k] if use else tn
                    best = min(tn, amx[k])
                    tot += math.log(chosen / best); worst = min(worst, best / chosen); n += 1
            res.append((tot / n, worst, R, S, X))
print()
for l, w, R, S, X in sorted(res)[:8]:
    print(f"rows bytes >= {R:4d}, {S:6d} <= matrix bytes < {X >> 20 if X else 'inf'} MB: mean loss {l:.3f}, worst {w:.2f}")
