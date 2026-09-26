# AMX/NEON time ratio of the GEMV sweeps and scores of crossover rules in bytes: rows sizeof >= R, rows cols sizeof >= S.
# usage: tools/eigen_amx_gemv_crossover.py <neon.txt> <amx_all.txt> [<neon2.txt> <amx_all2.txt> ...]
import math, sys

def load(f):
    d = {}
    for l in open(f):
        if l.startswith("#") or not l.strip(): continue
        t = l.split()
        d[(int(t[0]), int(t[1]))] = float(t[2])
    return d

pairs = [(load(sys.argv[i]), load(sys.argv[i + 1]), sys.argv[i + 1]) for i in range(1, len(sys.argv), 2)]
size = {name: 8 if "f64" in name else 4 for _, _, name in pairs}
for neon, amx, name in pairs:
    rows = sorted({r for r, _ in neon}); cols = sorted({c for _, c in neon})
    print(f"\n{name}: NEON time / AMX time (>1: AMX faster), rows down, cols across")
    print("      " + "".join(f"{c:6d}" for c in cols))
    for r in rows:
        print(f"{r:6d}" + "".join(f"{neon[(r, c)] / amx[(r, c)]:6.2f}" if (r, c) in neon else "     -" for c in cols))
res = []
for R in (0, 256, 384, 512, 768, 1024, 2048):
    for S in (0, 8192, 16384, 32768, 65536, 131072):
        tot, worst = 0.0, 1.0
        n = 0
        for neon, amx, name in pairs:
            es = size[name]
            for k, tn in neon.items():
                use = k[0] * es >= R and k[0] * k[1] * es >= S
                chosen = amx[k] if use else tn
                best = min(tn, amx[k])
                tot += math.log(chosen / best); worst = min(worst, best / chosen); n += 1
        res.append((tot / n, worst, R, S))
print()
for l, w, R, S in sorted(res)[:8]:
    print(f"rows bytes >= {R:4d}, matrix bytes >= {S:6d}: mean loss {l:.3f}, worst {w:.2f}")
