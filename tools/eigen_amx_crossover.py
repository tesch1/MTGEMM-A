# Scores NEON/AMX crossover rules (rows cols >= A, rows cols depth sizeof >= W) on the cross sweeps of results/eigen_amx.
import math, os
R = os.path.join(os.path.dirname(__file__), "..", "results", "eigen_amx")
S = os.path.join(R, "m4pro") + "/"
V = os.path.join(R, "vp_m2_batch2") + "/"
def load(f):
    d = {}
    for l in open(f):
        if l.startswith("#") or not l.strip(): continue
        t = l.split()
        d[(int(t[1]), int(t[2]), int(t[3]))] = float(t[4])
    return d
sets = {"M4 f32": (S + "cross_neon.txt", S + "cross_amx.txt", 4), "M4 f64": (S + "cross_neon64.txt", S + "cross_amx64.txt", 8),
        "M2 f32": (V + "04_eigen_neon.txt", V + "05_eigen_amx_all.txt", 4), "M2 f64": (V + "06_eigen_neon.txt", V + "07_eigen_amx_all.txt", 8)}
data = {k: (load(a), load(b), es) for k, (a, b, es) in sets.items()}
res = []
for A in (192, 256, 320, 384, 512, 768):
    for W in (2**15, 2**16, 2**17, 3 * 2**16, 2**18, 3 * 2**17, 2**19):
        per = {}
        for name, (neon, amx, es) in data.items():
            loss, worst = 0, 1
            for (m, n, k), x in neon.items():
                use = m * n >= A and m * n * k * es >= W
                c = amx[(m, n, k)] if use else x
                top = max(x, amx[(m, n, k)])
                loss += math.log(top / c); worst = min(worst, c / top)
            per[name] = (loss / len(neon), worst)
        res.append((sum(v[0] for v in per.values()) / 4, A, W, per))
for tot, A, W, per in sorted(res, key=lambda r: r[0])[:8]:
    print(f"A={A:4d} W=2^{math.log2(W):.2f}  mean {tot:.4f}  " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.2f}" for k, v in per.items()))
for tot, A, W, per in res:
    if A == 256 and W == 2**17:
        print(f"current A=256 W=2^17 mean {tot:.4f}  " + "  ".join(f"{k} {v[0]:.3f}/{v[1]:.2f}" for k, v in per.items()))
