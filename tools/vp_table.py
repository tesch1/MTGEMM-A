#!/usr/bin/env python3
"""Side-by-side GFLOPS table of the bench runs in a device batch directory (build/vp_results/out_<id>).
usage: vp_table.py <dir> [filter-substring]   Runs over the same shape set share one table; the test run is summarized."""
import os
import re
import sys
from collections import OrderedDict

d = sys.argv[1]
flt = sys.argv[2] if len(sys.argv) > 2 else ""
groups = OrderedDict()  # (set, order, precision) -> list of (label, {shape: gflops})
for f in sorted(os.listdir(d)):
    if not re.match(r"\d\d_", f):
        continue
    lines = open(os.path.join(d, f)).read().splitlines()
    if not lines:
        continue
    cmd = lines[0][2:]
    if not cmd.startswith("bench"):
        print(f"{f}: {cmd}: {' | '.join(l for l in lines[1:] if 'passed' in l or 'FAIL' in l)[:200]}")
        continue
    if flt not in cmd:
        continue
    w = cmd.split()
    key = (w[1], w[2], "f64" if "f64" in w else "f32")
    opts = [x for x in w[3:] if not re.match(r"(trials|ms|threads)=|f64$", x)]
    label = " ".join(opts)
    vals = OrderedDict()
    for l in lines[1:]:
        if l.startswith("#") or not l.strip():
            continue
        t = l.split()
        vals[(t[0], t[1], t[2], t[3])] = float(t[4])
    groups.setdefault(key, []).append((label, vals))

for key, runs in groups.items():
    print(f"\n## {' '.join(key)}")
    shapes = list(OrderedDict.fromkeys(s for _, v in runs for s in v))
    print("| shape | " + " | ".join(l for l, _ in runs) + " |")
    print("|---" * (len(runs) + 1) + "|")
    for s in shapes:
        name = (f"#{s[0]} " if s[0] != "0" else "") + "x".join(s[1:])
        print(f"| {name} | " + " | ".join(f"{v[s]:.0f}" if s in v else "-" for _, v in runs) + " |")
    import math
    gm = []
    for _, v in runs:
        xs = [max(v[s], 0.5) for s in shapes if s in v]
        gm.append(math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0)
    print("| **geomean** | " + " | ".join(f"**{g:.0f}**" for g in gm) + " |")
