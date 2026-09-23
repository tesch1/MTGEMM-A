#!/usr/bin/env python3
# Extract bar heights from a matplotlib bar-chart PDF: prints one row per x group.
import re, sys, zlib
def streams(path):
    d = open(path, 'rb').read()
    for m in re.finditer(rb'stream\r?\n(.*?)endstream', d, re.S):
        try: yield zlib.decompress(m.group(1)).decode('latin1')
        except Exception: pass
def parse(path):
    txt = max(streams(path), key=len)
    toks = txt.split()
    color = None; pts = []; bars = []; yt = []; labels = []; cm = None
    i = 0
    while i < len(toks):
        t = toks[i]
        if t == 'rg': color = tuple(toks[i-3:i])
        elif t == 'm': pts = [(float(toks[i-2]), float(toks[i-1]))]
        elif t == 'l': pts.append((float(toks[i-2]), float(toks[i-1])))
        elif t in ('f', 'B', 'b') and len(pts) == 4:
            xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
            w = max(xs) - min(xs); h = max(ys) - min(ys)
            if 3 < w < 40 and h > 0: bars.append((min(xs), min(ys), max(ys), color))
        elif t in ('B', 'S') and len(pts) == 2:
            (x0, y0), (x1, y1) = pts
            if abs(y0 - y1) < 1e-6 and 2 < abs(x0 - x1) < 6: yt.append(y0)
        elif t == 'cm': cm = (float(toks[i-2]), float(toks[i-1]))
        elif t == 'TJ':
            s = ''.join(re.findall(r'\((.*?)\)', ' '.join(toks[max(0,i-8):i])))
            labels.append((cm, s))
        i += 1
    return bars, sorted(set(yt)), labels
def table(path):
    bars, yt, labels = parse(path)
    y0 = yt[0]; scale = None
    lab = {}
    for (c, s) in labels:
        if c and s.isdigit(): lab[round(c[1], 0)] = int(s)
    ticks = []
    for y in yt:
        for ly, v in lab.items():
            if abs(ly + 6.8 - y) < 3: ticks.append((y, v))
    (ya, va), (yb, vb) = ticks[0], ticks[-1]
    k = (vb - va) / (yb - ya)
    base = [b for b in bars if abs(b[1] - ya) < 0.5]
    legend = sorted([b for b in bars if abs(b[1] - ya) >= 0.5], key=lambda b: b[0])
    order = [b[3] for b in legend]
    base.sort(key=lambda b: b[0])
    ns = len(order); rows = []
    for g in range(0, len(base), ns):
        grp = {b[3]: va + (b[2] - ya) * k for b in base[g:g+ns]}
        rows.append([grp.get(c, float('nan')) for c in order])
    return rows
if __name__ == '__main__':
    import sys
    for r in table(sys.argv[1]): print(' '.join('%6.0f' % v for v in r))
