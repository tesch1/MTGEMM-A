#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8"]
# ///
# Writes the SVG charts in docs/ (light and -dark variants) from the raw results in results/ext.
import math, os
import matplotlib
matplotlib.use('svg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
R = os.path.join(root, 'results', 'ext')
OUT = os.path.join(root, 'docs')

# Validated reference palette (categorical slots 1-4 by entity) and chart ink, light and dark.
THEMES = {
    '': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
             mid='#f0efec', blue='#184f95', red='#b23232',
             series={'Accelerate': '#2a78d6', 'MTGEMM-A': '#eb6834', 'Eigen': '#1baf7a', 'LIBXSMM': '#eda100', 'KleidiAI': '#eda100'}),
    '-dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                  mid='#383835', blue='#3987e5', red='#e66767',
                  series={'Accelerate': '#3987e5', 'MTGEMM-A': '#d95926', 'Eigen': '#199e70', 'LIBXSMM': '#c98500', 'KleidiAI': '#c98500'}),
}
MARKERS = {'Accelerate': 'o', 'MTGEMM-A': 's', 'Eigen': '^', 'LIBXSMM': 'D', 'KleidiAI': 'D'}

def load(name):
    d = {}
    for line in open(os.path.join(R, name + '.txt')):
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        d[(int(f[1]), int(f[2]), int(f[3]))] = float(f[4])
    return d

def merged(*names):
    d = {}
    for n in names:
        d.update(load(n))
    return d

def setup(t):
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'sans-serif',
                         'font.sans-serif': ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 10,
                         'text.color': t['ink2'], 'axes.labelcolor': t['muted'], 'xtick.color': t['muted'],
                         'ytick.color': t['muted'], 'axes.edgecolor': t['axis'], 'figure.facecolor': t['surface'],
                         'axes.facecolor': t['surface'], 'savefig.facecolor': t['surface'], 'axes.titlecolor': t['ink'],
                         'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left'})

def style_axes(ax, t):
    ax.grid(True, axis='y', color=t['grid'], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)

def lines(ax, t, series, xlabel, ticks, ymax):
    ends = []
    for label, pts in series:
        xs = sorted(pts)
        c = t['series'][label]
        ax.plot(xs, [pts[x] for x in xs], color=c, linewidth=2, marker=MARKERS[label], markersize=5,
                markeredgecolor=t['surface'], markeredgewidth=1, label=label)
        ends.append([pts[xs[-1]], label, xs[-1]])
    ax.set_xscale('log', base=2)
    ax.set_xticks(ticks, [str(x) for x in ticks])
    ax.minorticks_off()
    ax.set_ylim(0, ymax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('GFLOPS')
    ends.sort()
    gap = ymax * 0.055
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, label, x in ends:
        ax.annotate(label, (x, y), xytext=(8, 0), textcoords='offset points', va='center', color=t['ink2'], fontsize=9,
                    annotation_clip=False)
    style_axes(ax, t)

def ratios(ax, t, series, xlabel, ticks):
    """series[0] is Accelerate; plots every other series divided by it, log2 y."""
    base = series[0][1]
    ends = []
    ax.axhline(1, color=t['axis'], linewidth=1.2)
    for label, pts in series[1:]:
        xs = sorted(x for x in pts if x in base)
        c = t['series'][label]
        ys = [pts[x] / base[x] for x in xs]
        ax.plot(xs, ys, color=c, linewidth=2, marker=MARKERS[label], markersize=5, markeredgecolor=t['surface'],
                markeredgewidth=1, label=label)
        ends.append([math.log2(ys[-1]), label, xs[-1], ys[-1]])
    ax.set_xscale('log', base=2)
    ax.set_yscale('log', base=2)
    ax.set_ylim(1 / 8, 4)
    yt = [0.125, 0.25, 0.5, 1, 2, 4]
    ax.set_yticks(yt, ['%gx' % v for v in yt])
    ax.set_xticks(ticks, [str(x) for x in ticks])
    ax.minorticks_off()
    ax.set_xlabel(xlabel)
    ax.set_ylabel('speedup over Accelerate')
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.35)
    for ly, label, x, y in ends:
        ax.annotate(label, (x, 2 ** ly), xytext=(8, 0), textcoords='offset points', va='center', color=t['ink2'],
                    fontsize=9, annotation_clip=False)
    style_axes(ax, t)

def sq(d):
    return {m: v for (m, n, k), v in d.items() if m == n == k}

def squares_chart(t, suffix):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), sharey='row')
    for col, order, other, lib in ((0, 'col', 'LIBXSMM', 'libxsmm'), (1, 'row', 'KleidiAI', 'kleidiai')):
        ax = axs[0][col]
        s = [('Accelerate', sq(merged('accel_' + order, 'accel_small_' + order))),
             ('MTGEMM-A', sq(merged('mt_' + order, 'mt_small_' + order))),
             ('Eigen', sq(merged('eigen_' + order, 'eigen_small_' + order))),
             (other, sq(merged('%s_%s' % (lib, order), '%s_small_%s' % (lib, order))))]
        ticks = [4, 16, 64, 256, 1024, 4096]
        lines(ax, t, s, 'M = N = K', ticks, 2000)
        ax.set_title('Square sizes, %s' % ('column-major C += AB' if order == 'col' else 'row-major C = AB'))
        ax.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=t['ink2'])
        ratios(axs[1][col], t, s, 'M = N = K', ticks)
        axs[1][col].set_title('Speedup over Accelerate (log scale)')
    fig.tight_layout(w_pad=6, h_pad=2)
    fig.savefig(os.path.join(OUT, 'squares%s.svg' % suffix))
    plt.close(fig)

def thin_chart(t, suffix):
    fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), sharey='row')
    data = [('Accelerate', load('accel_thin_col')), ('MTGEMM-A', load('mt_thin_col')), ('Eigen', load('eigen_thin_col')),
            ('LIBXSMM', load('libxsmm_thin_col'))]
    for col, which in ((0, 'M'), (1, 'N')):
        ax = axs[0][col]
        s = []
        for label, d in data:
            if which == 'M':
                s.append((label, {m: v for (m, n, k), v in d.items() if n == 4096 and k == 4096 and m <= 64}))
            else:
                s.append((label, {n: v for (m, n, k), v in d.items() if m == 4096 and k == 4096 and n <= 64}))
        lines(ax, t, s, which, [1, 2, 4, 8, 16, 32, 64], 1250)
        other = 'N = K = 4096' if which == 'M' else 'M = K = 4096'
        ax.set_title('Thin: %s from 1 to 64, %s, column-major' % (which, other))
        ax.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=t['ink2'])
        ratios(axs[1][col], t, s, which, [1, 2, 4, 8, 16, 32, 64])
        axs[1][col].set_title('Speedup over Accelerate (log scale)')
    fig.tight_layout(w_pad=6, h_pad=2)
    fig.savefig(os.path.join(OUT, 'thin%s.svg' % suffix))
    plt.close(fig)

def heat(ax, t, num, den, title):
    sizes = [4 << i for i in range(11)]
    cmap = LinearSegmentedColormap.from_list('div', [t['red'], t['mid'], t['blue']])
    norm = LogNorm(vmin=0.25, vmax=4)
    grid = [[None] * 11 for _ in sizes]
    for (m, n, k), v in num.items():
        if (m, n, k) in den and m in sizes and n in sizes:
            grid[sizes.index(m)][sizes.index(n)] = v / den[(m, n, k)]
    vals = [[(g if g else float('nan')) for g in row] for row in grid]
    im = ax.imshow([[min(4, max(0.25, v)) if v == v else v for v in row] for row in vals], cmap=cmap, norm=norm,
                   origin='upper', aspect='equal')
    for i in range(11):
        for j in range(11):
            v = vals[i][j]
            if v != v:
                continue
            rgba = cmap(norm(min(4, max(0.25, v))))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, ('%.1f' % v) if v < 9.95 else '%.0f' % v, ha='center', va='center', fontsize=6.5,
                    color='#ffffff' if lum < 0.45 else '#0b0b0b')
    ax.set_xticks(range(11), [str(s) for s in sizes], rotation=90, fontsize=8)
    ax.set_yticks(range(11), [str(s) for s in sizes], fontsize=8)
    ax.set_xlabel('N')
    ax.set_ylabel('M')
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im

def grid_chart(t, suffix, name, lib, label, orders):
    panels = [(o, k) for o in orders for k in (512, 4096)]
    rows = (len(panels) + 1) // 2
    fig, axs = plt.subplots(rows, 2, figsize=(10, 4.9 * rows + 0.6), squeeze=False)
    for p, (o, k) in enumerate(panels):
        ax = axs[p // 2][p % 2]
        im = heat(ax, t, load('%s_grid%d_%s' % (lib, k, o)), load('accel_grid%d_%s' % (k, o)),
                  '%s / Accelerate, K = %d, %s' % (label, k, 'column-major' if o == 'col' else 'row-major'))
    fig.subplots_adjust(hspace=0.32, wspace=0.28)
    cb = fig.colorbar(im, ax=axs.ravel().tolist(), orientation='horizontal', fraction=0.035 / rows, pad=0.09 / rows,
                      aspect=40)
    cb.ax.minorticks_off()
    cb.set_ticks([0.25, 0.5, 1, 2, 4], labels=['0.25x', '0.5x', '1x', '2x', '4x'])
    cb.set_label('speedup over Accelerate (blue: faster, red: slower; clipped at 0.25x and 4x)', color=t['muted'])
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(OUT, '%s%s.svg' % (name, suffix)), bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for suffix, t in THEMES.items():
        setup(t)
        squares_chart(t, suffix)
        thin_chart(t, suffix)
        if os.path.exists(os.path.join(R, 'eigen_grid4096_col.txt')):
            grid_chart(t, suffix, 'grid_mtgemm', 'mt', 'MTGEMM-A', ('col', 'row'))
            grid_chart(t, suffix, 'grid_eigen', 'eigen', 'Eigen', ('col',))
    print('wrote', sorted(os.listdir(OUT)))
