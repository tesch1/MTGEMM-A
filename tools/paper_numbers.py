#!/usr/bin/env python3
# Reads the paper's bar charts (vector PDFs) and writes results/paper_numbers.csv.
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from pdfbars import table
FIG = os.path.join(os.path.dirname(__file__), '..', 'refs', 'mpgemm-paper', 'figs')
series = {
    'singlerowmajor': ['OpenBLAS', 'KleidiAI', 'Accelerate', 'MpGEMM'],
    'singlecolmajor': ['LIBXSMM', 'Accelerate', 'MpGEMM'],
    'multirowmajor': ['Accelerate', 'MpGEMM'],
    'multicolmajor': ['Accelerate', 'MpGEMM'],
    'singlerowmajor-fp64': ['Accelerate-Serial', 'MpGEMM-Serial', 'Accelerate-Parallel', 'MpGEMM-Parallel'],
}
out = open(os.path.join(os.path.dirname(__file__), '..', 'results', 'paper_numbers.csv'), 'w')
out.write('figure,series,id,gflops\n')
for fig, names in series.items():
    rows = table(os.path.join(FIG, fig + '.pdf'))
    for i, r in enumerate(rows):
        ident = 'geo' if i == 24 else str(i + 1)
        for name, v in zip(names, r):
            out.write('%s,%s,%s,%.0f\n' % (fig, name, ident, v))
out.close()
