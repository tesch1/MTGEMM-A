#!/bin/bash
# Per-call times of small square GEMMs (bench_small: minimum over many short batches, three passes): OpenBLAS
# develop, the SME2 port and MTGEMM-A, one thread, both orders and precisions.
# usage: bench/run_openblas_small.sh [outdir]   (default results/openblas)
set -e
cd "$(dirname "$0")/.."
O=${1:-results/openblas}
N="4 8 12 14 15 16 17 18 20 24 28 32 36 40 44 $(seq 48 4 192 | tr '\n' ' ') 224 256"
for o in row col; do
  for p in f32 f64; do
    BIN=./build/bench_small_ob bench/run_all.sh $O small_ob_${o}_$p $o $p 1 $N
    BIN=./build/bench_small_obs2 bench/run_all.sh $O small_obs2_${o}_$p $o $p 1 $N
    BIN=./build/bench_small_mt bench/run_all.sh $O small_mt_${o}_$p $o $p 1 $N
  done
done
