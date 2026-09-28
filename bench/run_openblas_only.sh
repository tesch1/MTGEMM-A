#!/bin/bash
# Re-runs only the two OpenBLAS builds of bench/run_openblas.sh (same files, same options).
set -e
cd "$(dirname "$0")/.."
O=${1:-results/openblas}
for set in all small thin; do
  for o in row col; do
    for v in "ob ./build/bench_openblas" "obs2 ./build/bench_openblas_sme2"; do
      set -- $v
      BIN=$2 bench/run_all.sh $O $1_${set}_$o $set $o threads=1
      BIN=$2 bench/run_all.sh $O $1_${set}_${o}_mt $set $o threads=0
      BIN=$2 bench/run_all.sh $O $1_${set}_${o}_f64 $set $o f64 threads=1
    done
  done
done
