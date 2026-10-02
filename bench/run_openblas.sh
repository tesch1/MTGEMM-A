#!/bin/bash
# OpenBLAS develop, OpenBLAS with the SME2 port (pull request #6074, third_party/build.sh), MTGEMM-A and Accelerate in one session.
# usage: bench/run_openblas.sh [outdir]   (default results/openblas)
set -e
cd "$(dirname "$0")/.."
O=${1:-results/openblas}
for set in all small thin; do
  for o in row col; do
    BIN=./build/bench_openblas bench/run_all.sh $O ob_${set}_$o $set $o threads=1
    BIN=./build/bench_openblas_sme2 bench/run_all.sh $O obs2_${set}_$o $set $o threads=1
    bench/run_all.sh $O mt_${set}_$o $set $o mt
    BIN=./build/bench_openblas bench/run_all.sh $O ob_${set}_${o}_mt $set $o threads=0
    BIN=./build/bench_openblas_sme2 bench/run_all.sh $O obs2_${set}_${o}_mt $set $o threads=0
    VECLIB_MAXIMUM_THREADS=default bench/run_all.sh $O mt_${set}_${o}_t0 $set $o mt threads=0
    BIN=./build/bench_openblas bench/run_all.sh $O ob_${set}_${o}_f64 $set $o f64 threads=1
    BIN=./build/bench_openblas_sme2 bench/run_all.sh $O obs2_${set}_${o}_f64 $set $o f64 threads=1
    bench/run_all.sh $O mt_${set}_${o}_f64 $set $o mt f64
    bench/run_all.sh $O accel_${set}_$o $set $o accel
    VECLIB_MAXIMUM_THREADS=default bench/run_all.sh $O accel_${set}_${o}_t0 $set $o accel
    bench/run_all.sh $O accel_${set}_${o}_f64 $set $o accel f64
  done
done
