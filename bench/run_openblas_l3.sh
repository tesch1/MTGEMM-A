#!/bin/bash
# SYMM, SYRK, SYR2K, TRMM, TRSM (and GEMM for scale): OpenBLAS develop, the SME2 port and Accelerate in one session.
# usage: bench/run_openblas_l3.sh [outdir]   (default results/openblas)
set -e
cd "$(dirname "$0")/.."
O=${1:-results/openblas}
N="64 128 256 512 1024 2048 4096"
for t in 1 0; do
  s=$([ $t = 1 ] && echo 1t || echo default)
  BIN=./build/bench_openblas_l3 bench/run_all.sh $O l3_ob_$s $t $N
  BIN=./build/bench_openblas_sme2_l3 bench/run_all.sh $O l3_obs2_$s $t $N
  if [ $t = 1 ]; then BIN=./build/bench_accel_l3 bench/run_all.sh $O l3_accel_$s 1 $N
  else VECLIB_MAXIMUM_THREADS=default BIN=./build/bench_accel_l3 bench/run_all.sh $O l3_accel_$s 0 $N; fi
done
