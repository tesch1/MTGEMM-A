#!/bin/bash
# Default-threading comparison: Accelerate at its default thread count, MTGEMM-A in its automatic mode (threads=0),
# Eigen master and the Eigen SME branch on a thread pool of one thread per core; plus the Eigen branch on one thread.
cd "$(dirname "$0")/.."
R=results/regular; E=results/ext
for set in all small thin grid512 grid4096; do
  for o in col row; do
    VECLIB_MAXIMUM_THREADS=default bench/run_all.sh $R accel_${set}_$o $set $o accel
    bench/run_all.sh $R mt_${set}_$o $set $o mt threads=0
    BIN=./build/bench_eigen_br_mt bench/run_all.sh $R eigenbr_${set}_$o $set $o
    BIN=./build/bench_eigen_mt bench/run_all.sh $R eigen_${set}_$o $set $o
    BIN=./build/bench_eigen_br bench/run_all.sh $E eigenbr_${set}_$o $set $o
  done
done
echo ALL DONE
