#!/bin/bash
# Accelerate, MTGEMM-A and Eigen in one session, all shape sets and both orders: default threading into
# results/regular (Accelerate's own thread count, MTGEMM-A threads=0, Eigen on a pool of one thread per core) and one
# thread into results/ext (file names there omit the set name for "all").
# usage: bench/run_compare.sh [set ...]   (default: all small thin grid512 grid4096)
cd "$(dirname "$0")/.."
R=results/regular; E=results/ext
for set in ${@:-all small thin grid512 grid4096}; do
  for o in col row; do
    x=${set}_; [ $set = all ] && x=
    VECLIB_MAXIMUM_THREADS=default bench/run_all.sh $R accel_${set}_$o $set $o accel
    bench/run_all.sh $R mt_${set}_$o $set $o mt threads=0
    BIN=./build/bench_eigen_mt bench/run_all.sh $R eigen_${set}_$o $set $o
    bench/run_all.sh $E accel_$x$o $set $o accel
    bench/run_all.sh $E mt_$x$o $set $o mt threads=1
    BIN=./build/bench_eigen bench/run_all.sh $E eigen_$x$o $set $o
  done
done
echo ALL DONE
