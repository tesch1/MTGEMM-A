#!/bin/bash
# Accelerate, MTGEMM-A, Eigen master and OpenBLAS develop in one session, both orders. Default threading goes into
# results/regular (Accelerate's own thread count, MTGEMM-A threads=0, Eigen on a pool of one thread per core,
# OpenBLAS with its pool), one thread into results/ext (file names there omit the set name for "all").
# fp32 on every set; fp64 (suffix _f64) on all, small and thin.
# usage: bench/run_compare.sh [set ...]   (default: all small thin grid512 grid4096)
cd "$(dirname "$0")/.."
R=results/regular; E=results/ext
for set in ${@:-all small thin grid512 grid4096}; do
  precs="f32"; case $set in all|small|thin) precs="f32 f64";; esac
  for p in $precs; do
    f=; s=; [ $p = f64 ] && { f=f64; s=_f64; }
    for o in col row; do
      x=${set}_; [ $set = all ] && x=
      VECLIB_MAXIMUM_THREADS=default bench/run_all.sh $R accel_${set}_$o$s $set $o accel $f
      bench/run_all.sh $R mt_${set}_$o$s $set $o mt $f threads=0
      BIN=./build/bench_eigen_mt bench/run_all.sh $R eigen_${set}_$o$s $set $o $f
      BIN=./build/bench_openblas bench/run_all.sh $R openblas_${set}_$o$s $set $o $f threads=0
      bench/run_all.sh $E accel_$x$o$s $set $o accel $f
      bench/run_all.sh $E mt_$x$o$s $set $o mt $f threads=1
      BIN=./build/bench_eigen bench/run_all.sh $E eigen_$x$o$s $set $o $f
      BIN=./build/bench_openblas bench/run_all.sh $E openblas_$x$o$s $set $o $f threads=1
    done
  done
done
echo ALL DONE
