#!/bin/bash
# Re-runs only the Eigen SME branch (pool and one thread) after its pin moves; the other libraries' runs stay.
cd "$(dirname "$0")/.."
for set in all small thin grid512 grid4096; do
  for o in col row; do
    BIN=./build/bench_eigen_br_mt bench/run_all.sh results/regular eigenbr_${set}_$o $set $o
    BIN=./build/bench_eigen_br bench/run_all.sh results/ext eigenbr_${set}_$o $set $o
  done
done
echo ALL DONE
