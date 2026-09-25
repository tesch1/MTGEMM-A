#!/bin/bash
# SME against AMX on one M4 (build/bench_multi, backend=sme|amx), with Accelerate; usage: bench/run_sme_amx.sh <outdir>
out=${1:-results/sme_vs_amx}
export BIN=./build/bench_multi
R=bench/run_all.sh
for s in squares paper small thin irr; do
  for o in row col; do
    $R "$out/1t" "${s}_${o}_accel" $s $o accel
    $R "$out/1t" "${s}_${o}_sme" $s $o mt backend=sme threads=1
    $R "$out/1t" "${s}_${o}_amx" $s $o mt backend=amx threads=1
  done
done
for s in squares paper; do
  $R "$out/1t" "${s}_row_accel_f64" $s row accel f64
  $R "$out/1t" "${s}_row_sme_f64" $s row mt f64 backend=sme threads=1
  $R "$out/1t" "${s}_row_amx_f64" $s row mt f64 backend=amx threads=1
  for o in row col; do
    VECLIB_MAXIMUM_THREADS=default $R "$out/default" "${s}_${o}_accel" $s $o accel
    $R "$out/default" "${s}_${o}_sme" $s $o mt backend=sme threads=0
    $R "$out/default" "${s}_${o}_amx" $s $o mt backend=amx threads=1
    $R "$out/default" "${s}_${o}_amx2" $s $o mt backend=amx threads=2
  done
done
