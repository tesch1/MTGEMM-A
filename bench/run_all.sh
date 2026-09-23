#!/bin/bash
# Runs the benchmark matrix one binary at a time; logs load averages before and after each run.
# usage: bench/run_all.sh <outdir> <label> <bench args...>
out=$1; label=$2; shift 2
mkdir -p "$out"
f="$out/$label.txt"
{
  echo "# $(date '+%F %T') load before: $(uptime | sed 's/.*averages*: //')"
  VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1} ./build/bench "$@"
  echo "# load after: $(uptime | sed 's/.*averages*: //')"
} > "$f" 2>&1
echo "$label done"
