#!/bin/bash
# Runs one benchmark binary at a time after the 1-minute load drops below 4; logs load before and after.
# usage: bench/run_all.sh <outdir> <label> <bench args...>
out=$1; label=$2; shift 2
mkdir -p "$out"
f="$out/$label.txt"
while [ "$(sysctl -n vm.loadavg | awk '{print int($2)}')" -ge 4 ]; do sleep 10; done
{
  echo "# $(date '+%F %T') load before: $(sysctl -n vm.loadavg)"
  VECLIB_MAXIMUM_THREADS=${VECLIB_MAXIMUM_THREADS:-1} ./build/bench "$@"
  echo "# load after: $(sysctl -n vm.loadavg)"
} > "$f" 2>&1
echo "$label done"
