#!/bin/bash
# Runs one benchmark binary at a time after the 1-minute load drops below 4; logs load before and after.
# usage: [BIN=build/bench_ext] bench/run_all.sh <outdir> <label> <bench args...>   (VECLIB_MAXIMUM_THREADS=default leaves it unset)
out=$1; label=$2; shift 2
mkdir -p "$out"
f="$out/$label.txt"
# Also waits while another benchmark binary (any bench* or ubench*) runs, for example from another checkout.
while [ "$(sysctl -n vm.loadavg | awk '{print int($2)}')" -ge 4 ] || ps -Ao comm | grep -qE '/u?bench[^/]*$'; do sleep 10; done
vt=${VECLIB_MAXIMUM_THREADS:-1}
bin=${BIN:-./build/bench}
{
  echo "# $(date '+%F %T') load before: $(sysctl -n vm.loadavg) VECLIB_MAXIMUM_THREADS=$vt"
  if [ "$vt" = default ]; then env -u VECLIB_MAXIMUM_THREADS "$bin" "$@" &
  else VECLIB_MAXIMUM_THREADS=$vt "$bin" "$@" & fi
  pid=$! other=0
  while kill -0 $pid 2>/dev/null; do  # samples every 2 s for another benchmark binary
    ps -Ao pid=,comm= | grep -E '/u?bench[^/]*$' | grep -qv "^ *$pid " && other=$((other + 1))
    sleep 2
  done
  wait $pid
  echo "# load after: $(sysctl -n vm.loadavg) other benchmarks seen: $other"
} > "$f" 2>&1
echo "$label done"
