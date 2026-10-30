#!/usr/bin/env bash
# Wait until a 26B server is up, let it run 10 min, then sample both GPUs at 100 ms for 120 s into logs/power/ (persistent).
W=$(cd "$(dirname "$0")/.." && pwd); cd "$W"; mkdir -p logs/power
until ps -eo cmd | grep -q '[l]lama-server -m .*26B'; do sleep 30; done
sleep 600
f=logs/power/26b_single_$(date +%m%d_%H%M).csv
timeout 120 nvidia-smi --query-gpu=index,power.draw --format=csv,noheader,nounits -lms 100 > "$f" 2>/dev/null
echo "saved $f ($(wc -l < "$f") rows)"
