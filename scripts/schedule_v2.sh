#!/usr/bin/env bash
# Collection schedule v2 (D3, 2026-10-23): confirmatory cells first (min/max × 3 models × 5 seeds, seed-major,
# model order 26B→12B→E4B), exploratory mid cells (seeds 1–3) only after all confirmatory cells are complete.
# Resumable; waits for any running evaluator to finish before starting. usage: scripts/schedule_v2.sh [N_TASKS]
set -uo pipefail
W=$(cd "$(dirname "$0")/.." && pwd); cd "$W"
N=${1:-$(grep -c . TASKSET.txt)}
MODELS=(26b 12b e4b); declare -A ALIAS=([26b]=gemma-4-26b-a4b-it [12b]=gemma-4-12b-it [e4b]=gemma-4-e4b-it)
LOG=logs/schedule.log
running() { ps -eo cmd | grep -c '[e]val_cli.py eval'; }
cell_done() { local d="runs/$1_${ALIAS[$2]}_s$3"; local n=0; for k in 0 1; do [ -f "$d/shard$k/task_results.jsonl" ] && n=$((n + $(grep -c . "$d/shard$k/task_results.jsonl"))); done; [ "$n" -ge "$N" ]; }
swap_servers() {
  pkill -x llama-server 2>/dev/null; sleep 3
  (nohup ./run_server.sh "$1" 0 8000 1 > "logs/server_$1_gpu0.log" 2>&1 &); (nohup ./run_server.sh "$1" 1 8001 1 > "logs/server_$1_gpu1.log" 2>&1 &)
  for i in $(seq 1 60); do a=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000/v1/models); b=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8001/v1/models); [ "$a" = 200 ] && [ "$b" = 200 ] && return 0; sleep 5; done
  echo "$(date -Iseconds) server start failed for $1" >> "$LOG"; return 1
}
run_cell() { # dose model seed
  echo "$(date -Iseconds) START $1 $2 s$3" >> "$LOG"
  MAX_TIME_MINUTES=6 MAX_TOOL_CALLS=50 MAX_TURNS=80 TIMEOUT_SECONDS=150 scripts/run_cell.sh "$1" "${ALIAS[$2]}" "$3" TASKSET.txt >> "$LOG" 2>&1
  echo "$(date -Iseconds) END $1 $2 s$3" >> "$LOG"
}
while [ "$(running)" -gt 0 ]; do sleep 60; done   # let a cell started by the previous scheduler finish
CUR=""
for seed in 1 2 3 4 5; do for m in "${MODELS[@]}"; do for d in min max; do
  cell_done "$d" "$m" "$seed" && continue
  [ "$CUR" = "$m" ] || { swap_servers "$m" || exit 1; CUR=$m; }
  run_cell "$d" "$m" "$seed"
done; done; done
echo "$(date -Iseconds) CONFIRMATORY_COMPLETE" >> "$LOG"
for seed in 1 2 3; do for m in "${MODELS[@]}"; do
  cell_done mid "$m" "$seed" && continue
  [ "$CUR" = "$m" ] || { swap_servers "$m" || exit 1; CUR=$m; }
  run_cell mid "$m" "$seed"
done; done
echo "$(date -Iseconds) SCHEDULE_COMPLETE" >> "$LOG"
