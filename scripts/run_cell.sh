#!/usr/bin/env bash
# Run one experimental cell (dose x model x seed) sharded over two llama-servers (GPU0:8000, GPU1:8001).
# usage: run_cell.sh <min|mid|max> <alias e.g. gemma-4-e4b-it> <seed> [task_ids_file]
# Servers must already be up (run_server.sh <model> 0 8000 ; run_server.sh <model> 1 8001).
# Results: runs/<dose>_<alias>_s<seed>/shard{0,1}/ (summary.json, task_results.jsonl, patches/, traces/, logs/)
set -euo pipefail
W=$(cd "$(dirname "$0")/.." && pwd)
DOSE=$1; ALIAS=$2; SEED=$3; IDS_FILE=${4:-}
SUB=$("$W/subs/instantiate.sh" "$DOSE" "$ALIAS")
OUT="$W/runs/${DOSE}_${ALIAS}_s${SEED}"; mkdir -p "$OUT"
# seed goes into sampling config (forwarded to the server per request)
grep -q '^seed:' "$SUB/configs/sampling.yaml" || echo "seed: $SEED" >> "$SUB/configs/sampling.yaml"
cp -r "$SUB" "$OUT/submission_used"
EXTRA=()
[ -n "$IDS_FILE" ] && EXTRA=(--task-ids $(cat "$IDS_FILE"))
cd "$W/data_repaired"
export TMPDIR="$W/tmp_repaired2" KAGGLE_SANDBOX_DIR="$PWD"
# D8: NSHARDS=1 runs the whole cell on GPU0 (:8000) only; default 2 = GPU0/GPU1 shards
NSHARDS=${NSHARDS:-2}; SHARDS="0 1"; [ "$NSHARDS" = 1 ] && SHARDS="0"
for k in $SHARDS; do
  ( export LOCAL_INFERENCE_URL="http://127.0.0.1:800$k/v1"
    "$W/venv/bin/python" "$W/scripts/eval_cli.py" eval --tasks tasks.jsonl --snapshots-dir snapshots \
      --submission-dir "$SUB" --results-dir "$OUT/shard$k" --sandbox docker --image swebench-sandbox:repaired2 \
      --shard-index $k --num-shards $NSHARDS --max-tool-calls "${MAX_TOOL_CALLS:-50}" --max-turns "${MAX_TURNS:-80}" \
      --max-time-minutes "${MAX_TIME_MINUTES:-6}" --timeout-seconds "${TIMEOUT_SECONDS:-150}" \
      --display quiet "${EXTRA[@]}" > "$OUT/shard$k.log" 2>&1 ) &
done
wait
python3 - "$OUT" <<'EOF'
import json, sys, glob
out=sys.argv[1]; rows=[]
for f in sorted(glob.glob(f'{out}/shard*/task_results.jsonl')):
    rows += [json.loads(l) for l in open(f) if l.strip()]
res=sum(1 for r in rows if r.get('resolved'))
json.dump({'tasks':len(rows),'resolved':res,'rate':res/len(rows) if rows else None}, open(f'{out}/summary_merged.json','w'), indent=1)
print(f'{out}: resolved {res}/{len(rows)}')
EOF
