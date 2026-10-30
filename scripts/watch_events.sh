#!/usr/bin/env bash
# Event-driven watcher: prints one line and exits on the first of: a cell END/ABORT line, scheduler gone,
# no evaluator running for 10 min while scheduler alive (hang), or a fatal signature in the newest shard logs.
# usage: scripts/watch_events.sh <max_seconds>
W=$(cd "$(dirname "$0")/.." && pwd); cd "$W"; MAX=${1:-18000}; t0=$(date +%s)
n0=$(grep -c -E ' (END|ABORT|SCHEDULE_COMPLETE|CONFIRMATORY_COMPLETE|server start failed)' logs/schedule.log); idle=0
while :; do
  n=$(grep -c -E ' (END|ABORT|SCHEDULE_COMPLETE|CONFIRMATORY_COMPLETE|server start failed)' logs/schedule.log)
  [ "$n" -gt "$n0" ] && { echo "EVENT: $(tail -1 logs/schedule.log)"; break; }
  s=$(ps -eo cmd | grep -c -E '[s]chedule_v[0-9]+.sh'); e=$(ps -eo cmd | grep -c '[e]val_cli.py eval')
  [ "$s" = 0 ] && { echo "EVENT: scheduler not running (evals=$e)"; break; }
  if [ "$e" = 0 ]; then idle=$((idle+60)); [ "$idle" -ge 600 ] && { echo "EVENT: no evaluator for 10 min while scheduler alive"; break; }; else idle=0; fi
  f=$(grep -l -E 'CUDA error|out of memory|Xid' $(ls -t runs/*_s[0-9]/shard*.log 2>/dev/null | head -2) 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "EVENT: fatal signature in $f"; grep -m2 -E 'CUDA error|out of memory|Xid' "$f"; break; }
  [ $(( $(date +%s) - t0 )) -ge "$MAX" ] && { echo "HEARTBEAT: no event in ${MAX}s; $(tail -1 logs/schedule.log)"; break; }
  sleep 60
done
python3 - <<'PY'
import json, glob
for cell in sorted(glob.glob('runs/*_s[0-9]')):
    rows=[]
    for f in glob.glob(f'{cell}/shard*/task_results.jsonl'): rows += [json.loads(l) for l in open(f)]
    if rows: print(cell.split('/')[-1], len(rows), 'tasks', 'resolved', sum(r['resolved'] for r in rows), 'cap', sum(1 for r in rows if (r.get('error') or '').startswith('Agent exceeded')), 'fatal', sum(1 for r in rows if (r.get('error') or '').startswith('Sandbox') or 'peg-gemma4' in (r.get('error') or '')))
PY
