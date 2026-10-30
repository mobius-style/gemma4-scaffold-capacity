#!/usr/bin/env bash
# Boot-time resume for the Kaggle Gemma 4 collection (systemd user unit kaggle-gemma4-resume.service).
# 1) exit if the schedule is complete or a scheduler is already running
# 2) wait for Docker and both GPUs
# 3) archive every incomplete cell (PREREG §2: an infrastructure abort is re-run once from scratch) and log it
# 4) remove exited sandbox containers, notify Discord, exec the scheduler
# DRY_RUN=1 prints what it would do and changes nothing.
set -uo pipefail
W=/home/happy/.codex/work/kaggle_gemma4_paper_2026-10; cd "$W"
LOG=logs/schedule.log; N=$(grep -c . TASKSET.txt); TS=$(date +%Y%m%dT%H%M)
say() { python3 /home/happy/.codex/tools/coding_loop/discord_say.py "$1" >/dev/null 2>&1 || true; }
run() { if [ "${DRY_RUN:-0}" = 1 ]; then echo "DRY: $*"; else eval "$@"; fi; }

grep -q 'SCHEDULE_COMPLETE' "$LOG" && { echo "schedule complete; nothing to do"; exit 0; }
# Reboot-loop brake (added 2026-10-25 after 4 resets in 10 min, each 1-4 min after load start):
# if >=2 boots in the last 60 min, do NOT resume; notify and wait for the owner.
recent=$(journalctl --list-boots --no-pager 2>/dev/null | awk -v t="$(date -d '-60 min' '+%Y-%m-%d %H:%M:%S')" 'NR>1 {ts=$4" "$5; if (ts >= t) n++} END {print n+0}')
if [ "$recent" -ge 2 ] && [ "${FORCE:-0}" != 1 ]; then
  say "[kaggle] $recent boots in the last 60 min — reboot-loop brake engaged, collection NOT resumed. Owner action needed."
  echo "reboot-loop brake: $recent boots in 60 min"; exit 0
fi
if ps -eo cmd | grep -q '[s]chedule_v2.sh'; then echo "scheduler already running; exit"; exit 0; fi

for i in $(seq 1 60); do docker info >/dev/null 2>&1 && break; sleep 5; done
docker info >/dev/null 2>&1 || { say "[kaggle] reboot detected but Docker is not up after 5 min; collection NOT resumed"; exit 1; }
g=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
[ "$g" -ge 2 ] || { say "[kaggle] reboot detected but only $g GPU(s) visible; collection NOT resumed"; exit 1; }

archived=""
for d in runs/*_s[0-9]; do
  [ -d "$d" ] || continue
  n=0; for k in 0 1; do [ -f "$d/shard$k/task_results.jsonl" ] && n=$((n + $(grep -c . "$d/shard$k/task_results.jsonl"))); done
  if [ "$n" -lt "$N" ]; then
    b=$(basename "$d"); archived="$archived $b($n/$N)"
    run "mv '$d' 'runs/_aborted/${b}_reboot_${TS}'"
    run "echo \"\$(date -Iseconds) ABORT $b (host reboot; $n/$N done; re-run from scratch per PREREG §2)\" >> $LOG"
    run "printf '| D-auto | %s | Host reboot interrupted %s (%s/%s tasks). | Infrastructure abort. | Partial cell moved to runs/_aborted/%s_reboot_%s; re-run from scratch with the same seed (PREREG §2). | auto-applied by resume_after_boot.sh |\n' \"\$(date '+%F %H:%M JST')\" '$b' '$n' '$N' '$b' '$TS' >> DEVIATIONS_auto.md"
  fi
done
for c in $(docker ps -a --filter status=exited --format '{{.ID}} {{.Image}}' | awk '$2 ~ /^swebench-sandbox/ {print $1}'); do run "docker rm $c >/dev/null"; done

msg="[kaggle] host rebooted ($(uptime -s)); collection resuming. Archived partial cells:${archived:- none}"
echo "$msg"
[ "${DRY_RUN:-0}" = 1 ] && exit 0
say "$msg"
echo "- $(date '+%F %H:%M'): auto-resume after reboot; archived:${archived:- none}" >> PLAN.md
exec scripts/schedule_v3.sh "$N" >> logs/schedule_v2_stdout.log 2>&1
