#!/usr/bin/env python3
"""Scorable-set check using the REAL scoring path (swegemma.harness.verification.verify_task):
for each task, verify (a) gold patch -> must be resolved, (b) empty patch -> must NOT be resolved.

Run from data_repaired/ with TMPDIR/KAGGLE_SANDBOX_DIR exported:
  ../venv/bin/python ../scripts/gold_verify.py --out ../logs/gold_verify.jsonl [--ids ...] [--image swebench-sandbox:repaired2]
Appends one JSON line per task; resumable.
"""
import argparse, asyncio, json, time
from pathlib import Path

from adk_submission import ModelRegistry
from swegemma.config import EvalConfig
from swegemma.evaluate import Evaluator
from swegemma.harness.verification import verify_task
from swegemma.models.task import Task


def load_tasks(path):
    for line in open(path):
        if line.strip():
            d = json.loads(line)
            yield Task(instance_id=d['instance_id'], repo=d['repo'], base_commit=d['base_commit'],
                       patch=d.get('patch', ''), test_patch=d.get('test_patch', ''),
                       problem_statement=d.get('problem_statement', ''), hints_text=d.get('hints_text', ''),
                       created_at=d.get('created_at', ''))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='tasks.jsonl')
    ap.add_argument('--snapshots', default='snapshots')
    ap.add_argument('--image', default='swebench-sandbox:repaired2')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ids', nargs='*')
    ap.add_argument('--submission-dir', default='../subs/min')
    a = ap.parse_args()
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)['instance_id'] for l in open(out)} if out.exists() else set()
    config = EvalConfig(tasks_path=Path(a.tasks), snapshots_dir=Path(a.snapshots), results_dir=Path('../runs/_gold_verify_tmp'),
                        submission_dir=Path(a.submission_dir), models=ModelRegistry(), image=a.image, sandbox='docker',
                        display_mode='quiet')
    ev = Evaluator(config)
    tasks = [t for t in load_tasks(a.tasks) if (not a.ids or t.instance_id in a.ids) and t.instance_id not in done]
    print(f'{len(tasks)} tasks ({len(done)} done)', flush=True)
    for t in tasks:
        snap = Path(a.snapshots) / f'{t.instance_id}.tgz'
        rec = {'instance_id': t.instance_id, 'repo': t.repo}
        t0 = time.time()
        for label, patch in (('gold', t.patch), ('empty', '')):
            try:
                r = await verify_task(ev.docker, config, t, snap, agent_patch=patch, start_time=time.perf_counter())
                rec[f'{label}_resolved'] = bool(r.resolved); rec[f'{label}_exit'] = r.test_exit_code
                rec[f'{label}_error'] = (r.error or '')[:300]; rec[f'{label}_tail'] = (r.test_output or '')[-600:]
            except Exception as e:
                rec[f'{label}_exception'] = f'{type(e).__name__}: {str(e)[:300]}'
        rec['scorable'] = bool(rec.get('gold_resolved')) and not rec.get('empty_resolved', True)
        rec['wall_s'] = round(time.time() - t0, 1)
        with open(out, 'a') as f:
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f"{t.instance_id}: gold={rec.get('gold_resolved')} empty={rec.get('empty_resolved')} scorable={rec['scorable']} {rec['wall_s']}s {rec.get('gold_error','')[:80]}", flush=True)
    if hasattr(ev.docker, 'cleanup_all'):
        try: ev.docker.cleanup_all()
        except Exception: pass


if __name__ == '__main__':
    asyncio.run(main())
