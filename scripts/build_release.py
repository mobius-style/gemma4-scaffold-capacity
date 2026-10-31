#!/usr/bin/env python3
"""Assemble the public release tree from canonical sources (never from its own output).

Policy: no Competition Data is redistributed — no problem statements, reference/test patches, repository code,
agent patches, traces or test logs. Run records keep outcome metadata only (plus a SHA-256 of each agent patch so
later checks can match patches without the patch text). Files that come from the competition's sample submission
(analyzer prompt, sub-agent config) are described, not copied.
Usage: venv/bin/python scripts/build_release.py [--out release/gemma4-scaffold-capacity]
"""
import argparse, hashlib, json, shutil, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402
import diagnostics  # noqa: E402

W = analyze.W
R = W / 'analysis' / 'confirmatory_runs'
COPY = {  # release path -> source path (relative to W)
    'PREREGISTRATION.md': 'PREREGISTRATION.md', 'DEVIATIONS.md': 'DEVIATIONS.md', 'DEVIATIONS_auto.md': 'DEVIATIONS_auto.md',
    'freeze/FREEZE.sha256': 'FREEZE.sha256', 'freeze/FREEZE.timestamp': 'FREEZE.timestamp',
    'freeze/FREEZE.addendum.sha256': 'FREEZE.addendum.sha256', 'freeze/FREEZE.addendum2.sha256': 'FREEZE.addendum2.sha256',
    'TASKSET.txt': 'TASKSET.txt', 'logs/schedule.log': 'logs/schedule.log',
    'prompts/system_min.md': 'subs/prompts/system_min.md', 'prompts/system_full.md': 'subs/prompts/system_full.md',
    'configs/sampling.yaml': 'subs/sampling.yaml', 'configs/eval_config.yaml': 'subs/eval_config.yaml',
    'configs/min/agent.yaml': 'subs/min/agent.yaml', 'configs/mid/agent.yaml': 'subs/mid/agent.yaml', 'configs/max/agent.yaml': 'subs/max/agent.yaml',
    'run_server.sh': 'run_server.sh', 'paper/paper.md': 'paper/DRAFT.md',
    'README.md': 'scripts/RELEASE_README.md',
}
SCRIPTS = ['analyze.py', 'exploratory.py', 'secondary.py', 'diagnostics.py', 'figures.py', 'check_numbers.py', 'eval_cli.py',
           'run_cell.sh', 'schedule_v2.sh', 'schedule_v3.sh', 'gold_verify.py', 'resume_after_boot.sh', 'watch_events.sh',
           'sample_power.sh', 'build_release.py', 'reproduce.py']
MODELS = {'e4b': 'google/gemma-4-E4B-it-qat-q4_0-gguf', '12b': 'google/gemma-4-12B-it-qat-q4_0-gguf', '26b': 'google/gemma-4-26B-A4B-it-qat-q4_0-gguf'}


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=str(W / 'release' / 'gemma4-scaffold-capacity')); a = ap.parse_args()
    out = Path(a.out)
    assert W not in out.parents or 'release' in out.parts, 'refusing to write outside release/'
    if out.exists(): shutil.rmtree(out)
    for dst, src in COPY.items():
        (out / dst).parent.mkdir(parents=True, exist_ok=True); shutil.copy2(W / src, out / dst)
    for s in SCRIPTS:
        if (W / 'scripts' / s).exists():
            (out / 'scripts').mkdir(parents=True, exist_ok=True); shutil.copy2(W / 'scripts' / s, out / 'scripts' / s)
    shutil.copy2(W / 'subs' / 'instantiate.sh', out / 'scripts' / 'instantiate.sh')
    for f in ['summary.json', 'exploratory.json', 'secondary.json', 'diagnostics.json']:
        (out / 'analysis').mkdir(exist_ok=True); shutil.copy2(W / 'analysis' / f, out / 'analysis' / f)
    for f in sorted((W / 'paper').glob('fig*')):
        (out / 'paper').mkdir(exist_ok=True); shutil.copy2(f, out / 'paper' / f.name)
    # writeup illustrations (image-model renderings from the author's specification; provenance in their README)
    ill = out / 'paper' / 'illustrations'; ill.mkdir(parents=True, exist_ok=True); G = W / 'release_meta' / 'figures_gpt'
    for src, dst in (('FA_design.png', 'design_overview.png'), ('FB_result.png', 'result_summary.png'), ('FC_cover.png', 'cover.png'), ('ILLUSTRATIONS_README.md', 'README.md')):
        shutil.copy2(G / src, ill / dst)
    shutil.copy2(W / 'venv' / 'lib' / 'python3.12' / 'site-packages' / 'google_adk-1.39.1.dist-info' / 'licenses' / 'LICENSE', out / 'LICENSE')
    # models: SHA-256 as recorded before the freeze + source repositories
    (out / 'models').mkdir(exist_ok=True)
    with open(out / 'models' / 'SHA256SUMS', 'w') as f:
        for k in MODELS:
            f.write((W / 'logs' / f'model_sha256_{k}.txt').read_text().strip() + f'   # https://huggingface.co/{MODELS[k]}\n')
    # tasks: metadata and gold/empty verdicts only
    tasks = analyze.load_tasks(); keep = set((W / 'TASKSET.txt').read_text().split())
    (out / 'data').mkdir(exist_ok=True)
    with open(out / 'data' / 'gold_empty_verification.jsonl', 'w') as f:
        for l in open(W / 'logs' / 'gold_verify_repaired.jsonl'):
            g = json.loads(l)
            f.write(json.dumps({k: g.get(k) for k in ('instance_id', 'repo', 'gold_resolved', 'gold_exit', 'empty_resolved', 'empty_exit', 'scorable')}
                               | {'created_at': tasks.get(g['instance_id'], {}).get('created_at'), 'in_taskset': g['instance_id'] in keep}) + '\n')
    # run records: one line per task-run, outcome metadata only
    cells, _ = analyze.load_runs(R)
    import exploratory
    mid = exploratory.complete_mid_cells(); cells.update(mid)
    n = 0; n_mid = sum(len(v) for seeds in mid.values() for v in seeds.values())
    with open(out / 'data' / 'runs.jsonl', 'w') as f:
        for (dose, alias), seeds in sorted(cells.items()):
            for s, recs in sorted(seeds.items()):
                for iid, r in sorted(recs.items()):
                    base = R if dose != 'mid' else W / 'runs'
                    sh = next(x for x in base.glob(f'{dose}_{alias}_s{s}/shard*') if x.is_dir() and (x / 'traces' / f'trace_{iid}.json').exists())
                    ti = diagnostics.trace_info(sh / 'traces' / f'trace_{iid}.json')
                    p = sh / 'patches' / f'{iid}.patch'
                    f.write(json.dumps({'level': dose, 'model': alias, 'seed': s, 'shard': sh.name, 'instance_id': iid, 'repo': r.get('repo'),
                                        'resolved': bool(r.get('resolved')), 'ladder': r['ladder'], 'end_reason': diagnostics.reason(r),
                                        'error_class': (r.get('error') or '')[:60] or None, 'duration_seconds': r.get('duration_seconds'),
                                        'tool_calls': r.get('tool_calls'), 'total_llm_calls': r.get('total_llm_calls'),
                                        'agent_patch_size': r.get('agent_patch_size'), 'test_exit_code': r.get('test_exit_code'),
                                        'submit_patch_called': ti['submit'], 'tool_counts': dict(ti['tools']),
                                        'trace_parser_conversions': ti['peg'], 'trace_malformed_args': ti['malformed'],
                                        'agent_patch_sha256': sha(p) if p.exists() else None}) + '\n'); n += 1
    assert n == 30 * 104 + n_mid, (n, n_mid)
    print(f'mid task-runs included: {n_mid}')
    # grader repair: wheel set listing + Dockerfile delta (our lines only)
    (out / 'grader').mkdir(exist_ok=True)
    (out / 'grader' / 'wheels_repaired.txt').write_text('\n'.join(sorted(p.name for p in (W / 'data_repaired' / 'wheels').iterdir())) + '\n')
    df = (W / 'build_ctx3' / 'Dockerfile.public').read_text().splitlines()
    i = next(k for k, l in enumerate(df) if l.startswith('# repaired'))
    (out / 'grader' / 'Dockerfile.delta').write_text('# Lines appended to the competition sandbox Dockerfile (Dockerfile.public) for the repaired image\n' + '\n'.join(df[i:]) + '\n')
    (out / 'requirements.lock').write_text(subprocess.run([str(W / 'venv' / 'bin' / 'pip'), 'freeze'], capture_output=True, text=True, check=True).stdout)
    # release manifest: every released file, written last (excludes itself)
    files = sorted(p for p in out.rglob('*') if p.is_file())
    (out / 'freeze' / 'FREEZE.release.sha256').write_text(f'# RELEASE MANIFEST {__import__("datetime").datetime.now().astimezone().isoformat()}\n' + ''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    print(f'release tree written to {out} ({n} run records, {len(files)} files + manifest)')


if __name__ == '__main__':
    main()
