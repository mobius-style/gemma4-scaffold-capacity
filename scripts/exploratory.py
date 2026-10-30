#!/usr/bin/env python3
"""EXPLORATORY analyses (not part of the pre-registered confirmatory family; written 2026-10-29 after
CONFIRMATORY_COMPLETE and after the confirmatory results were seen). Reads the same 30 confirmatory cells
via analyze.load_runs. Output: analysis/exploratory.json (+ stdout).

1. H1c opposite direction: one-sided bootstrap p for Delta_E4B > 0 (same B, same RNG seed as analyze.py)
2. Cost/cap: cap-hit rate, mean duration, tool calls, LLM calls per model x dose
3. Layout check (D8/D9): 26B per seed x dose with GPU layout; Delta within dual seeds vs single seeds
4. Contamination split by task created_at vs the Gemma 4 cutoff (2025-01-01)
5. Per-repository Delta
6. Flip rate per cell (share of tasks with mixed outcomes across seeds)
7. Harness-event counts per cell from shard logs (D2 malformed args, D5 peg conversions, context overflow)
"""
import glob, json, random, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

W = analyze.W
MODELS = ['gemma-4-26b-a4b-it', 'gemma-4-12b-it', 'gemma-4-e4b-it']
CUTOFF = '2025-01-01'
LAYOUT_26B = {1: 'dual', 2: 'dual', 3: 'single', 4: 'single', 5: 'single'}  # D8/D9; seed 3 completed on GPU1 after D9


def cap(r): return (r.get('error') or '').startswith('Agent exceeded')  # any budget: time, tool calls or turns
def time_cap(r): return 'session timeout' in (r.get('error') or '')
def tool_cap(r): return 'tool call budget' in (r.get('error') or '')


def complete_mid_cells():
    """mid cells (exploratory, seeds 1-3) from runs/, keeping only seeds with all tasks present."""
    cells, _ = analyze.load_runs(W / 'runs')
    n = len((W / 'TASKSET.txt').read_text().split())
    return {k: {s: v for s, v in seeds.items() if len(v) >= n} for k, seeds in cells.items() if k[0] == 'mid'}


def per_task_delta(cells, alias, keep=None, lo_dose='min', hi_dose='max'):
    lo, hi = cells[(lo_dose, alias)], cells[(hi_dose, alias)]
    common = sorted(set.intersection(*[set(s) for s in list(lo.values()) + list(hi.values())]))
    if keep is not None: common = [i for i in common if keep(i)]
    return {i: sum(bool(hi[s][i].get('resolved')) for s in hi) / len(hi) - sum(bool(lo[s][i].get('resolved')) for s in lo) / len(lo) for i in common}


def boot(d, B=2000, seed=20261022):
    rng = random.Random(seed); ks = sorted(d); n = len(ks); v = []
    for _ in range(B):
        v.append(sum(d[ks[rng.randrange(n)]] for _ in range(n)) / n)
    v.sort()
    return {'delta': sum(d.values()) / n, 'n_tasks': n, 'ci95': (v[int(0.025 * B)], v[int(0.975 * B) - 1]), 'ci90': (v[int(0.05 * B)], v[int(0.95 * B) - 1]),
            'p_pos': sum(1 for x in v if x <= 0) / B, 'p_neg': sum(1 for x in v if x >= 0) / B}


def main():
    cells, tasks = analyze.load_runs(W / 'analysis' / 'confirmatory_runs')
    out = {}
    # 1
    out['e4b_opposite_direction'] = boot(per_task_delta(cells, 'gemma-4-e4b-it'))
    out['e4b_opposite_direction']['note'] = 'p_pos = one-sided bootstrap p for Delta>0; Bonferroni bound over m=4 is 4*p_pos'
    # 1b mid level (exploratory, seeds 1-3; paired on the seeds the mid cell has)
    mid = complete_mid_cells(); out['mid_contrasts'] = {}
    for alias in MODELS:
        ms = mid.get(('mid', alias), {})
        if not ms: continue
        sub = {('mid', alias): ms}
        for dose in ('min', 'max'):
            sub[(dose, alias)] = {s: cells[(dose, alias)][s] for s in ms if s in cells[(dose, alias)]}
        rows = [r for s in ms.values() for r in s.values()]
        out['mid_contrasts'][alias] = {'seeds': sorted(ms), 'mid_resolved': sum(bool(r.get('resolved')) for r in rows) / len(rows),
                                       'mid_minus_min': boot(per_task_delta(sub, alias, lo_dose='min', hi_dose='mid')),
                                       'max_minus_mid': boot(per_task_delta(sub, alias, lo_dose='mid', hi_dose='max'))}
    # 2
    cost = {}
    for (dose, alias), seeds in cells.items():
        rows = [r for s in seeds.values() for r in s.values()]
        cost[f'{alias}/{dose}'] = {'runs': len(rows), 'resolved': sum(bool(r.get('resolved')) for r in rows) / len(rows),
                                  'cap_rate': sum(cap(r) for r in rows) / len(rows), 'time_cap_rate': sum(time_cap(r) for r in rows) / len(rows), 'tool_cap_rate': sum(tool_cap(r) for r in rows) / len(rows),
                                  'mean_duration_s': sum(r.get('duration_seconds') or 0 for r in rows) / len(rows),
                                  'mean_tool_calls': sum(r.get('tool_calls') or 0 for r in rows) / len(rows),
                                  'mean_llm_calls': sum(r.get('total_llm_calls') or 0 for r in rows) / len(rows)}
    out['cost'] = cost
    # 3
    lay = {}
    for dose in ('min', 'max'):
        for s, recs in sorted(cells[(dose, 'gemma-4-26b-a4b-it')].items()):
            rows = list(recs.values())
            lay[f'{dose}/s{s}'] = {'layout': LAYOUT_26B[s], 'resolved': sum(bool(r.get('resolved')) for r in rows),
                                   'cap': sum(cap(r) for r in rows), 'mean_duration_s': round(sum(r.get('duration_seconds') or 0 for r in rows) / len(rows), 1)}
    sub = {}
    for name, ss in (('dual', [1, 2]), ('single', [3, 4, 5])):
        lo = {s: cells[('min', 'gemma-4-26b-a4b-it')][s] for s in ss}; hi = {s: cells[('max', 'gemma-4-26b-a4b-it')][s] for s in ss}
        common = sorted(set.intersection(*[set(v) for v in list(lo.values()) + list(hi.values())]))
        d = {i: sum(bool(hi[s][i].get('resolved')) for s in ss) / len(ss) - sum(bool(lo[s][i].get('resolved')) for s in ss) / len(ss) for i in common}
        sub[name] = boot(d)
    out['layout_26b'] = {'per_seed': lay, 'delta_by_layout': sub}
    # 4
    cont = {}
    for alias in MODELS:
        for name, keep in (('pre_cutoff', lambda i: tasks[i]['created_at'][:10] < CUTOFF), ('post_cutoff', lambda i: tasks[i]['created_at'][:10] >= CUTOFF)):
            cont[f'{alias}/{name}'] = boot(per_task_delta(cells, alias, keep))
    out['contamination_split'] = cont
    # 5
    repo = {}
    for alias in MODELS:
        for rp in sorted({t['repo'] for t in tasks.values()}):
            d = per_task_delta(cells, alias, lambda i, rp=rp: tasks[i]['repo'] == rp)
            if d: repo[f'{alias}/{rp}'] = boot(d)
    out['per_repo'] = repo
    # 6
    flip = {}
    for (dose, alias), seeds in cells.items():
        common = set.intersection(*[set(v) for v in seeds.values()])
        mixed = sum(1 for i in common if 0 < sum(bool(seeds[s][i].get('resolved')) for s in seeds) < len(seeds))
        flip[f'{alias}/{dose}'] = mixed / len(common)
    out['flip_rate'] = flip
    # 7
    ev = {}
    for d in sorted(glob.glob(str(W / 'analysis' / 'confirmatory_runs' / '*_s[0-9]'))):
        txt = ''.join(Path(f).read_text(errors='ignore') for f in glob.glob(f'{d}/shard*.log'))
        ev[Path(d).name] = {'malformed_args': txt.count('malformed tool-call arguments'), 'peg_conversions': txt.count('peg-gemma4 parse failure'),
                            'context_overflow_log_lines': txt.count('litellm.exceptions.ContextWindowExceededError')}
    out['harness_events'] = ev
    (W / 'analysis').mkdir(exist_ok=True)
    json.dump(out, open(W / 'analysis' / 'exploratory.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
