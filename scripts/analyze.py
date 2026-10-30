#!/usr/bin/env python3
"""Single source of numbers for the paper. Reads runs/<dose>_<alias>_s<seed>/shard*/ artifacts.

Outputs (stdout + analysis/summary.json):
- per cell: n tasks, resolved rate, flip rate across seeds, ladder distribution
- per model: min vs max paired comparison (McNemar exact on majority-of-seeds, bootstrap CI on risk difference)
Ladder: L0 no patch / L1 non-empty patch / L2 touches a reference-patch file / L3 tests collect / L4 >=1 required node passed / L5 resolved
"""
import argparse, glob, json, math, re, sys
from collections import defaultdict
from pathlib import Path

W = Path(__file__).resolve().parent.parent
TASKS = W / 'data_repaired' / 'tasks.jsonl'


def ref_files(patch):
    return set(re.findall(r'^\+\+\+ b/(\S+)', patch, re.M)) | set(re.findall(r'^--- a/(\S+)', patch, re.M))


def load_tasks():
    out = {}
    for l in open(TASKS):
        if l.strip():
            d = json.loads(l); out[d['instance_id']] = {'ref_files': ref_files(d['patch']), 'repo': d['repo'], 'created_at': d.get('created_at', '')}
    return out


def ladder(rec, run_dir, task):
    iid = rec['instance_id']
    if rec.get('resolved'): return 5
    size = rec.get('agent_patch_size') or 0
    if size <= 0: return 0
    patch = ''
    p = run_dir / 'patches' / f'{iid}.patch'
    if p.exists(): patch = p.read_text(errors='ignore')
    touched = ref_files(patch)
    lvl = 1
    if touched & task['ref_files']: lvl = 2
    exit_code = rec.get('test_exit_code')
    out = ''
    t = run_dir / 'test_outputs' / f'{iid}.log'
    if t.exists(): out = t.read_text(errors='ignore')
    collected = exit_code not in (2, 4, None, -1) and 'Interrupted: ' not in out and 'error during collection' not in out
    if collected and lvl >= 2: lvl = 3
    if collected and lvl < 2: lvl = max(lvl, 1)  # applied but wrong files: keep L1
    m = re.search(r'(\d+) passed', out)
    if lvl >= 3 and m and int(m.group(1)) > 0: lvl = 4
    return lvl


def load_runs(runs_dir):
    tasks = load_tasks()
    cells = defaultdict(lambda: defaultdict(dict))  # (dose, alias) -> seed -> iid -> rec
    for d in sorted(glob.glob(str(runs_dir / '*_s[0-9]*'))):
        name = Path(d).name
        m = re.match(r'(min|mid|max)_(.+)_s(\d+)$', name)
        if not m: continue
        dose, alias, seed = m.group(1), m.group(2), int(m.group(3))
        for shard in sorted(glob.glob(f'{d}/shard*')):
            f = Path(shard) / 'task_results.jsonl'
            if not f.exists(): continue
            for l in open(f):
                if not l.strip(): continue
                r = json.loads(l); iid = r['instance_id']
                if iid not in tasks: continue
                r['ladder'] = ladder(r, Path(shard), tasks[iid])
                cells[(dose, alias)][seed][iid] = r
    return cells, tasks


def mcnemar_exact(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def summarize(cells):
    out = {'cells': {}, 'contrasts': {}}
    for (dose, alias), seeds in sorted(cells.items()):
        iids = sorted(set.intersection(*[set(s.keys()) for s in seeds.values()])) if seeds else []
        R = len(seeds)
        res = {iid: [bool(seeds[s][iid].get('resolved')) for s in sorted(seeds)] for iid in iids}
        lad = {iid: [seeds[s][iid]['ladder'] for s in sorted(seeds)] for iid in iids}
        rate = sum(sum(v) for v in res.values()) / (R * len(iids)) if iids else None
        flip = sum(1 for v in res.values() if 0 < sum(v) < R) / len(iids) if iids and R > 1 else None
        dist = defaultdict(int)
        for v in lad.values():
            for x in v: dist[x] += 1
        dur = [seeds[s][iid].get('duration_seconds', 0) for s in seeds for iid in iids]
        out['cells'][f'{dose}|{alias}'] = {'seeds': R, 'n_tasks': len(iids), 'resolved_rate': rate, 'flip_rate': flip,
                                             'ladder_dist': {f'L{k}': dist[k] for k in range(6)},
                                             'mean_duration_s': sum(dur) / len(dur) if dur else None}
    aliases = sorted({a for (_, a) in cells})
    for alias in aliases:
        lo, hi = cells.get(('min', alias)), cells.get(('max', alias))
        if not lo or not hi: continue
        common = sorted(set.intersection(*[set(s.keys()) for s in list(lo.values()) + list(hi.values())]))
        def maj(seeds, iid):
            v = [bool(seeds[s][iid].get('resolved')) for s in seeds]; return sum(v) * 2 > len(v)
        b = sum(1 for i in common if maj(hi, i) and not maj(lo, i))
        c = sum(1 for i in common if maj(lo, i) and not maj(hi, i))
        rate_lo = sum(sum(bool(lo[s][i].get('resolved')) for s in lo) for i in common) / (len(lo) * len(common)) if common else None
        rate_hi = sum(sum(bool(hi[s][i].get('resolved')) for s in hi) for i in common) / (len(hi) * len(common)) if common else None
        out['contrasts'][alias] = {'n': len(common), 'max_only': b, 'min_only': c, 'mcnemar_p': mcnemar_exact(b, c),
                                   'rate_min': rate_lo, 'rate_max': rate_hi,
                                   'risk_diff': (rate_hi - rate_lo) if common else None}
    return out


def confirmatory(cells, B=2000, seed=20261022, mid=0.05):
    """Pre-registered inference. Per model: Delta = mean_t[p_max(t) - p_min(t)], p = fraction of seeds resolved.
    Paired bootstrap over tasks (percentile), one-sided p per declared sign, Holm over m=4 (H1a 26B>0, H1b 12B>0,
    H1c E4B<0, H2 Delta_26B - Delta_E4B > 0); TOST for E4B within +/-mid; descriptive McNemar on majority."""
    import random
    rng = random.Random(seed)
    sign = {'gemma-4-26b-a4b-it': +1, 'gemma-4-12b-it': +1, 'gemma-4-e4b-it': -1}
    per_task = {}
    for alias in sign:
        lo, hi = cells.get(('min', alias)), cells.get(('max', alias))
        if not lo or not hi: continue
        common = sorted(set.intersection(*[set(s.keys()) for s in list(lo.values()) + list(hi.values())]))
        d = {}
        for i in common:
            pmin = sum(bool(lo[s][i].get('resolved')) for s in lo) / len(lo)
            pmax = sum(bool(hi[s][i].get('resolved')) for s in hi) / len(hi)
            d[i] = pmax - pmin
        per_task[alias] = d
    if not per_task: return {}
    tasks = sorted(set.intersection(*[set(d) for d in per_task.values()]))
    n = len(tasks)
    def mean(v): return sum(v) / len(v) if v else float('nan')
    est = {a: mean([per_task[a][t] for t in tasks]) for a in per_task}
    boots = {a: [] for a in per_task}; inter = []
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        bs = {a: mean([per_task[a][tasks[j]] for j in idx]) for a in per_task}
        for a in bs: boots[a].append(bs[a])
        if 'gemma-4-26b-a4b-it' in bs and 'gemma-4-e4b-it' in bs: inter.append(bs['gemma-4-26b-a4b-it'] - bs['gemma-4-e4b-it'])
    def ci(v): v = sorted(v); return (v[int(0.025 * len(v))], v[int(0.975 * len(v)) - 1]) if v else (None, None)
    def p_one(v, s):  # bootstrap one-sided p: share of resamples on the wrong side of 0
        if not v: return None
        return (sum(1 for x in v if x <= 0) if s > 0 else sum(1 for x in v if x >= 0)) / len(v)
    tests = {}
    for a in per_task:
        tests[f'H1_{a}'] = {'delta': est[a], 'ci': ci(boots[a]), 'p_one_sided': p_one(boots[a], sign[a]), 'n_tasks': n,
                            'tost_90ci': (sorted(boots[a])[int(0.05 * B)], sorted(boots[a])[int(0.95 * B) - 1]) if boots[a] else None}
    if inter:
        e = est['gemma-4-26b-a4b-it'] - est['gemma-4-e4b-it']
        tests['H2_26B_minus_E4B'] = {'delta': e, 'ci': ci(inter), 'p_one_sided': p_one(inter, +1), 'n_tasks': n}
    # Holm over the declared family (only tests with data)
    fam = [(k, v['p_one_sided']) for k, v in tests.items() if v.get('p_one_sided') is not None]
    m = len(fam); adj = {}
    for rank, (k, p) in enumerate(sorted(fam, key=lambda kv: kv[1])):
        adj[k] = min(1.0, max([adj.get(prev, 0) for prev in adj] + [p * (m - rank)]))
    for k in tests: tests[k]['p_holm'] = adj.get(k); tests[k]['m'] = m
    e4 = tests.get('H1_gemma-4-e4b-it')
    if e4 and e4.get('tost_90ci'):
        lo, hi = e4['tost_90ci']; e4['tost_equivalent_within_mid'] = (lo > -mid and hi < mid)
    return {'tests': tests, 'B': B, 'seed': seed, 'mid': mid}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default=str(W / 'runs')); a = ap.parse_args()
    cells, _ = load_runs(Path(a.runs))
    s = summarize(cells)
    s['confirmatory'] = confirmatory(cells)
    for alias, c in s['contrasts'].items():
        t = s['confirmatory'].get('tests', {}).get(f'H1_{alias}')
        if t: c['ci_lo'], c['ci_hi'], c['p_holm'] = t['ci'][0], t['ci'][1], t['p_holm']
    (W / 'analysis').mkdir(exist_ok=True)
    json.dump(s, open(W / 'analysis' / 'summary.json', 'w'), indent=1)
    print(json.dumps(s, indent=1))
