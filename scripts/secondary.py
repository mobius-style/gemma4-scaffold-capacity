#!/usr/bin/env python3
"""Pre-registered secondary and exploratory analyses that analyze.py (frozen) does not compute.
PREREGISTRATION §5: GEE logistic (secondary); ladder paired sign test, BH over the exploratory family;
13-gram overlap between agent patch and reference patch (contamination screen, G5).
Written 2026-10-29 after the confirmatory results were seen; the specifications are the pre-registered ones.
Output: analysis/secondary.json (+ stdout). Requires analysis/exploratory.json (scripts/exploratory.py).
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

W = analyze.W
MODELS = ['gemma-4-26b-a4b-it', 'gemma-4-12b-it', 'gemma-4-e4b-it']
CUTOFF = '2025-01-01'


def gee(cells):
    import pandas as pd
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    rows = [{'task': i, 'model': alias, 'dose': dose, 'seed': s, 'y': int(bool(r.get('resolved')))}
            for (dose, alias), seeds in cells.items() for s, recs in seeds.items() for i, r in recs.items()]
    df = pd.DataFrame(rows)
    df['model'] = pd.Categorical(df['model'], categories=['gemma-4-e4b-it', 'gemma-4-12b-it', 'gemma-4-26b-a4b-it'])
    df['dose'] = pd.Categorical(df['dose'], categories=['min', 'max'])
    m = smf.gee('y ~ C(dose) * C(model)', groups='task', data=df, family=sm.families.Binomial(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int()
    return {'n_rows': len(df), 'n_tasks': df['task'].nunique(), 'working_corr': float(m.model.cov_struct.dep_params),
            'coef': {k: {'logit': float(m.params[k]), 'ci95': [float(ci.loc[k, 0]), float(ci.loc[k, 1])], 'p': float(m.pvalues[k])} for k in m.params.index}}


def ladder_sign(cells):
    from scipy.stats import binomtest
    out = {}
    for alias in MODELS:
        lo, hi = cells[('min', alias)], cells[('max', alias)]
        common = sorted(set.intersection(*[set(s) for s in list(lo.values()) + list(hi.values())]))
        diffs = [sum(hi[s][i]['ladder'] for s in hi) / len(hi) - sum(lo[s][i]['ladder'] for s in lo) / len(lo) for i in common]
        pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        out[alias] = {'n_tasks': len(common), 'mean_level_min': sum(sum(lo[s][i]['ladder'] for s in lo) / len(lo) for i in common) / len(common),
                      'mean_level_max': sum(sum(hi[s][i]['ladder'] for s in hi) / len(hi) for i in common) / len(common),
                      'tasks_up': pos, 'tasks_down': neg, 'ties': len(diffs) - pos - neg,
                      'p_two_sided': binomtest(pos, pos + neg).pvalue if pos + neg else 1.0}
    return out


def added_ngrams(patch, n=13):
    toks = ' '.join(l[1:] for l in patch.splitlines() if l.startswith('+') and not l.startswith('+++')).split()
    return {tuple(toks[k:k + n]) for k in range(len(toks) - n + 1)}


def removed_ngrams(patch, n=13):
    toks = ' '.join(l[1:] for l in patch.splitlines() if l.startswith('-') and not l.startswith('---')).split()
    return {tuple(toks[k:k + n]) for k in range(len(toks) - n + 1)}


def overlap(cells, tasks_meta, novel_only=False):
    # novel_only: drop reference 13-grams that also occur in the reference patch's removed lines (code moves),
    # since reproducing moved code is copying from the repository, not recall of the fix.
    ref = {}
    for l in open(analyze.TASKS):
        if l.strip():
            d = json.loads(l); ref[d['instance_id']] = added_ngrams(d['patch']) - (removed_ngrams(d['patch']) if novel_only else set())
    runs_dir = W / 'analysis' / 'confirmatory_runs'
    out = {}
    for (dose, alias), seeds in cells.items():
        vals = {'pre': [], 'post': []}
        for s in seeds:
            for shard in sorted(runs_dir.glob(f'{dose}_{alias}_s{s}/shard*')):
                for p in (shard / 'patches').glob('*.patch'):
                    iid = p.stem
                    if iid not in ref or not ref[iid] or not seeds[s].get(iid, {}).get('resolved'): continue
                    a = added_ngrams(p.read_text(errors='ignore'))
                    stratum = 'pre' if tasks_meta[iid]['created_at'][:10] < CUTOFF else 'post'
                    vals[stratum].append(len(a & ref[iid]) / len(ref[iid]))
        out[f'{alias}/{dose}'] = {k: {'n_resolved_patches': len(v), 'mean_ref13_covered': (sum(v) / len(v)) if v else None,
                                      'share_ge_0.5': (sum(x >= 0.5 for x in v) / len(v)) if v else None} for k, v in vals.items()}
    return out


def bh(pairs):
    m = len(pairs); ranked = sorted(pairs, key=lambda kv: kv[1]); adj = {}; prev = 1.0
    for rank in range(m, 0, -1):
        k, p = ranked[rank - 1]; prev = min(prev, p * m / rank); adj[k] = prev
    return adj


def main():
    cells, tasks = analyze.load_runs(W / 'analysis' / 'confirmatory_runs')
    cells = {k: v for k, v in cells.items() if k[0] in ('min', 'max')}
    out = {'gee': gee(cells), 'ladder_sign_test': ladder_sign(cells), 'ngram13_overlap': overlap(cells, tasks), 'ngram13_overlap_novel': overlap(cells, tasks, novel_only=True)}
    ex = json.load(open(W / 'analysis' / 'exploratory.json'))
    # BH family as registered (PREREG §5 'Exploratory (BH)'): ladder sign tests + per-repository contrasts
    # (+ mid vs min/max once the mid cells exist). Contamination strata (G5) and the layout check (D8/D9) are
    # reported with CIs and unadjusted p outside the family.
    fam = [(f'ladder/{a}', v['p_two_sided']) for a, v in out['ladder_sign_test'].items()]
    fam += [(f'per_repo/{k}', min(1.0, 2 * min(v['p_pos'], v['p_neg']))) for k, v in ex['per_repo'].items()]
    for a, v in ex.get('mid_contrasts', {}).items():
        fam += [(f'mid/{a}/{c}', min(1.0, 2 * min(v[c]['p_pos'], v[c]['p_neg']))) for c in ('mid_minus_min', 'max_minus_mid')]
    out['bh_exploratory'] = {'m': len(fam), 'p_raw': dict(fam), 'q': bh(fam)}
    out['unadjusted_outside_family'] = {f'{s}/{k}': min(1.0, 2 * min(v['p_pos'], v['p_neg'])) for s in ('contamination_split',) for k, v in ex[s].items()}
    out['unadjusted_outside_family'].update({f'layout_26b/{k}': min(1.0, 2 * min(v['p_pos'], v['p_neg'])) for k, v in ex['layout_26b']['delta_by_layout'].items()})
    # Registered secondary ordered check (PREREG §1 H2): OLS slope of Delta over capacity rank (E4B 1, 12B 2, 26B 3);
    # with equally spaced ranks the slope is (Delta_26B - Delta_E4B) / 2, so its bootstrap distribution is H2's halved.
    h2 = json.load(open(W / 'analysis' / 'summary.json'))['confirmatory']['tests']['H2_26B_minus_E4B']
    out['slope_check'] = {'slope': h2['delta'] / 2, 'ci95': [h2['ci'][0] / 2, h2['ci'][1] / 2], 'p_one_sided_gt0': h2['p_one_sided'], 'note': 'equals H2/2 for ranks 1,2,3'}
    # Registered descriptive: exact McNemar on majority-of-5 (from analyze.py summary)
    out['mcnemar_majority'] = {a: {k: c[k] for k in ('max_only', 'min_only', 'mcnemar_p')} for a, c in json.load(open(W / 'analysis' / 'summary.json'))['contrasts'].items()}
    json.dump(out, open(W / 'analysis' / 'secondary.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
