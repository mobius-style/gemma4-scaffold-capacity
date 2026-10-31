#!/usr/bin/env python3
"""Acceptance gate: every key number printed in paper/DRAFT.md must equal the value in analysis/*.json (rounded as printed)."""
import json, sys
from pathlib import Path
W = Path(__file__).resolve().parent.parent
d = (W / 'paper' / 'DRAFT.md').read_text()
s = json.load(open(W / 'analysis/summary.json')); e = json.load(open(W / 'analysis/exploratory.json')); x = json.load(open(W / 'analysis/secondary.json'))
T = s['confirmatory']['tests']
def pct(v): return f'{100*v:.1f}%'
def f3(v): return f'{v:+.3f}'.replace('+-', '-')
def s3(v): return f'{v:.3f}' if v >= 0 else f'−{abs(v):.3f}'
exp = []
for a, k in (('gemma-4-26b-a4b-it', 'H1_gemma-4-26b-a4b-it'), ('gemma-4-12b-it', 'H1_gemma-4-12b-it'), ('gemma-4-e4b-it', 'H1_gemma-4-e4b-it')):
    t = T[k]; lo, hi = t['ci']; tl, th = t['tost_90ci']
    exp += [f"[{s3(lo) if lo<0 else '+'+format(lo,'.3f')}, {s3(hi) if hi<0 else '+'+format(hi,'.3f')}]"]
    if a != 'gemma-4-e4b-it': exp += [f"[{s3(tl) if tl<0 else '+'+format(tl,'.3f')}, {s3(th) if th<0 else '+'+format(th,'.3f')}]"]
h2 = T['H2_26B_minus_E4B']; exp += [f"[{s3(h2['ci'][0])}, {s3(h2['ci'][1])}]"]
g_ = json.load(open(W / 'analysis/diagnostics.json'))
for key, v in e['cost'].items():
    exp += [pct(v['resolved']), pct(v['time_cap_rate']), pct(v['tool_cap_rate'])]
for key, v in g_['cells'].items():
    exp += [pct(v['pairwise_two_seed_disagreement']), pct(v['no_submit_patch']), pct(v['zero_size_patch'])]
sen = g_['sensitivity_12b_excl_parser_window']; exp += [f"−{abs(sen['delta']):.3f} (CI −{abs(sen['ci95'][0]):.3f} to +{sen['ci95'][1]:.3f})", f"{sen['excluded_task_runs']['min'] + sen['excluded_task_runs']['max']} of 1,040"]
ov = g_['context_overflow']; exp += [f"All {sum(v['events'] for v in ov.values())} context-window overflows", f"at least {sum(v['in_analyzer'] for v in ov.values())} of them inside the analyzer"]
assert all(v['events'] == 0 for k, v in ov.items() if k.endswith('/min')), 'overflow in a min cell'
cw = [g_['cells'][k]['capped_without_submit'] for k in g_['cells'] if not k.startswith('gemma-4-e4b')]; exp += [f"{100*min(cw):.0f}–{100*max(cw):.0f}% ended without"]
lay90 = e['layout_26b']['delta_by_layout']
def sg(v): return f"+{v:.3f}" if v >= 0 else f"−{abs(v):.3f}"
exp += [f"[{sg(lay90[k]['ci90'][0])}, {sg(lay90[k]['ci90'][1])}]" for k in ('dual', 'single')]
sl = x['slope_check']; exp += [f"−{abs(sl['slope']):.3f} (−{abs(sl['ci95'][0]):.3f} to −{abs(sl['ci95'][1]):.3f})"]
q = x['bh_exploratory']['q']; exp += [f"q = {q['ladder/gemma-4-12b-it']:.4f}".rstrip('0'), f"q = {q['ladder/gemma-4-26b-a4b-it']:.2f}"]
assert x['bh_exploratory']['m'] == 18 and '(m = 18)' in d, 'BH family size'
for a in ('gemma-4-e4b-it', 'gemma-4-12b-it', 'gemma-4-26b-a4b-it'):
    exp += [pct(e['mid_contrasts'][a]['mid_resolved'])]
me = e['mid_contrasts']['gemma-4-e4b-it']; exp += [f"mid − min +{me['mid_minus_min']['delta']:.3f}, max − mid +{me['max_minus_mid']['delta']:.3f}"]
assert min(v for k, v in q.items() if k.startswith('mid/')) >= 0.26, 'mid q below stated floor'
# run-record count stated in §7 must equal the released records (confirmatory = min/max, exploratory = mid)
_rp = next(q for q in ('release/gemma4-scaffold-capacity/data/runs.jsonl', 'data/runs.jsonl') if __import__('os').path.exists(q))
_runs = [__import__('json').loads(l) for l in open(_rp)]
_conf = sum(r['level'] in ('min', 'max') for r in _runs); _mid = sum(r['level'] == 'mid' for r in _runs)
exp += [f"({len(_runs):,}: {_conf:,} confirmatory and {_mid:,} exploratory mid-level)"]
mc = x['mcnemar_majority']; exp += [f"{mc['gemma-4-e4b-it']['max_only']} vs {mc['gemma-4-e4b-it']['min_only']} discordant tasks for E4B", f"{mc['gemma-4-12b-it']['max_only']} vs {mc['gemma-4-12b-it']['min_only']} for 12B", f"{mc['gemma-4-26b-a4b-it']['max_only']} vs {mc['gemma-4-26b-a4b-it']['min_only']} for 26B"]
L = x['ladder_sign_test']
exp += [f"{L['gemma-4-e4b-it']['mean_level_min']:.2f} to {L['gemma-4-e4b-it']['mean_level_max']:.2f}", f"{L['gemma-4-12b-it']['mean_level_min']:.2f} to {L['gemma-4-12b-it']['mean_level_max']:.2f}",
        f"{L['gemma-4-e4b-it']['tasks_up']} tasks move up and {L['gemma-4-e4b-it']['tasks_down']} down", f"{L['gemma-4-12b-it']['tasks_down']} tasks move down and {L['gemma-4-12b-it']['tasks_up']} up",
        f"{L['gemma-4-26b-a4b-it']['tasks_up']} up, {L['gemma-4-26b-a4b-it']['tasks_down']} down"]
g = x['gee']['coef']
exp += [f"+{g['C(dose)[T.max]']['logit']:.2f}", f"−{abs(g['C(dose)[T.max]:C(model)[T.gemma-4-12b-it]']['logit']):.2f}", f"−{abs(g['C(dose)[T.max]:C(model)[T.gemma-4-26b-a4b-it]']['logit']):.2f}"]
c = e['contamination_split']
exp += [f"+{c['gemma-4-e4b-it/post_cutoff']['delta']:.3f}", f"−{abs(c['gemma-4-26b-a4b-it/pre_cutoff']['delta']):.3f}", f"+{c['gemma-4-26b-a4b-it/post_cutoff']['delta']:.3f}"]
lay = e['layout_26b']['delta_by_layout']; exp += [f"| +{lay['dual']['delta']:.3f} |", f"| −{abs(lay['single']['delta']):.3f} |"]
miss = [v for v in exp if v not in d]
print(f'checked {len(exp)} values; missing {len(miss)}'); [print('  MISSING', m) for m in miss]
sys.exit(1 if miss else 0)
