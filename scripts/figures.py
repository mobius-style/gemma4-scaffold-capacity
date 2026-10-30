#!/usr/bin/env python3
"""Figures for the paper, from analysis/summary.json (produced by analyze.py) and run-level artifacts.
Palette: dataviz reference instance (categorical slots 1-3 validated all-pairs, light mode): blue #2a78d6 (26B),
orange #eb6834 (12B), aqua #1baf7a (E4B); sequential blue ramp for the ordinal ladder. Text uses ink tokens.
Outputs paper/fig1_resolution.png, fig2_ladder.png, fig3_delta.png (PNG, 200 dpi) and a CSV per figure.
"""
import json, glob, re, csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

W = Path(__file__).resolve().parent.parent
OUT = W / 'paper'; OUT.mkdir(exist_ok=True)
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e6e5e1', '#fcfcfb'
CAT = {'gemma-4-26b-a4b-it': ('26B-A4B', '#2a78d6'), 'gemma-4-12b-it': ('12B', '#eb6834'), 'gemma-4-e4b-it': ('E4B', '#1baf7a')}
ORDER = ['gemma-4-e4b-it', 'gemma-4-12b-it', 'gemma-4-26b-a4b-it']
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95']  # L0..L5, light->dark
DOSES = ['min', 'mid', 'max']

plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'text.color': INK, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def seed_rates():
    """cell -> list of per-seed resolved rates, from runs/*_s<seed>/shard*/task_results.jsonl"""
    out = {}
    rel = W / 'data' / 'runs.jsonl'
    if not (W / 'runs').exists() and rel.exists():  # released tree: per-run records instead of run directories
        acc = {}
        for l in open(rel):
            r = json.loads(l); k = (r['level'], r['model'], r['seed']); acc.setdefault(k, []).append(bool(r['resolved']))
        for (dose, alias, seed), v in sorted(acc.items()): out.setdefault((dose, alias), []).append(sum(v) / len(v))
        return out
    for d in sorted(glob.glob(str(W / 'runs' / '*_s[0-9]*'))):
        m = re.match(r'(min|mid|max)_(.+)_s(\d+)$', Path(d).name)
        if not m: continue
        rows = []
        for f in glob.glob(f'{d}/shard*/task_results.jsonl'):
            rows += [json.loads(l) for l in open(f) if l.strip()]
        if rows:
            out.setdefault((m.group(1), m.group(2)), []).append(sum(bool(r.get('resolved')) for r in rows) / len(rows))
    return out


def fig1(summary, seeds):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs, labels = [], []
    rows = []
    for i, alias in enumerate(ORDER):
        for j, dose in enumerate(DOSES):
            key = f'{dose}|{alias}'
            if key not in summary['cells']: continue
            x = i * 4 + j; xs.append(x); labels.append(f'{CAT[alias][0]}\n{dose}')
            pts = seeds.get((dose, alias), [])
            ax.scatter([x] * len(pts), pts, s=28, color=CAT[alias][1], alpha=0.45, edgecolor=SURF, linewidth=1, zorder=2)
            mean = summary['cells'][key]['resolved_rate']
            ax.hlines(mean, x - 0.35, x + 0.35, color=CAT[alias][1], linewidth=2, zorder=3)
            ax.text(x, mean + 0.025, f'{mean:.2f}', ha='center', va='bottom', fontsize=8, color=INK)
            rows.append({'model': CAT[alias][0], 'dose': dose, 'mean_rate': mean, 'seed_rates': pts})
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel('Resolved fraction (104 tasks)'); ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
    ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    ax.set_title('Resolution by capacity x scaffold level (points = seeds, bar = mean)', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig1_resolution.png', dpi=200); plt.close(fig)
    with open(OUT / 'fig1_resolution.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'dose', 'mean_rate', 'seed_rates']); w.writeheader(); w.writerows(rows)


def fig2(summary):
    cells = [(d, a) for a in ORDER for d in DOSES if f'{d}|{a}' in summary['cells']]
    fig, ax = plt.subplots(figsize=(6.6, 3.9))
    rows = []
    for k, (d, a) in enumerate(cells):
        dist = summary['cells'][f'{d}|{a}']['ladder_dist']; tot = sum(dist.values()) or 1
        left = 0
        for L in range(6):
            frac = dist[f'L{L}'] / tot
            ax.barh(k, frac, left=left, color=SEQ[L], edgecolor=SURF, linewidth=1.5, height=0.62)
            if frac >= 0.08: ax.text(left + frac / 2, k, f'L{L}', ha='center', va='center', fontsize=7.5, color=INK if L < 3 else SURF)
            left += frac
        mean_lvl = sum(L * dist[f'L{L}'] for L in range(6)) / tot
        ax.text(1.01, k, f'{mean_lvl:.2f}', va='center', fontsize=7.5, color=INK2)
        rows.append({'model': CAT[a][0], 'dose': d, **{f'L{L}': dist[f'L{L}'] for L in range(6)}})
    ax.set_yticks(range(len(cells))); ax.set_yticklabels([f'{CAT[a][0]} {d}' for d, a in cells]); ax.invert_yaxis()
    ax.set_xlim(0, 1); ax.set_xlabel('Share of runs (all seeds x 104 tasks)')
    ax.set_title('Failure ladder per cell (number at right = mean level)', loc='left', fontsize=10)
    from matplotlib.patches import Patch
    names = ['L0 no patch', 'L1 patch', 'L2 right file', 'L3 tests collect', 'L4 a test passes', 'L5 resolved']
    ax.legend(handles=[Patch(color=SEQ[L], label=names[L]) for L in range(6)], ncol=3, fontsize=7, frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.22))
    fig.tight_layout(); fig.savefig(OUT / 'fig2_ladder.png', dpi=200); plt.close(fig)
    with open(OUT / 'fig2_ladder.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)


def fig3(summary):
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    rows = []
    for i, a in enumerate(ORDER):
        c = summary['contrasts'].get(a)
        if not c: continue
        lo, hi = c.get('ci_lo'), c.get('ci_hi')
        y = len(ORDER) - 1 - i
        if lo is not None and hi is not None:
            ax.hlines(y, lo, hi, color=CAT[a][1], linewidth=2)
        ax.scatter([c['risk_diff']], [y], s=46, color=CAT[a][1], edgecolor=SURF, linewidth=1.2, zorder=3)
        ax.text(c['risk_diff'], y + 0.22, f"{c['risk_diff']:+.3f}", ha='center', fontsize=8)
        rows.append({'model': CAT[a][0], 'delta': c['risk_diff'], 'ci_lo': lo, 'ci_hi': hi, 'p_holm': c.get('p_holm')})
    ax.axvline(0, color=INK2, linewidth=0.8); ax.axvspan(-0.05, 0.05, color=GRID, alpha=0.6, zorder=0)
    ax.set_yticks([len(ORDER) - 1 - i for i in range(len(ORDER))]); ax.set_yticklabels([CAT[a][0] for a in ORDER])
    xs = [v for r in rows for v in (r['delta'], r['ci_lo'], r['ci_hi']) if v is not None]
    span = max(0.08, max(abs(x) for x in xs) * 1.25 if xs else 0.08); ax.set_xlim(-span, span)
    ax.set_xlabel('Δ = mean over tasks of p(max) − p(min)'); ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_title('Δ = max − min, 95% bootstrap CI (grey: ±0.05)', loc='left', fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / 'fig3_delta.png', dpi=200); plt.close(fig)
    with open(OUT / 'fig3_delta.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['model', 'delta', 'ci_lo', 'ci_hi', 'p_holm']); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    summary = json.load(open(W / 'analysis' / 'summary.json'))
    seeds = seed_rates()
    fig1(summary, seeds); fig2(summary)
    if summary.get('contrasts'): fig3(summary)
    print('figures written to', OUT)
