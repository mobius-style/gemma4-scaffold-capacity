#!/usr/bin/env python3
"""Recompute the paper's confirmatory numbers from the released run records (data/runs.jsonl) on CPU, with the
frozen analysis functions (scripts/analyze.py), and compare them with analysis/summary.json. Exit 0 only if every
confirmatory estimate, CI and p-value matches within the stated tolerance and every per-cell rate matches exactly.
Usage (from the release root): python scripts/reproduce.py
"""
import json, sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
import analyze  # noqa: E402


def main():
    cells = defaultdict(lambda: defaultdict(dict))
    for l in open(ROOT / 'data' / 'runs.jsonl'):
        r = json.loads(l)
        cells[(r['level'], r['model'])][r['seed']][r['instance_id']] = r
    s = analyze.summarize(cells)
    s['confirmatory'] = analyze.confirmatory(cells)
    ref = json.load(open(ROOT / 'analysis' / 'summary.json'))
    # Tolerances: the frozen code sums floats with the built-in sum(); Python >= 3.12 uses compensated (Neumaier)
    # summation, older interpreters do not, so estimates can differ by ~1e-17 and a bootstrap resample whose mean is
    # numerically zero can fall on either side of 0 (one-sided p differs by a few thousandths). A changed record moves
    # an estimate by >= 1/520 ~ 0.0019, far above these tolerances. Python 3.12 reproduces analysis/summary.json exactly.
    TOL = {'delta': 1e-9, 'ci': 1e-9, 'tost_90ci': 1e-9, 'p_one_sided': 0.01, 'p_holm': 0.02}
    def close(a, b, tol):
        if isinstance(a, (list, tuple)): return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
        return a is None and b is None or (a is not None and b is not None and abs(a - b) <= tol)
    bad = []
    for k, t in ref['confirmatory']['tests'].items():
        got = s['confirmatory']['tests'][k]
        for f, tol in TOL.items():
            if f in t and not close(t[f], got[f], tol): bad.append((k, f, t[f], got[f]))
    for k, c in ref['cells'].items():
        for f in ('resolved_rate', 'flip_rate', 'ladder_dist'):
            if json.dumps(c[f]) != json.dumps(s['cells'][k][f]): bad.append((k, f, c[f], s['cells'][k][f]))
    for k, t in s['confirmatory']['tests'].items():
        print(f"{k:28s} delta {t['delta']:+.3f}  95% CI [{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]  p_one_sided {t['p_one_sided']:.4f}  Holm {t['p_holm']:.2f}")
    print('MISMATCHES:', bad if bad else 'none')
    print('REPRODUCE:', 'PASS' if not bad else 'FAIL')
    sys.exit(1 if bad else 0)


if __name__ == '__main__':
    main()
