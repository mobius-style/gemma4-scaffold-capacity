#!/usr/bin/env python3
"""EXPLORATORY diagnostics requested by review round 1 (2026-10-29; written after the confirmatory results were seen).
Per cell, from task_results.jsonl and traces/trace_<iid>.json:
- end reason: resolved / time cap / tool-call cap / turn cap / other error / submitted-unresolved
- share of task-runs with no submit_patch call; with zero-size patch
- max-arm component use: graph tools, analyzer sub-agent calls (any non-workspace tool name is counted by name)
- task-runs touched by a D5 parser conversion or a D2 malformed-argument event (from the trace), and the 12B Delta
  recomputed with those task-runs excluded (sensitivity)
- pairwise two-seed disagreement (comparable with a single re-run): mean over seed pairs of share of tasks differing
- 26B throughput by GPU layout: mean LLM calls per minute of agent time
Output: analysis/diagnostics.json (+ stdout).
"""
import itertools, json, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze  # noqa: E402

W = analyze.W
R = W / 'analysis' / 'confirmatory_runs'
PEG = '[harness] The previous model output could not be parsed'
WORKSPACE = {'run_command', 'read_file', 'edit_file', 'write_file', 'get_status', 'submit_patch'}
LAYOUT_26B = {1: 'dual', 2: 'dual', 3: 'single', 4: 'single', 5: 'single'}


def trace_info(path):
    info = {'tools': Counter(), 'submit': False, 'peg': 0, 'malformed': 0}
    if not path.exists(): info['missing'] = True; return info
    try: t = json.loads(path.read_text(errors='ignore'))
    except Exception: info['unreadable'] = True; return info
    for st in t.get('steps', []):
        if st.get('source') == 'agent' and PEG in (st.get('message') or ''): info['peg'] += 1
        for tc in st.get('tool_calls') or []:
            n = tc.get('function_name'); info['tools'][n] += 1
            if n == 'submit_patch': info['submit'] = True
            if '__malformed_arguments__' in json.dumps(tc.get('arguments') or {}): info['malformed'] += 1
    return info


def reason(r):
    e = r.get('error') or ''
    if r.get('resolved'): return 'resolved'
    if 'session timeout' in e: return 'time_cap'
    if 'tool call budget' in e: return 'tool_cap'
    if 'turns budget' in e: return 'turn_cap'
    if e: return 'other_error'
    return 'submitted_unresolved'


def main():
    cells, tasks = analyze.load_runs(R)
    out = {'cells': {}, 'sensitivity_12b_excl_parser_events': None}
    flagged = defaultdict(set)  # (dose, alias, seed) -> iids with peg/malformed
    for (dose, alias), seeds in sorted(cells.items()):
        n = 0; capn = 0; cap_nosub = 0; rs = Counter(); nosub = 0; zero = 0; tools = Counter(); peg_runs = 0; mal_runs = 0; missing = 0
        for s, recs in seeds.items():
            for iid, r in recs.items():
                n += 1; rs[reason(r)] += 1; zero += (r.get('agent_patch_size') or 0) <= 0
                tp = next(iter(sorted(R.glob(f'{dose}_{alias}_s{s}/shard*/traces/trace_{iid}.json'))), Path('/nonexistent'))
                ti = trace_info(tp); missing += bool(ti.get('missing') or ti.get('unreadable'))
                nosub += not ti['submit']; tools.update(ti['tools'])
                if (r.get('error') or '').startswith('Agent exceeded'): capn += 1; cap_nosub += not ti['submit']
                if ti['peg']: peg_runs += 1
                if ti['malformed']: mal_runs += 1
                if ti['peg'] or ti['malformed']: flagged[(dose, alias, s)].add(iid)
        pair = [sum(bool(seeds[a][i].get('resolved')) != bool(seeds[b][i].get('resolved')) for i in set(seeds[a]) & set(seeds[b])) / len(set(seeds[a]) & set(seeds[b]))
                for a, b in itertools.combinations(sorted(seeds), 2)]
        out['cells'][f'{alias}/{dose}'] = {
            'task_runs': n, 'end_reason': {k: v / n for k, v in rs.items()}, 'no_submit_patch': nosub / n, 'zero_size_patch': zero / n,
            'traces_missing_or_unreadable': missing, 'tool_calls_per_run': {k: v / n for k, v in tools.most_common()},
            'non_workspace_tools_per_run': sum(v for k, v in tools.items() if k not in WORKSPACE) / n,
            'runs_with_parser_conversion': peg_runs, 'runs_with_malformed_args': mal_runs,
            'pairwise_two_seed_disagreement': sum(pair) / len(pair), 'capped_runs': capn, 'capped_without_submit': (cap_nosub / capn) if capn else None}
    # 12B sensitivity: drop task-runs with any parser/malformed event, recompute per-task p over remaining seeds
    alias = 'gemma-4-12b-it'
    lo, hi = cells[('min', alias)], cells[('max', alias)]
    d = {}
    for i in sorted(set.intersection(*[set(v) for v in list(lo.values()) + list(hi.values())])):
        a = [bool(lo[s][i].get('resolved')) for s in lo if i not in flagged[('min', alias, s)]]
        b = [bool(hi[s][i].get('resolved')) for s in hi if i not in flagged[('max', alias, s)]]
        if a and b: d[i] = sum(b) / len(b) - sum(a) / len(a)
    import exploratory
    out['sensitivity_12b_excl_parser_events'] = exploratory.boot(d)
    out['sensitivity_12b_excl_parser_events']['excluded_task_runs'] = {'min': sum(len(flagged[('min', alias, s)]) for s in lo), 'max': sum(len(flagged[('max', alias, s)]) for s in hi)}
    # Conservative variant: attribute each D5/D2 stderr line in the sequential shard log to every task that could have
    # been running (after the last task with a logged end, up to and including the next one), since tasks that end
    # normally write no line. Over-excludes by design; traces miss most conversions.
    import re as _re
    win = defaultdict(set)
    for (dose, alias_), seeds in cells.items():
        if alias_ != alias: continue
        for s in seeds:
            for sh in sorted(x for x in R.glob(f'{dose}_{alias}_s{s}/shard*') if x.is_dir()):
                order = [json.loads(l)['instance_id'] for l in open(sh / 'task_results.jsonl') if l.strip()]
                pos = {iid: k for k, iid in enumerate(order)}
                lines = Path(f'{sh}.log').read_text(errors='ignore').splitlines()
                ends = [(n, pos[m.group(1)]) for n, l in enumerate(lines) for m in [_re.match(r'Task (\S+?)[: ]', l)] if m and m.group(1) in pos]
                for n, l in enumerate(lines):
                    if 'peg-gemma4 parse failure' in l or 'malformed tool-call arguments' in l:
                        prev = max([k for ln, k in ends if ln < n], default=-1)
                        nxt = min([k for ln, k in ends if ln > n and k > prev], default=len(order) - 1)
                        win[(dose, s)].update(order[prev + 1:nxt + 1])
    d2 = {}
    for i in sorted(set.intersection(*[set(v) for v in list(lo.values()) + list(hi.values())])):
        a = [bool(lo[s][i].get('resolved')) for s in lo if i not in win[('min', s)]]
        b = [bool(hi[s][i].get('resolved')) for s in hi if i not in win[('max', s)]]
        if a and b: d2[i] = sum(b) / len(b) - sum(a) / len(a)
    out['sensitivity_12b_excl_parser_window'] = exploratory.boot(d2)
    out['sensitivity_12b_excl_parser_window']['excluded_task_runs'] = {k: sum(len(win[(k, s)]) for s in lo) for k in ('min', 'max')}
    out['sensitivity_12b_excl_parser_window']['tasks_kept'] = len(d2)
    # context overflows (log lines): every event writes one 'Non-retryable model error' line; analyzer-raised ones add a 'Tool ... raised' line
    ov = {}
    for (dose, alias_), seeds in cells.items():
        txt = ''.join(Path(f).read_text(errors='ignore') for s in seeds for f in R.glob(f'{dose}_{alias_}_s{s}/shard*.log'))
        ov[f'{alias_}/{dose}'] = {'events': txt.count('Non-retryable model error: ContextWindowExceededError'), 'in_analyzer': txt.count("Tool 'code_analyzer_agent' raised ContextWindowExceededError")}
    out['context_overflow'] = ov
    # capped runs without submit_patch (share)
    # 26B throughput by layout
    thr = {}
    for dose in ('min', 'max'):
        for s, recs in cells[(dose, 'gemma-4-26b-a4b-it')].items():
            rows = list(recs.values())
            calls = sum(r.get('total_llm_calls') or 0 for r in rows); mins = sum(r.get('duration_seconds') or 0 for r in rows) / 60
            thr[f'{dose}/s{s}'] = {'layout': LAYOUT_26B[s], 'llm_calls_per_min': calls / mins, 'cap_time': sum('session timeout' in (r.get('error') or '') for r in rows),
                                   'cap_tool': sum('tool call budget' in (r.get('error') or '') for r in rows)}
    out['throughput_26b_by_layout'] = thr
    json.dump(out, open(W / 'analysis' / 'diagnostics.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
