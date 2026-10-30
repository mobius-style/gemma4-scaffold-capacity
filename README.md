# Scaffolding × capacity for local Gemma 4 SWE agents — data and code

This repository holds the materials behind the Kaggle Paper Track writeup *"Scaffolding Helps the Smallest Gemma 4 Agent, Not the Larger Ones"*: a pre-registered factorial of 3 Gemma 4 capacities (E4B, 12B, 26B-A4B) × 2 scaffold levels (min, max) × 5 seeds on 104 public tasks of the Gemma 4 Developer Agent competition.

**Links.** GitHub: https://github.com/mobius-style/gemma4-scaffold-capacity · Zenodo archive: DOI_PENDING · Kaggle Dataset: https://www.kaggle.com/datasets/taikotoeda/gemma4-scaffold-capacity · Kaggle Notebook: https://www.kaggle.com/code/taikotoeda/gemma4-scaffold-capacity-reproduce

The confirmatory numbers can be checked on CPU in seconds:

```bash
python3 scripts/reproduce.py   # Python >= 3.10, standard library only; prints REPRODUCE: PASS/FAIL (3.12 matches summary.json exactly; 3.10/3.11 differ by <=1e-17 in estimates and a few thousandths in bootstrap p, see the script header)
```

`reproduce.py` runs the frozen `scripts/analyze.py` — paired task bootstrap, B = 2000, seed 20261022, Holm correction and TOST — on the released run records. It then compares every estimate, interval, p-value and per-cell rate with `analysis/summary.json`, and exits non-zero on any mismatch.

## What is here

| Path | Content |
|---|---|
| `PREREGISTRATION.md`, `freeze/` | The design as frozen before the first scored run (git commit 6fac761, 2026-10-22 23:04 JST), the SHA-256 manifest, and two addenda. The timestamp is self-attested by the local git history; addendum 2 (6a367a5) was written before any confirmatory inference. |
| `DEVIATIONS.md`, `DEVIATIONS_auto.md`, `logs/schedule.log` | Every departure from the registration, D1–D11, including host-reset restarts and post-result analyses, plus the raw scheduler log. |
| `TASKSET.txt`, `data/gold_empty_verification.jsonl` | The 104 scorable tasks, and the gold-patch / empty-patch verdict for each of the 129 public tasks, taken from the real scoring path (`verify_task`). |
| `data/runs.jsonl` | One line per task-run (4,056: 3,120 confirmatory and 936 exploratory mid-level runs, `level` = min / mid / max). Each line gives the outcome, ladder level (L0–L5), end reason (resolved, time cap, tool-call cap, …), time, tool and LLM calls, tool-use counts, whether `submit_patch` was called, and the SHA-256 of the agent patch. |
| `analysis/*.json` | Outputs of `analyze.py` (confirmatory), `exploratory.py`, `secondary.py` and `diagnostics.py`. |
| `scripts/` | Analysis, figures, the number gate (`check_numbers.py`), the harness wrapper (`eval_cli.py`: deviations D1/D2/D5), the cell runner, the schedulers, and this release builder. |
| `prompts/`, `configs/` | The two system instructions written for this study, and the per-level agent, sampling and budget configs. |
| `grader/` | The grader repair: the wheel set used, and the lines added to the competition sandbox Dockerfile. |
| `models/SHA256SUMS` | SHA-256 of the three GGUF files and their Hugging Face sources. |
| `paper/` | The writeup text and figures, with the CSV behind each figure. |

## Not included, by design

- **No competition data.** There are no problem statements, reference or test patches, repository snapshots, agent patches, traces or test logs. Competition participants have the task data and the harness.
- **No files from the competition's sample submission.** The analyzer sub-agent's prompt and config were used unchanged from the sample and are referenced, not copied.
- **No model weights.** Download them from the Hugging Face repositories listed in `models/SHA256SUMS`.

## Re-running the experiment

You need the competition data, the `swegemma` harness wheelhouse, llama.cpp (commit 5a4d0fe) built with CUDA, and the three GGUF files. The steps are:

1. Rebuild the sandbox image with `grader/Dockerfile.delta`.
2. Assemble the wheel set listed in `grader/wheels_repaired.txt`.
3. Start one `llama-server` per GPU with `run_server.sh`.
4. Run `scripts/run_cell.sh <level> <model> <seed>`.

A 26B cell took about 8.5 hours on one RTX 5070 Ti.

## License

Apache-2.0 (see `LICENSE`). The design, code and analysis were drafted with an AI assistant (Claude); the author reviewed every step.
