# PREREGISTRATION — Scaffolding dose × model capacity for local Gemma 4 SWE agents

Frozen: see FREEZE.sha256 / FREEZE.timestamp (written immediately after this file; no confirmatory run starts before that timestamp). Author: Taiko Toeda (design, analysis plan and code drafted with Claude; the human author reviewed and is responsible).

## 0. Question
Does scaffolding (system instruction + tool affordances) change the resolution rate of local Gemma 4 agents on real Python bug-fix tasks, and does the effect depend on model capacity? Prior (Toeda, dose × capacity grid, July 2026): interventions are harmful or null below ~4B effective parameters and helpful above.

## 1. Confirmatory hypotheses (signs declared; α = 0.05; Holm over m = 4)
Let p_d(t) be the fraction of seeds (R = 5) in which task t is resolved under dose d, and Δ_M = mean_t[p_max(t) − p_min(t)] for model M over the task set.
- H1a (26B-A4B): Δ_26B > 0 (one-sided).
- H1b (12B): Δ_12B > 0 (one-sided).
- H1c (E4B): Δ_E4B ≤ 0; tested as Δ_E4B < 0 (one-sided); if not rejected, equivalence within ±Δmin is tested by TOST (90% CI inside [−Δmin, +Δmin]).
- H2 (interaction): Δ_26B − Δ_E4B > 0 (one-sided). Secondary ordered check: the slope of Δ over capacity rank (E4B = 1, 12B = 2, 26B = 3) > 0.
- Minimal important difference Δmin = 0.05 (≈ 5 tasks of 104). MDE (power 0.80, two-sided 0.05, observed flip rate ψ = 0.133 from the E4B pilot, N = 104, R = 5): 0.045 (τ² = 0) to 0.065 (τ² = 0.03); interaction MDE ≈ 0.064–0.092. Effects below Δmin are not claimed in either direction.
- Rejection conditions: H1a/H1b are rejected if the one-sided Holm-adjusted bootstrap p ≥ 0.05; H1c is "not supported" if Δ_E4B > 0 with adjusted p < 0.05 (that is a positive finding against the prior and is reported as such); H2 is rejected if its adjusted p ≥ 0.05. A fully null outcome (no hypothesis rejected and TOST not passed) is reported as a negative result with MDE and CIs.

## 2. Design
- Factor A, scaffolding dose (confirmatory levels: **min**, **max**; exploratory: **mid**):
  - min: one-sentence instruction (subs/prompts/system_min.md) + tools {run_command, read_file, edit_file, write_file, get_status, submit_patch}.
  - mid: agent-optimised instruction (subs/prompts/system_full.md, 9 rules) + the same 6 tools.
  - max: mid + {get_code_neighbors, search_similar_code, get_code_subgraph} + code_analyzer sub-agent (agent_tool, prompts/analyzer.md from the competition sample).
- Factor B, capacity: gemma-4-E4B / gemma-4-12B / gemma-4-26B-A4B, Google QAT q4_0 GGUF (sha256 in logs/model_sha256_{e4b,12b,26b}.txt), served by llama-server (llama.cpp 0.4.0-dev commit 5a4d0fe, CUDA) with --jinja, ctx 32768, reasoning budget 4096, 1 slot per GPU, one server per GPU (GPU0 :8000, GPU1 :8001), tasks sharded deterministically (task_idx % 2).
- Sampling (configs/sampling.yaml): temperature 1.0, top_p 0.95, max_output_tokens 16384, thinking_budget 4096, include_thoughts true, seed s ∈ {1,…,5}.
- Harness: swegemma 0.2.11 / adk-submission 0.2.13 / google-adk 1.39.1 / litellm 1.106.2 (venv/), budgets max_time_minutes 6, max_tool_calls 50, max_turns 80, command timeout 150 s, verification timeout 300 s, Docker sandbox swebench-sandbox:repaired2 with host wheel set data_repaired/wheels (manifest in FREEZE.sha256).
- Repeats: confirmatory cells 3 capacities × 2 doses × R = 5 seeds = 30 runs; mid cells R = 3 (seeds 1–3), exploratory. Run order: for seed in 1..5, for model in (26B, 12B, E4B), doses (min, max[, mid]) — scripts/schedule.sh. No run is added, dropped or re-run after the freeze; a run that aborts for infrastructure reasons (server crash, Docker failure) is re-run once from scratch for the same seed and the event is logged in DEVIATIONS.md.
- Expected cost (timing pilots, mid scaffold): E4B 68 s, 26B 131 s, 12B 150 s per task → ≈ 70 GPU-pair hours.

## 3. Task set (frozen)
- TASKSET.txt: 104 of the 129 public tasks (fastapi 60, rich 39, requests 5), sha256 in FREEZE.sha256. Scorable = gold patch resolved AND empty patch not resolved on the real scoring path (swegemma.harness.verification.verify_task; scripts/gold_verify.py; logs/gold_verify_repaired.jsonl). Excluded 25: requests ×8 and httpx_3672 (environment: network=none / wheel shadowing), fastapi_14851 and fastapi_14077 (empty patch resolves), fastapi_14186 and rich_3486 (required tests cannot pass), fastapi ×4 and rich ×8 needing period-correct dependency pins (starlette / Pygments).
- Grader repair (reported in the paper): drop starlette ≥ 1.0 and the four repo-under-test wheels from the host wheel set; add test wheels from the public Kirchner dataset (typing_inspection, dirty_equals, inline_snapshot, httpbin, …); bake pytest-httpbin into the image; isolate TMPDIR per wheel set because the harness caches unpacked wheels.
- Pilot data taken before the freeze (exploratory, reported separately, tasks NOT excluded): E4B mid on 30 tasks × seeds 1–2 (runs/pilot30_ids.txt); E4B min/max on 4 tasks; 26B and 12B mid on 10 tasks (runs/timing10_ids.txt). Prompts were written before any scored run and are not edited after the freeze.

## 4. Outcomes
- Primary: resolved (official verify_task: pytest exit 0 and every required test node passed).
- Secondary ordinal ladder per run, derived mechanically by scripts/analyze.py from the harness artifacts: L0 no patch / L1 non-empty patch / L2 patch touches ≥ 1 file modified by the reference patch / L3 patch applies and tests collect / L4 ≥ 1 test passes / L5 resolved.
- Cost: tool calls, LLM calls, duration, tokens (traces).

## 5. Analysis (scripts/analyze.py is the single source of all numbers; frozen with this file; any later change is logged with before/after numbers)
- Primary inference: paired bootstrap over tasks (B = 2000, BCa if available, else percentile): Δ_M with 95% CI and one-sided p for H1a–H1c; H2 as the bootstrap distribution of Δ_26B − Δ_E4B (same task resample); Holm over the 4 tests. Descriptive: exact McNemar on majority-of-5 resolution per model.
- Run-level model (secondary): GEE logistic regression resolved ~ dose × capacity with exchangeable correlation within task (statsmodels 0.15.0); reported with cluster-robust CIs.
- Exploratory (BH): mid vs min/max; ladder level (paired sign test on per-task mean level); cost; per-repository rates; flip rate per cell (fraction of tasks with mixed outcomes across seeds).
- Negative result: TOST with ±Δmin; report MDE; report flip rates.
- Contamination (G5): Gemma 4 pre-training cutoff January 2025 (model card). Task created_at: 21 tasks ≤ 2025-01, 83 tasks ≥ 2025-02. Report all contrasts in both strata; report 13-gram overlap between each agent patch and the reference patch (memorisation screen); conclusions rest on the post-cutoff stratum if the strata disagree in sign.

## 6. Pre-declared limitations and the two most painful objections
1. Tuning asymmetry: the min prompt is deliberately minimal (one sentence); the full prompt was written once on 2026-10-22 from the competition sample and the author's prior agent-prompt study, without iteration on scored runs. We report the pilot runs that preceded the freeze.
2. Single benchmark family (four public repositories), single machine, quantised GGUF rather than the competition's w4a16-ct checkpoints, 31B absent (hardware), one harness configuration (6 min / 50 calls). Generalisation claims are limited accordingly.
