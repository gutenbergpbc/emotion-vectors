# Behavioral emotion-denial experiment

Independent Gemma 27B and GPT-OSS API generation/judging scripts compare the
original named-emotion and general-emotion denial prompts in Coding and Sales.
The original experiment is not a Gemma 4B result.

`gemma27b/run.py` and `gptoss120b/run.py` contain exact treatment prompts, model,
sampling settings, source scenario configs, output root and deterministic job
assignment. Each defaults to 100 traces per arm/task. `judge.py` applies the
recorded RH rubric; Gemma's `control.py` supplies the session-label control.
Baseline judgments are separate (`judge_control.py` / `judge_baseline.py`).

These scripts consume scenarios from archived steering configs in `runs/` and
use the shared OpenRouter generator in `emotion-rh-forecasting/`. They require
`OPENROUTER_API_KEY`. Before new paid collection set `OUT` to a new run directory;
do not modify the supplied archived run-spec or its checksums. The existing
source constants make the exact experimental setup inspectable by an agent.

Published result tables are downloadable under `results/`. Historical GPT-OSS
Coding baseline and treatments differ in token budget and reasoning mode; keep
that limitation with comparisons. `figure_prompt_rates.py` retains original
plot conventions. See [the script reference](../../../docs/scripts.md).
