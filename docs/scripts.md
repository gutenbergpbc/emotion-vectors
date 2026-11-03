# Script reference

Run scripts from the repository root with `uv run python PATH`. Each experiment
README identifies the recommended entrypoints. The Gemma walkthrough selects
the files needed for that model; historical scheduler and recovery helpers
retain explicit constants and should be inspected before use.

This reference is generated without importing code. Input/output constants
are source expressions, not assertions that their files already exist.
Paths into local `runs/` may require a prior independent experiment. GPU
and API scripts do not necessarily implement a no-cost `--help`/dry-run;
use the documented primary entrypoints for workload previews.

## `src/experiments/emotion-denial/gemma27b/control.py`

Run the session-label system-message control under the frozen Gemma harness.

## `src/experiments/emotion-denial/gemma27b/figure_prompt_rates.py`

Render the Gemma 27B prompt comparison from saved result JSON.

Main interfaces: `read`, `values`, `main`.

Input/output settings:

```python
ROOT = _REPO
OUT = ROOT / 'runs/gemma27-session-label-control-20261026/figures'
```

## `src/experiments/emotion-denial/gemma27b/judge.py`

Score completed cohorts with the original whole-trace RH rubric.

Main interfaces: `read`, `digest`, `score`, `main`.

Input/output settings:

```python
ROOT = _REPO
OUT = ROOT / 'runs/gemma27-emotion-denial-twoarms-20261026'
```

## `src/experiments/emotion-denial/gemma27b/judge_control.py`

Judge the session-label control with the same whole-trace rubric.

## `src/experiments/emotion-denial/gemma27b/run.py`

Resumable Gemma 27B OpenRouter emotion-denial cohorts.

Main interfaces: `sha`, `SamplingSession`, `one`, `main`.

Input/output settings:

```python
ROOT = _REPO
SOURCE = ROOT / 'src/experiments/emotion-rh-forecasting/generate_openrouter_traces.py'
OUT = ROOT / 'runs/gemma27-emotion-denial-twoarms-20261026'
```

## `src/experiments/emotion-denial/gptoss120b/figure_prompt_rates.py`

Draw GPT-OSS prompt comparison from the saved judge summaries.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = _REPO
OUT = RESULT / 'figures'
```

## `src/experiments/emotion-denial/gptoss120b/judge.py`

Score completed cohorts with the original whole-trace RH rubric.

Main interfaces: `read`, `digest`, `score`, `main`.

Input/output settings:

```python
ROOT = _REPO
OUT = ROOT / 'runs/gptoss120b-emotion-denial-lowreason-20261027'
```

## `src/experiments/emotion-denial/gptoss120b/judge_baseline.py`

Judge all saved unsteered GPT-OSS sales and coding traces; no generation.

Main interfaces: `digest`, `read`, `main`.

Input/output settings:

```python
ROOT = _REPO
OUT = ROOT / 'runs/gptoss120b-emotion-denial-lowreason-20261027'
```

## `src/experiments/emotion-denial/gptoss120b/run.py`

Resumable GPT-OSS 120B OpenRouter emotion-denial cohorts.

Main interfaces: `sha`, `SamplingSession`, `one`, `main`.

Input/output settings:

```python
ROOT = _REPO
SOURCE = ROOT / 'src/experiments/emotion-rh-forecasting/generate_openrouter_traces.py'
OUT = ROOT / 'runs/gptoss120b-emotion-denial-lowreason-20261027'
```

## `src/experiments/emotion-pressure/build_report.py`

Render all emotion results as a standalone interactive HTML and twelve-page PDF.

Arguments: `root`, `--model`.

Main interfaces: `build`, `pdf`.

Input/output settings:

```python
LABELS = ['No instruction', 'Named emotions', 'General emotions']
```

## `src/experiments/emotion-pressure/check_viewer.py`

Check the completed report in a real browser and inspect PDF coverage.

Arguments: `root`.

Main interfaces: `check`.

## `src/experiments/emotion-pressure/pressure_core.py`

Frozen three-condition pressure experiment: no generation or steering.

Main interfaces: `sha`, `write_json`, `append_json`, `bundle`, `messages_for`, `render`, `summarize`, `plots`, `verify_rows`, `run_worker`.

## `src/experiments/emotion-pressure/run.py`

Replay the paper's frozen pressure histories on a local GPU.

Inputs: data/pressure/<model>.json, including original denial wording,
twenty scenarios, selected probes, revisions and archived baseline token IDs.
Outputs: measurements, local residuals, runtime and verification in --output.
This experiment makes forward passes only. It never generates task responses.

Arguments: `--inputs`, `--model`, `--output`, `--dry-run`.

Main interfaces: `main`.

## `src/experiments/emotion-pressure/verify.py`

Recheck pressure-run residuals and statistics: verify.py RUN --model MODEL.
Residuals are runtime intermediates, deliberately absent from release archives.

Arguments: `run`, `--model`.

## `src/experiments/emotion-probes/adapters/gemma27b/modal_worker.py`

Run an existing prepared directory on one H200 and mirror progress locally.

Example:
  uv run --extra modal modal run modal_worker.py --run-dir /absolute/run --through report
HF_TOKEN uses an ephemeral Modal secret when provided locally; otherwise the named huggingface secret.
The persistent volume holds run data; model weights use a separate cache volume.

Main interfaces: `worker`, `transport_guard`, `main`.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/__init__.py`

Frozen twelve-emotion replication on Gemma 3.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/cli.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Arguments: `--run`, `--smoke-only`, `--host`, `--port`, `--watch`, `--interval`, `--bundle`, `--model`, `--revision`, `--attention`, `--max-tokens`, `--shard-size`.

Main interfaces: `resolve_revision`, `prepare`, `synthetic_features`, `parser`, `main`.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/common.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `require`, `read`, `sha`, `digest`, `atomic`, `now`, `source_hashes`, `validate_inputs`, `Run`.

Input/output settings:

```python
LABELS = EMOTIONS + ['neutral']
```

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/evaluation.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `evaluate`.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/extraction.py`

Correctness-first CUDA extraction, batch size one, resumable checked shards.

Main interfaces: `identity`, `validate_shard`, `feature_manifest`, `layer_features`, `prepare_tokens`, `extract`, `capture_case`.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/fitting.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `negative_conditions`, `direction`, `winning_layer`, `fit`.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/rendering.py`

Gemma-native roles; explicit text representation of supplied tool/reasoning streams.

No input case is edited. System text is prefixed to the first user turn; tool
messages become user turns with a visible wrapper; adjacent same-role turns
join with two newlines. Saved reasoning is ordinary assistant text in <think>.
Assistant-first histories receive an empty user turn (no added prose).
Both training and evaluation use this identical adapter. Marker round-trip
checks establish exact character offsets even for repeated target strings.

Main interfaces: `render_case`, `tokenize_case`.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/reporting.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `verify`, `report`.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/server.py`

Read-only, loopback-default dashboard server; only allowlisted artifacts served.

Main interfaces: `handler_for`, `serve`.

## `src/experiments/emotion-probes/adapters/gemma27b/probe_harness/statistics.py`

Condition-weighted metrics, deterministic cutoff selection and grouped intervals.

Main interfaces: `condition_weights`, `measure`, `candidates`, `choose_threshold`, `verify_threshold`, `bootstrap_setup`, `bootstrap`.

## `src/experiments/emotion-probes/adapters/gptoss120b/modal_worker.py`

Run an existing prepared directory on one H200 and mirror progress locally.

Example:
  uv run --extra modal modal run modal_worker.py --run-dir /absolute/run --through report
HF_TOKEN uses an ephemeral Modal secret when provided locally; otherwise the named huggingface secret.
The persistent volume holds run data; model weights use a separate cache volume.

Main interfaces: `worker`, `transport_guard`, `main`.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/__init__.py`

Frozen twelve-emotion replication on Gemma 3.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/cli.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Arguments: `--run`, `--smoke-only`, `--host`, `--port`, `--watch`, `--interval`, `--bundle`, `--model`, `--revision`, `--attention`, `--max-tokens`, `--shard-size`.

Main interfaces: `resolve_revision`, `prepare`, `synthetic_features`, `parser`, `main`.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/common.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `require`, `read`, `sha`, `digest`, `atomic`, `now`, `source_hashes`, `validate_inputs`, `Run`.

Input/output settings:

```python
LABELS = EMOTIONS + ['neutral']
```

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/evaluation.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `evaluate`.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/extraction.py`

Correctness-first CUDA extraction, batch size one, resumable checked shards.

Main interfaces: `identity`, `validate_shard`, `feature_manifest`, `layer_features`, `prepare_tokens`, `extract`, `capture_case`.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/fitting.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `negative_conditions`, `direction`, `winning_layer`, `fit`.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/rendering.py`

Versioned Harmony replay: native channels, preserved text, frozen date.

Main interfaces: `render_case`, `tokenize_case`.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/reporting.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `verify`, `report`.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/server.py`

Read-only, loopback-default dashboard server; only allowlisted artifacts served.

Main interfaces: `handler_for`, `serve`.

## `src/experiments/emotion-probes/adapters/gptoss120b/probe_harness/statistics.py`

Condition-weighted metrics, deterministic cutoff selection and grouped intervals.

Main interfaces: `condition_weights`, `measure`, `candidates`, `choose_threshold`, `verify_threshold`, `bootstrap_setup`, `bootstrap`.

## `src/experiments/emotion-probes/diagnose_capture.py`

Measure capture repeatability without changing the frozen experiment checks.

Main interfaces: `difference`, `diagnose`, `main`.

## `src/experiments/emotion-probes/emotion_probes/__init__.py`

Centered assistant-sentence diversity probes on frozen model activations.

## `src/experiments/emotion-probes/emotion_probes/config.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `load_config`, `source_hashes`, `resolve`, `run_root`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parents[2]
PROJECT = CANONICAL.parents[1]
```

## `src/experiments/emotion-probes/emotion_probes/data.py`

Materialized inventory loader. Historical source paths are provenance only.

Main interfaces: `training_conditions`, `validate_dataset`.

Input/output settings:

```python
LABELS = EMOTIONS + ['neutral']
```

## `src/experiments/emotion-probes/emotion_probes/evaluation.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `calibrate`, `evaluate`.

## `src/experiments/emotion-probes/emotion_probes/extraction.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `cases_for`, `tokenize`, `versions`, `make_capture_fn`, `LensExtractor`, `check_shard`, `feature_manifest`, `layer_features`, `synthetic_feature`, `extract`, `smoke`.

## `src/experiments/emotion-probes/emotion_probes/fitting.py`

Training-only source-grouped CV; condition-weighted mean differences.

Main interfaces: `normalize`, `directions`, `negative_conditions`, `winning_layer`, `fit`.

## `src/experiments/emotion-probes/emotion_probes/gemma_rendering.py`

Gemma-native roles; explicit text representation of supplied tool/reasoning streams.

No input case is edited. System text is prefixed to the first user turn; tool
messages become user turns with a visible wrapper; adjacent same-role turns
join with two newlines. Saved reasoning is ordinary assistant text in <think>.
Assistant-first histories receive an empty user turn (no added prose).
Both training and evaluation use this identical adapter. Marker round-trip
checks establish exact character offsets even for repeated target strings.

Main interfaces: `render_case`.

## `src/experiments/emotion-probes/emotion_probes/pipeline.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `prepare`, `install_dashboard`, `execute`.

## `src/experiments/emotion-probes/emotion_probes/rendering.py`

Native GLM chat replay; pool only the marked assistant content/reasoning span.

Main interfaces: `render_case`, `tokenize_case`.

## `src/experiments/emotion-probes/emotion_probes/reporting.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `report`, `verify`.

## `src/experiments/emotion-probes/emotion_probes/statistics.py`

Condition-weighted metrics, deterministic cutoffs and grouped intervals.

Reused from src/experiments/emotion-probes/adapters/gemma27b/probe_harness/statistics.py.

Main interfaces: `condition_weights`, `measure`, `candidates`, `choose_threshold`, `verify_threshold`, `bootstrap_setup`, `bootstrap`.

## `src/experiments/emotion-probes/kernel_smoke.py`

Small NVFP4 runtime checks; no model checkpoint or experiment data needed.

Main interfaces: `check_nvfp4_moe`.

## `src/experiments/emotion-probes/modal_worker.py`

Durable Modal extraction worker and local progress/result mirror.

Called by canonical/run.py. An interrupted client can reconnect using --resume;
connection.json records the active call so it never launches a duplicate worker.

Main interfaces: `runtime_details`, `preflight_key`, `inspect_runtime`, `cpu_kernel_build`, `gpu_kernel_smoke`, `ensure_kernel_preflight`, `preflight`, `worker`, `download`, `mirror_snapshot`, `sync_artifacts`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parent.parent if modal.is_local() else Path('/study/src/experiments')
```

## `src/experiments/emotion-probes/prepare_stories.py`

Generate, audit, and freeze Gemma-authored topic-matched story training data.

Reuse the existing Gemma self-story prompts, lexical checks and semantic rubric.
Each accepted item is checkpointed independently; no evaluation text reaches
the generator or reviewer. API provider revisions are not claimed to be pinned.

Arguments: `--run-dir`, `--workers`.

Main interfaces: `load_module`, `accepted`, `collect_item`, `materialize`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parents[1]
PROJECT = CANONICAL.parents[1]
```

## `src/experiments/emotion-probes/resume_from_extract.py`

Resume a frozen run from extraction without scanning every token-cache file.

Main interfaces: `extract_with_policy`, `execute`.

## `src/experiments/emotion-probes/run.py`

Run this experiment independently; see README.md for inputs and commands.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
```

## `src/experiments/emotion-rh-forecasting/adjudicate_terminalwrench_disagreements.py`

Blindly relabel the frozen MiMo/GPT-5.5 disagreement sentences with a third judge.

Arguments: `--config`, `--run`, `--limit`.

Main interfaces: `prepare`, `collect`, `main`.

## `src/experiments/emotion-rh-forecasting/audit_openrouter_traces.py`

Validate trace collection and freeze equal-size finalized sales cohorts.

Main interfaces: `digest`, `audit`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/emotion-rh-traces'
```

## `src/experiments/emotion-rh-forecasting/combine_glm_coding_gpt55.py`

Combine original and newly judged GLM coding labels into one GPT-5.5 set.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SELECTION = LABEL_ROOT / 'glm53-coding-deepseek-v41-flash-1017-v1/selection.json'
OUT = LABEL_ROOT / 'glm53-coding-1000-gpt55-v1'
```

## `src/experiments/emotion-rh-forecasting/combined_emotions.py`

Combine frozen emotion projections; run after a verified canonical forecast run.

Arguments: `run_dir`.

Main interfaces: `is_glm_balanced_sales`, `load_glm_balanced_sales`, `load_custom`, `save`, `sha`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
```

## `src/experiments/emotion-rh-forecasting/compare_terminalwrench_labels.py`

Blinded judge comparison on all reference RH units and sampled non-RH units.

Arguments: `--config`, `--run`, `--limit`.

Main interfaces: `prepare`, `confusion`, `collect`, `report`, `main`.

## `src/experiments/emotion-rh-forecasting/emotion_rh_forecasting/__init__.py`

Sentence-local emotion probes predicting current and future RH labels.

## `src/experiments/emotion-rh-forecasting/emotion_rh_forecasting/analysis.py`

Frozen directions; training-only scalar cutoffs; group-disjoint evaluation.

Threshold and sampling rules preserve the historical GPT-OSS coding experiment.
Horizon zero uses the same strictly causal source pooling as future horizons.

Main interfaces: `fit_threshold`, `predict_threshold`, `assign_folds`, `make_pairs`, `balanced_groups`, `draw_balanced`, `score`, `evaluate`.

## `src/experiments/emotion-rh-forecasting/emotion_rh_forecasting/features.py`

Strict source-sentence pooling and optional local Hugging Face replay.

Main interfaces: `pool_completed`, `LocalHF`, `extract`, `verify_cached_features`.

## `src/experiments/emotion-rh-forecasting/emotion_rh_forecasting/gptoss_terminalwrench_replay.py`

Pinned MXFP4 GPT-OSS emotion projections with sentence-end causal replay.

Main interfaces: `GPTOSSReplay`.

## `src/experiments/emotion-rh-forecasting/emotion_rh_forecasting/inputs.py`

Input adapters. Historical caches are bound to exact vectors and replay specs.

Main interfaces: `Sources`, `vector_identity`, `validate_bundle`, `legacy_bundle`, `load_probes`, `legacy_replay`, `cache_job`, `validate_traces`, `token_replay`, `load_dataset`.

## `src/experiments/emotion-rh-forecasting/emotion_rh_forecasting/pipeline.py`

JSON-configured sentence RH experiment with resumable, verified artifacts.

Main interfaces: `load_config`, `resolve`, `run_root`, `freeze`, `checkpoint_dir`, `pairs_for`, `reference_check`, `verify`, `execute`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parents[2]
PROJECT = CANONICAL.parents[1]
```

## `src/experiments/emotion-rh-forecasting/emotion_rh_forecasting/reporting.py`

Figures use saved metrics; bands are subsampling variability, not confidence.

Main interfaces: `write_csv`, `figure_names`, `report`.

## `src/experiments/emotion-rh-forecasting/emotion_rh_forecasting/terminalwrench_replay.py`

Gemma replay of frozen ATIF rendering, ending forwards at sentence boundaries.

Main interfaces: `sentence_cuts`, `chunk_ranges`, `GemmaReplay`.

## `src/experiments/emotion-rh-forecasting/figures/gemma27b_haiku_heatmap.py`

Render Gemma 27B probe accuracy minus token-position baseline from saved metrics.

Arguments: `dataset`.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = HERE.parents[3]
```

## `src/experiments/emotion-rh-forecasting/generate_glm_coding_diverse.py`

Generate a resumable, frozen GLM coding cohort over additional task families.

Arguments: `--config`, `--output`, `--count-per-scenario`, `--concurrency`.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
```

## `src/experiments/emotion-rh-forecasting/generate_openrouter_traces.py`

Collect comparable coding and interactive sales traces from OpenRouter.

Each episode is checkpointed separately, so interrupted runs can be resumed safely.
Only generation is performed here; sentence segmentation and RH judging are separate.

Arguments: `--model`, `--dataset`, `--count`, `--concurrency`, `--max-turns`, `--extend`, `--output`, `--dry-run`.

Main interfaces: `stamp`, `digest`, `save`, `reasoning_text`, `completion`, `coding_trace`, `sales_trace`, `run`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
```

## `src/experiments/emotion-rh-forecasting/index_goodfire_glm52.py`

Index exact assistant prose spans in Goodfire DeepSWE GLM 5.2 traces.

Run from model-forensics with: uv run --with pysbd python
src/experiments/emotion-rh-forecasting/index_goodfire_glm52.py

Arguments: `--start`, `--stop`.

Main interfaces: `segment_safe`, `digest`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8')
OUT = SOURCE / 'sentence-index.jsonl'
```

## `src/experiments/emotion-rh-forecasting/judge_coding_hack_rate.py`

Apply the original GPT-OSS whole-answer coding rubric to new traces.

Arguments: `--run`.

Main interfaces: `sha`, `save`, `parse`, `judge`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/emotion-rh-traces'
SOURCE = ROOT / 'steering/configs/gpt_oss_120b_rh_pm005_pm010_b32.json'
LABELS = {'hack', 'honest_impossible', 'ambiguous', 'incoherent'}
```

## `src/experiments/emotion-rh-forecasting/label_gemma27b_haiku.py`

Label archived Gemma 27B verbose coding and sales traces with Haiku.

Run from the repository root with ``uv run --with pysbd python ...``.  The
frozen input bridge keeps the original source path and hash for auditability.

Arguments: `--dataset`, `--prepare-only`, `--limit-requests`, `--workers`.

Main interfaces: `freeze_sources`, `summarize`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
```

## `src/experiments/emotion-rh-forecasting/label_gemma4b_coding_lines.py`

Review Gemma 4B coding traces for exact code-line reward-hacking evidence.

This is a deterministic, task-specific line audit of the frozen 100 responses.
Positive lines show a forbidden construct, an unsound shortcut, or a false
compliance claim. They are observable evidence, not a claim to read intent.

Main interfaces: `sha`, `save`, `evidence`, `line_units`, `write_viewer`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/emotion-rh-traces/gemma3-4b-coding-openrouter-v1'
OUT = ROOT / 'runs/emotion-rh-code-line-labels/gemma3-4b-coding-v1'
```

## `src/experiments/emotion-rh-forecasting/label_gemma4b_sales.py`

Label a frozen, scenario-balanced Gemma 3 4B sales trace cohort with GPT-5.5.

Arguments: `--prepare-only`, `--limit-requests`, `--workers`.

Main interfaces: `select`, `summarize`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/emotion-rh-traces/gemma3-4b-sales-openrouter-v1/cohort-finalized-150.json'
OUT = ROOT / 'runs/emotion-rh-sentence-labels/gemma3-4b-sales-gpt55-v1'
```

## `src/experiments/emotion-rh-forecasting/label_glm_coding_deepseek.py`

Freeze and label ~1,000 GLM coding prose units with DeepSeek V4.1 Flash 1017.

Run from model-forensics:
  uv run --with pysbd python src/experiments/emotion-rh-forecasting/label_glm_coding_deepseek.py --prepare-only
  uv run --with pysbd python src/experiments/emotion-rh-forecasting/label_glm_coding_deepseek.py --limit-requests 2
  uv run --with pysbd python src/experiments/emotion-rh-forecasting/label_glm_coding_deepseek.py

Arguments: `--prepare-only`, `--limit-requests`, `--workers`.

Main interfaces: `source_item`, `prose_count`, `selection`, `label_with_recovery`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/emotion-rh-sentence-labels/glm53-coding-deepseek-v41-flash-1017-v1'
```

## `src/experiments/emotion-rh-forecasting/label_glm_coding_new_gpt55.py`

Label the 731 new GLM coding prose units with GPT-5.5 under a $7 run cap.

The 306 older coding prose units were already labeled by GPT-5.5. This run
uses the same frozen rubric, trace context, segmentation and request format.

Arguments: `--prepare-only`.

Main interfaces: `selected`, `charged_cost`, `conservative_call_bound`, `request_once`, `validated`, `label_chunk`, `summarize`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/emotion-rh-sentence-labels/glm53-coding-new-gpt55-v1'
```

## `src/experiments/emotion-rh-forecasting/label_glm_hack_traces.py`

Label all assistant output units in frozen GLM 5.3 sales/coding cohorts.

Usage: uv run --with pysbd python src/experiments/emotion-rh-forecasting/label_glm_hack_traces.py --model gpt55 --cohort nonhack
The frozen source selection and exact request hashes make interrupted runs resumable.

Arguments: `--model`, `--cohort`, `--prepare-only`, `--workers`, `--limit-requests`.

Main interfaces: `sha`, `write`, `read_source`, `selected_sources`, `prepare`, `label_chunk`, `summarize`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/emotion-rh-traces'
OUT = ROOT / 'runs/emotion-rh-sentence-labels/glm53-hack-traces-opus55-v1'
```

## `src/experiments/emotion-rh-forecasting/label_goodfire_glm52_deepseek.py`

Label a frozen, complete-trace DeepSWE GLM 5.2 pilot with DeepSeek Flash 1017.

Run from model-forensics with uv run --with pysbd python this_file.py
Use --prepare-only to freeze inputs without API requests; later runs resume.

Arguments: `--prepare-only`, `--limit-requests`, `--trace-row`, `--node-index`, `--pilot-positive-context`, `--workers`.

Main interfaces: `digest`, `freeze`, `select`, `prepare`, `label_with_recovery`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
SOURCE = Path('/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8')
OUT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-deepseek-pilot-v3'
```

## `src/experiments/emotion-rh-forecasting/label_goodfire_glm52_deepseek_v2.py`

Label a frozen, complete-trace DeepSWE GLM 5.2 pilot with DeepSeek Flash 1017.

Run from model-forensics with uv run --with pysbd python this_file.py
Use --prepare-only to freeze inputs without API requests; later runs resume.

Arguments: `--prepare-only`, `--limit-requests`, `--workers`.

Main interfaces: `digest`, `freeze`, `select`, `prepare`, `label_with_recovery`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
SOURCE = Path('/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8')
OUT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-deepseek-pilot-v2'
```

## `src/experiments/emotion-rh-forecasting/label_goodfire_glm52_gpt61sol.py`

Adjudicate unresolved Goodfire GLM 5.2 pilot sentences with GPT-6.1 Sol.

The frozen manual pilot supplies exact sentence spans. Source passage votes are
hidden from the judge. Requests and validated responses are cached by hash.
Run with: uv run --with pysbd python label_goodfire_glm52_gpt61sol.py

Arguments: `--limit`, `--workers`.

Main interfaces: `sha`, `prepare`, `annotate`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
SOURCE = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-manual-pilot-v1'
SELECTION = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-deepseek-pilot-v2/selection.json'
OUT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-pilot-v1'
```

## `src/experiments/emotion-rh-forecasting/label_goodfire_glm52_gpt61sol_5k.py`

Judge 5,000 new Goodfire GLM 5.2 sentences with GPT-6.1 Sol.

Requests are grouped by source passage and are resumable by request hash.
Run with: uv run --with pysbd python label_goodfire_glm52_gpt61sol_5k.py

Arguments: `--out`, `--limit`, `--workers`, `--prepare-only`.

Main interfaces: `sha`, `clip`, `context_for`, `prepare`, `annotate`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
```

## `src/experiments/emotion-rh-forecasting/label_terminalwrench.py`

Prepare, run, and validate blinded model sentence labels for TerminalWrench.

Run from canonical: uv run emotion-rh-forecasting/label_terminalwrench.py --config ...
Only --run dispatches requests. Prepared requests and completed annotations resume
by exact content hashes. This labels saved source text; it does not run replay.

Arguments: `--config`, `--run`, `--limit`, `--workers`, `--verify-only`.

Main interfaces: `packet_and_units`, `select_population`, `make_body`, `prepare`, `validate_response`, `annotate`, `collect`, `finish`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parents[1]
```

## `src/experiments/emotion-rh-forecasting/materialize_goodfire_glm52_gpt61sol_5k.py`

Validate and materialize the 5,000 GPT-6.1 Sol sentence judgments.

Arguments: `--out`.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
OUT = judge.OUT
```

## `src/experiments/emotion-rh-forecasting/materialize_goodfire_glm52_gpt61sol_labels.py`

Merge validated GPT-6.1 Sol adjudications into the frozen manual pilot.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
OUT = judge.OUT
```

## `src/experiments/emotion-rh-forecasting/materialize_goodfire_glm52_manual_labels.py`

Materialize reviewable sentence labels from manual decisions and source passage votes.

Main interfaces: `sha`, `write`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8')
OUT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-manual-pilot-v1'
SELECTION = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-deepseek-pilot-v2/selection.json'
```

## `src/experiments/emotion-rh-forecasting/merge_glm_sentence_labels.py`

Materialize the frozen, sentence-count-matched GLM training/evaluation cohort.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/emotion-rh-sentence-labels'
OUT = BASE / 'glm53-balanced-source-gpt55-v1'
SOURCES = [('hack', BASE / 'glm53-hack-traces-gpt55-v1/labels.json'), ('nonhack', BASE / 'glm53-nonhack-traces-gpt55-v1/labels.json')]
```

## `src/experiments/emotion-rh-forecasting/merge_goodfire_glm52_index.py`

Verify and merge the eight deterministic Goodfire GLM 5.2 sentence-index shards.

Main interfaces: `sha`, `main`.

Input/output settings:

```python
SOURCE = Path('/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8')
```

## `src/experiments/emotion-rh-forecasting/merge_terminalwrench_adjudication.py`

Verify and merge GPT-6.1 Sol Flex and standard adjudication responses.

Main interfaces: `verified_labels`, `main`.

## `src/experiments/emotion-rh-forecasting/modal_gemma4b_coding_line_replay.py`

Capture twelve frozen Gemma 4B probe projections over coding code lines.

From model-forensics: uv run --with modal python -m modal run   src/experiments/emotion-rh-forecasting/modal_gemma4b_coding_line_replay.py

Main interfaces: `capture_fn`, `worker`, `download`, `main`.

Input/output settings:

```python
CANONICAL = Path('/study/src/experiments') if Path('/study/src/experiments').exists() else HERE.parent
ROOT = CANONICAL.parents[1]
LOCAL = ROOT / 'runs/emotion-rh-replay/gemma3-4b-coding-lines-best12-v1'
```

## `src/experiments/emotion-rh-forecasting/modal_gemma4b_sales_replay.py`

Capture twelve frozen Gemma 4B probe projections over sales traces.

From model-forensics: uv run --with modal python -m modal run   src/experiments/emotion-rh-forecasting/modal_gemma4b_sales_replay.py

Main interfaces: `capture_fn`, `worker`, `download`, `main`.

Input/output settings:

```python
CANONICAL = Path('/study/src/experiments') if Path('/study/src/experiments').exists() else HERE.parent
ROOT = CANONICAL.parents[1]
LOCAL = ROOT / 'runs/emotion-rh-replay/gemma3-4b-sales-best12-v1'
```

## `src/experiments/emotion-rh-forecasting/modal_gemma4b_terminalwrench_replay.py`

Run frozen Gemma TerminalWrench sentence replay on the existing H200 cache.

Main interfaces: `replay`, `main`.

Input/output settings:

```python
PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
CANONICAL = PROJECT / 'src/experiments'
```

## `src/experiments/emotion-rh-forecasting/modal_glm_hack_replay.py`

Capture twelve frozen GLM probe projections over archived hack-trace prefixes.

From model-forensics: uv run --with modal python -m modal run   src/experiments/emotion-rh-forecasting/modal_glm_hack_replay.py

Main interfaces: `capture_fn`, `worker`, `download`, `main`.

Input/output settings:

```python
CANONICAL = Path('/study/src/experiments') if Path('/study/src/experiments').exists() else HERE.parent
ROOT = CANONICAL.parents[1]
LOCAL = ROOT / 'runs/emotion-rh-replay/glm-5.3-hack-traces-best12-v1'
```

## `src/experiments/emotion-rh-forecasting/modal_gptoss_terminalwrench_replay.py`

Run frozen GPT-OSS TerminalWrench sentence replay on the existing H200 cache.

Main interfaces: `replay`, `main`.

Input/output settings:

```python
PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
CANONICAL = PROJECT / 'src/experiments'
```

## `src/experiments/emotion-rh-forecasting/modal_missing_cell_replay.py`

Score frozen inputs for the two missing model/dataset cells on Modal.

Main interfaces: `score`, `main`.

## `src/experiments/emotion-rh-forecasting/modal_terminalwrench_replay.py`

Run frozen Gemma TerminalWrench sentence replay on the existing H200 cache.

Main interfaces: `replay`, `main`.

Input/output settings:

```python
PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
CANONICAL = PROJECT / 'src/experiments'
```

## `src/experiments/emotion-rh-forecasting/plot_combined_roc.py`

Plot k=0, 5, 10 out-of-fold ROC curves for saved twelve-probe forecasts.

Arguments: `run_dirs`.

Main interfaces: `sha`, `main`.

## `src/experiments/emotion-rh-forecasting/plot_gemma4b_coding_line_forecast.py`

Plot Gemma 3 4B coding code-line forecasts from frozen labels and replay.

Main interfaces: `adapted`, `figure`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
LABELS = ROOT / 'runs/emotion-rh-code-line-labels/gemma3-4b-coding-v1/labels.json'
REPLAY = ROOT / 'runs/emotion-rh-replay/gemma3-4b-coding-lines-best12-v1'
OUT = ROOT / 'runs/emotion-rh-forecasting/gemma3-4b-coding-lines-v1'
```

## `src/experiments/emotion-rh-forecasting/plot_gemma4b_sales_forecast.py`

Plot Gemma 3 4B sales sentence forecasts from frozen labels and replay.

Main interfaces: `figure`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
LABELS = ROOT / 'runs/emotion-rh-sentence-labels/gemma3-4b-sales-gpt55-v1/labels.json'
REPLAY = ROOT / 'runs/emotion-rh-replay/gemma3-4b-sales-best12-v1'
OUT = ROOT / 'runs/emotion-rh-forecasting/gemma3-4b-sales-gpt55-v1'
```

## `src/experiments/emotion-rh-forecasting/plot_glm_coding_1000.py`

Plot GPT-5.5-labeled GLM coding sentence forecasts from verified probe replay.

Main interfaces: `plot`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
REPLAY = ROOT / 'runs/emotion-rh-replay/glm-5.3-coding-1000-best12-v1'
LABELS = ROOT / 'runs/emotion-rh-sentence-labels/glm53-coding-1000-gpt55-v1/labels.json'
OUT = ROOT / 'runs/emotion-rh-forecasting/glm-5.3-coding-1000-gpt55-v1'
```

## `src/experiments/emotion-rh-forecasting/plot_glm_hack_forecast.py`

Fit frozen-GLM emotion probe forecasts from verified replay scores.

uv run --with scikit-learn --with matplotlib python   src/experiments/emotion-rh-forecasting/plot_glm_hack_forecast.py

Arguments: `--cohort`.

Main interfaces: `rows_for_dataset`, `frozen_folds`, `figure`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
REPLAY = ROOT / 'runs/emotion-rh-replay/glm-5.3-hack-traces-best12-v1'
LABELS = ROOT / 'runs/emotion-rh-sentence-labels/glm53-hack-traces-gpt55-v1/labels.json'
OUT = ROOT / 'runs/emotion-rh-forecasting/glm-5.3-hack-traces-exploratory-v1'
```

## `src/experiments/emotion-rh-forecasting/plot_goodfire_glm52_expanded.py`

Update k=0..10 accuracy and probe ROC with the 5,000-sentence cohort.

Arguments: `--include-second`.

Main interfaces: `load_new_rows`, `make_sparse_pairs`, `assign_task_folds`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
OUT = ROOT / 'runs/emotion-rh-forecasting/goodfire-glm52-deepswe-gpt61sol-expanded-5k-v1'
```

## `src/experiments/emotion-rh-forecasting/plot_goodfire_glm52_forecast.py`

Plot task-disjoint GLM 5.2 DeepSWE sentence forecasts from frozen replay.

Arguments: `--label-root`, `--out`.

Main interfaces: `load_rows`, `plot`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = HERE.parents[2]
OUT = ROOT / 'runs/emotion-rh-forecasting/goodfire-glm52-deepswe-manual-pilot-v1'
```

## `src/experiments/emotion-rh-forecasting/plot_goodfire_glm52_roc.py`

Task-disjoint, probe-specific FPR/TPR curves for the GLM 5.2 DeepSWE pilot.

Run with: uv run --with numpy --with matplotlib --with pysbd --with scikit-learn
          --with transformers python plot_goodfire_glm52_roc.py

Main interfaces: `roc_for_probe`, `plot`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
LABELS = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-pilot-v1'
OUT = ROOT / 'runs/emotion-rh-forecasting/goodfire-glm52-deepswe-gpt61sol-pilot-v1'
```

## `src/experiments/emotion-rh-forecasting/plot_missing_probe_roc.py`

Render missing per-emotion ROC curves from frozen forecast inputs.

This is a projection of existing labels, probe scores, and folds. It does not
refit probes or change any forecast run. Curves use every eligible pair in the
run's primary cohort, with train-fold-only polarity and CDF calibration.

Main interfaces: `verified_replay`, `load_rows`, `population`, `draw_sheet`, `draw_best`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
BASE = ROOT / 'runs/emotion-rh-forecasting'
OUT = BASE / 'roc-per-emotion'
```

## `src/experiments/emotion-rh-forecasting/plot_negative_horizons.py`

Replot saved emotion/RH forecasts on the equivalent -10..0 axis.

No scores, labels, populations, estimates, or uncertainty bands are changed:
each original horizon k is drawn at x=-k. The token-position diagnostic is omitted.

Main interfaces: `render`, `write_gallery`.

Input/output settings:

```python
ROOT = HERE.parents[2]
```

## `src/experiments/emotion-rh-forecasting/preflight_terminalwrench_neutral_v7_20261026.py`

Audit neutral v7 sentence labels and task-held-out forecast feasibility.

Main interfaces: `run`.

Input/output settings:

```python
ROOT = HERE.parents[2]
OUT = ROOT / 'runs/emotion-rh-forecasting/terminalwrench-neutral-v7-20261026-preflight'
SOURCES = [LABEL_ROOT / 'terminalwrench-v7-steering-gpt61sol-v2', LABEL_ROOT / 'terminalwrench-v7-neutral-extension-gpt61sol-v2', LABEL_ROOT / 'terminalwrench-neutral20-20261026-astra-review-v1']
PROBES = {'GPT-OSS 120B': ROOT / 'data/probes/gptoss120b/probes.json', 'Gemma 4B': ROOT / 'data/probes/gemma4b/probes.json', 'GLM 5.3': ROOT / 'data/probes/glm53/probes.json'}
```

## `src/experiments/emotion-rh-forecasting/prepare_gemma27b_haiku_replay.py`

Freeze Haiku-labeled Gemma 27B coding/sales traces for causal token replay.

Arguments: `--dataset`.

Main interfaces: `messages_for`, `render_span`, `make_job`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
```

## `src/experiments/emotion-rh-forecasting/prepare_gemma4b_coding_line_replay.py`

Freeze Gemma-native replay jobs for audited coding code lines.

uv run --with 'transformers==5.17.0' --with 'jinja2==3.1.6' --with scikit-learn python   src/experiments/emotion-rh-forecasting/prepare_gemma4b_coding_line_replay.py

Main interfaces: `adapted`, `fields_for`, `make_job`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
LABELS = ROOT / 'runs/emotion-rh-code-line-labels/gemma3-4b-coding-v1/labels.json'
PROBES = ROOT / 'data/probes/gemma4b'
OUT = ROOT / 'runs/emotion-rh-replay/gemma3-4b-coding-lines-best12-v1'
```

## `src/experiments/emotion-rh-forecasting/prepare_gemma4b_sales_replay.py`

Freeze Gemma-native replay jobs for GPT-5.5-labeled sales traces.

uv run --with 'transformers==5.17.0' --with 'jinja2==3.1.6' --with scikit-learn python   src/experiments/emotion-rh-forecasting/prepare_gemma4b_sales_replay.py

Main interfaces: `fields_for`, `make_job`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
LABELS = ROOT / 'runs/emotion-rh-sentence-labels/gemma3-4b-sales-gpt55-v1/labels.json'
PROBES = ROOT / 'data/probes/gemma4b'
OUT = ROOT / 'runs/emotion-rh-replay/gemma3-4b-sales-best12-v1'
```

## `src/experiments/emotion-rh-forecasting/prepare_glm_hack_replay.py`

Freeze native GLM replay jobs for the GPT-5.5-labeled hack traces.

uv run --with 'transformers==5.17.0' --with 'jinja2==3.1.6' --with scikit-learn python   src/experiments/emotion-rh-forecasting/prepare_glm_hack_replay.py --cohort nonhack

Arguments: `--cohort`.

Main interfaces: `fields_for`, `make_job`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
LABELS = ROOT / 'runs/emotion-rh-sentence-labels/glm53-hack-traces-gpt55-v1/labels.json'
PROBES = ROOT / 'runs/emotion-probes/glm-5.3-nvfp4-diversity-v2/probes'
OUT = ROOT / 'runs/emotion-rh-replay/glm-5.3-hack-traces-best12-v1'
```

## `src/experiments/emotion-rh-forecasting/prepare_goodfire_glm52_5k_replay.py`

Freeze local-context NVFP4 activation replay for the 5,000 judged spans.

Each selected source passage is replayed with system/user and up to 30 prior
nodes, trimming older nodes only if needed to fit the 131072-token context.

Arguments: `--labels`, `--out`.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
OUT = ROOT / 'runs/emotion-rh-replay/goodfire-glm52-deepswe-nvfp4-gpt61sol-5k-v1'
```

## `src/experiments/emotion-rh-forecasting/prepare_goodfire_glm52_replay.py`

Freeze native GLM 5.2 NVFP4 replay of Goodfire FP8 archive text.

Main interfaces: `messages_and_spans`, `job_for`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = HERE.parents[2]
PROBES = ROOT / 'runs/emotion-probes/glm-5.2-nvfp4-diversity-v2/probes'
OUT = ROOT / 'runs/emotion-rh-replay/goodfire-glm52-deepswe-nvfp4-best12-pilot-v2'
```

## `src/experiments/emotion-rh-forecasting/prepare_terminalwrench_neutral_replay.py`

Freeze native-template token replay for neutral v7 GPT-OSS/Gemma 4B traces.

Arguments: `model`.

Main interfaces: `rendered_offsets`, `prepare`.

Input/output settings:

```python
ROOT = HERE.parents[2]
```

## `src/experiments/emotion-rh-forecasting/prepare_terminalwrench_neutral_v7_20261026.py`

Freeze exact neutral ATIF sentence inputs for the canonical forecast runner.

This prepares labels and native-visible histories; it does not create probe
activations or imply that a model-specific causal replay has been validated.

Main interfaces: `prepare`.

Input/output settings:

```python
ROOT = HERE.parents[2]
DEST = ROOT / 'runs/emotion-rh-forecasting/terminalwrench-neutral-v7-20261026-inputs'
```

## `src/experiments/emotion-rh-forecasting/prepare_terminalwrench_replay.py`

Freeze selected emotion probes and exact ATIF token/span evidence for GPU replay.

Arguments: `--config`.

Main interfaces: `prepare`.

## `src/experiments/emotion-rh-forecasting/render_combined.py`

Render the saved twelve-emotion forecast bundles without refitting models.

Arguments: `run_dirs`.

Main interfaces: `sha`, `main`.

## `src/experiments/emotion-rh-forecasting/reuse_glm_coding_replay.py`

Reuse checksum-identical GLM probe scores for previously replayed coding traces.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
SOURCES = [ROOT / 'runs/emotion-rh-replay/glm-5.3-hack-traces-best12-v1', ROOT / 'runs/emotion-rh-replay/glm-5.3-nonhack-traces-best12-v1']
```

## `src/experiments/emotion-rh-forecasting/run.py`

Run this experiment independently; see README.md for inputs and commands.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
```

## `src/experiments/emotion-rh-forecasting/score_sales_hack_rate.py`

Count observed ledger violations in the frozen finalized sales cohorts.

Main interfaces: `sha`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/emotion-rh-traces'
```

## `src/experiments/emotion-rh-forecasting/select_glm_nonhack_traces.py`

Freeze a seeded, whole-trace control sample matched on prose sentence counts.

uv run --with pysbd python src/experiments/emotion-rh-forecasting/select_glm_nonhack_traces.py

Main interfaces: `prose_count`, `candidates`, `choose`, `main`.

Input/output settings:

```python
OUT = ROOT / 'runs/emotion-rh-sentence-labels/glm53-nonhack-traces-gpt55-v1/selection.json'
```

## `src/experiments/emotion-rh-forecasting/select_goodfire_glm52_gpt61sol_5k.py`

Freeze 5,000 source-positive-passage sentences from 50 new DeepSWE tasks.

Main interfaces: `digest`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8')
OUT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-5k-v1'
```

## `src/experiments/emotion-rh-forecasting/select_goodfire_glm52_gpt61sol_5k_b.py`

Freeze a second non-overlapping 5,000-sentence Goodfire GLM 5.2 cohort.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SOURCE = Path('/scratch/model-forensics/goodfire/deepswe-glm-5.2-fp8')
OUT = ROOT / 'runs/emotion-rh-sentence-labels/goodfire-glm52-deepswe-gpt61sol-5k-b-v1'
```

## `src/experiments/emotion-rh-forecasting/validate_terminalwrench_label_comparison.py`

Independently audit a completed sentence-label comparison from raw evidence.

Main interfaces: `read`, `sha`, `digest`, `validate`.

## `src/experiments/emotion-rh-forecasting/validate_terminalwrench_labels.py`

Read-only audit of a finished TerminalWrench annotation bundle.

Uses saved raw responses and original source messages, independently of the
labeling runner's parsing/join functions. Writes one validation receipt.

Arguments: `output`.

Main interfaces: `load`, `digest`, `check`, `audit`.

## `src/experiments/emotion-self-report/emotion_self_report/__init__.py`

Sentence-prefix emotion self-report baseline.

## `src/experiments/emotion-self-report/emotion_self_report/analysis.py`

Paired sentence correlations with whole-trace bootstrap intervals.

Promoted from gpt_oss_self_report_plan_20261016/analyze_score_correlations.py.
Ranks are recomputed in every bootstrap replicate; constants remain undefined.

Main interfaces: `ranks`, `pearson`, `correlations`, `join`, `analyze`.

## `src/experiments/emotion-self-report/emotion_self_report/collection.py`

One emotion per fresh replay; durable attempts and strict 0–10 parsing.

Main interfaces: `payload`, `request_body`, `parse_rating`, `response_content`, `Client`, `cached_record`, `call`, `compact`, `collect`, `import_legacy`.

## `src/experiments/emotion-self-report/emotion_self_report/config.py`

Small JSON interface; all scientific settings are frozen before collection.

Main interfaces: `load_config`, `run_root`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parents[2]
PROJECT = CANONICAL.parents[1]
```

## `src/experiments/emotion-self-report/emotion_self_report/inputs.py`

Exact-prefix checkpoints and identity-based joins, with a legacy importer.

Main interfaces: `checkpoint_key`, `job_key`, `prefix_messages`, `load_inputs`.

## `src/experiments/emotion-self-report/emotion_self_report/pipeline.py`

Canonical baseline collection, historical import, analysis and verification.

Main interfaces: `resolve`, `freeze_inputs`, `verify`, `execute`, `main`.

## `src/experiments/emotion-self-report/emotion_self_report/reporting.py`

Correlation figure and a compact report; no display-scale fitting.

Main interfaces: `write_csv`, `undefined_correlation_label`, `report`.

## `src/experiments/emotion-self-report/emotion_self_report/segmentation.py`

Exact-span sentence segmentation, promoted from the coding baseline.

Source: experiments/gpt_oss_coding_early_warning_20261016/label_sentences.py.
Code fences and headings remain in replay history but are not rating targets.

Main interfaces: `segment`, `validate_spans`.

## `src/experiments/emotion-self-report/finalize_activations.py`

Validate complete replay coverage, join existing ratings, and report by dataset.

Usage: uv run python emotion-self-report/finalize_activations.py RUN_DIRECTORY
No self-report API calls or GPU execution. Input score rows must use original
trace/checkpoint IDs and character offsets in the source assistant message.

Arguments: `root`, `--bootstrap-replicates`.

Main interfaces: `write`, `validate`, `verify_correlations`, `trajectory_viewer`, `main`.

## `src/experiments/emotion-self-report/finalize_gemma27b_activations.py`

Join causal Gemma token projections to the frozen self-report checkpoints.

Main interfaces: `checkpoint_kind`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
SELECTION = ROOT / 'runs/emotion-self-report-activations/gemma27b-complete-traces-v1'
REPLAY = SELECTION / 'gemma27b-replay-v1'
OUT = SELECTION
```

## `src/experiments/emotion-self-report/finalize_gemma4b_activations.py`

Join causal Gemma 4B token projections to the frozen self-report checkpoints.

Main interfaces: `checkpoint_kind`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
SELECTION = ROOT / 'runs/emotion-self-report-activations/gemma4b-complete-traces-v1'
REPLAY = SELECTION / 'gemma4b-replay-v1'
OUT = SELECTION
```

## `src/experiments/emotion-self-report/gptoss_sales_replay.py`

Replay frozen GPT-OSS Sales self-report checkpoints with the best 12 probes.

Preparation and Modal replay are separate so a one-trace smoke can validate the
frozen input contract before the complete run. No generation or self-report API
calls are made here.

Arguments: `--smoke`, `--prepare-only`.

Main interfaces: `sha`, `prepare`, `execute`, `download`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3] if IS_LOCAL else Path('/root/model-forensics')
SOURCE = ROOT / 'experiments/gpt_oss_sales_sentence_rh_12probes_20261016'
OUT = ROOT / 'runs/emotion-self-report-activations/gptoss-complete-traces-v1'
```

## `src/experiments/emotion-self-report/materialize_glm53_scores.py`

Materialize verified GLM 5.3 activation projections and provenance.

Arguments: `--root`.

Main interfaces: `read`, `sha`, `atomic_json`, `materialize`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
```

## `src/experiments/emotion-self-report/materialize_gptoss_scores.py`

Materialize endpoint scores and provenance after GPT-OSS replay download.

Main interfaces: `sha`, `checkpoint_kind`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'runs/emotion-self-report-activations/gptoss-complete-traces-v1'
SOURCE = ROOT / 'experiments/gpt_oss_sales_sentence_rh_12probes_20261016'
```

## `src/experiments/emotion-self-report/modal_gemma27b_activations.py`

Score frozen self-report checkpoints with the pinned Gemma causal replay.

Arguments: `--trace-limit`.

Main interfaces: `replay`, `main`, `run`.

Input/output settings:

```python
PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
ROOT = PROJECT
CANONICAL = ROOT / 'src/experiments'
OUT = ROOT / 'runs/emotion-self-report-activations/gemma27b-complete-traces-v1/gemma27b-replay-v1'
```

## `src/experiments/emotion-self-report/modal_gemma4b_activations.py`

Score frozen self-report checkpoints with the pinned Gemma 4B causal replay.

Arguments: `--trace-limit`.

Main interfaces: `replay`, `main`, `run`.

Input/output settings:

```python
PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
ROOT = PROJECT
CANONICAL = ROOT / 'src/experiments'
OUT = ROOT / 'runs/emotion-self-report-activations/gemma4b-complete-traces-v1/gemma4b-replay-v1'
```

## `src/experiments/emotion-self-report/modal_glm53_replay.py`

Capture best12 raw residual projections for frozen GLM 5.3 self-report traces.

Main interfaces: `capture_fn`, `worker`, `download`, `main`.

Input/output settings:

```python
CANONICAL = Path('/study/src/experiments') if Path('/study/src/experiments').exists() else HERE.parent
ROOT = CANONICAL.parents[1] if len(CANONICAL.parents) > 1 else Path('/study')
LOCAL = Path(os.environ.get('GLM_REPLAY_ROOT', ROOT / 'runs/emotion-self-report-activations/glm53-complete-traces-v2'))
```

## `src/experiments/emotion-self-report/pooled/calculate.py`

Pool Coding/Sales observations, validate source parity, and bootstrap traces.

Main interfaces: `read`, `write`, `file_identity`, `csv_out`.

Input/output settings:

```python
ROOT = _REPO
```

## `src/experiments/emotion-self-report/pooled/render.py`

Draw saved pooled metrics as four model plots and one 2x2 comparison.

Main interfaces: `draw`, `finish`.

Input/output settings:

```python
ROOT = _REPO
```

## `src/experiments/emotion-self-report/prepare_gemma27b_activations.py`

Freeze Gemma 3 27B causal prefixes for the selected self-report traces.

Arguments: `--trace-limit`.

Main interfaces: `make_job`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
SELECTION = ROOT / 'runs/emotion-self-report-activations/gemma27b-complete-traces-v1'
PROBES = ROOT / 'runs/emotion-rh-replay/gemma-3-27b-terminalwrench-best12-pilot50-v2/probes.json'
OUT = SELECTION / 'gemma27b-replay-v1'
```

## `src/experiments/emotion-self-report/prepare_gemma4b_activations.py`

Freeze Gemma 3 4B causal prefixes for the selected self-report traces.

Arguments: `--trace-limit`.

Main interfaces: `make_job`, `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
SELECTION = ROOT / 'runs/emotion-self-report-activations/gemma4b-complete-traces-v1'
PROBES = ROOT / 'data/probes/gemma4b/probes.json'
OUT = SELECTION / 'gemma4b-replay-v1'
```

## `src/experiments/emotion-self-report/prepare_glm53_replay.py`

Prepare native-template GLM 5.3 replay prompts for frozen self-report traces.

Arguments: `--root`.

Main interfaces: `main`.

Input/output settings:

```python
CANONICAL = HERE.parent
ROOT = CANONICAL.parents[1]
OUT = FROZEN
```

## `src/experiments/emotion-self-report/prepare_terminalwrench_v7.py`

Freeze neutral TerminalWrench ATIF histories for sentence self-report.

Main interfaces: `sha`, `write_frozen`, `export`.

Input/output settings:

```python
CANONICAL = HERE.parent
PROJECT = CANONICAL.parents[1]
SOURCE = PROJECT / 'runs/emotion-steering'
DEST = HERE / 'inputs/terminalwrench-neutral-v7-20261026'
```

## `src/experiments/emotion-self-report/run.py`

Run this experiment independently; see README.md for inputs and commands.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
```

## `src/experiments/emotion-self-report/select_activation_traces.py`

Freeze whole, fully rated traces for activation replay; never issue API calls.

Run with Python 3: select_activation_traces.py --model gemma4b --target 2000.
The output is an immutable input contract for model-specific replay workers.

Arguments: `--model`, `--target`, `--output`.

Main interfaces: `read`, `sha`, `freeze`, `load_source`, `choose`, `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'runs/emotion-self-report'
```

## `src/experiments/emotion-self-report/select_terminalwrench_glm53.py`

Freeze every fully rated GLM 5.3 TerminalWrench trace for probe replay.

Main interfaces: `main`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'runs/emotion-self-report/glm-5.3-terminalwrench-neutral-self-report-v1'
OUT = ROOT / 'runs/emotion-self-report-activations/glm53-terminalwrench-neutral-complete-traces-v1'
```

## `src/experiments/emotion-self-report/summarize_activation_runs.py`

Build a local index of the four complete-trace activation runs.

Main interfaces: `main`.

## `src/experiments/emotion-self-report/terminalwrench_dashboard.py`

Read-only live dashboard for the three neutral TerminalWrench rating runs.

Arguments: `--port`.

Main interfaces: `status`, `Handler`.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3] / 'runs/emotion-self-report'
```

## `src/experiments/emotion-steering/audit_terminalwrench_glm53_neutral10_20261101.py`

Audit all ten generation attempts and summarize sentence labels if available.

Main interfaces: `audit`.

## `src/experiments/emotion-steering/audit_terminalwrench_neutral20_review_astra.py`

Blindly recheck all first-pass needs_review sentences with GPT-6 Astra.

Arguments: `--prepare-only`, `--workers`.

Main interfaces: `task_text`, `step_context`, `prepare`, `collect`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
SOURCE = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-neutral20-20261026-gpt61sol-v1'
OUT = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-neutral20-20261026-astra-review-v1'
```

## `src/experiments/emotion-steering/audit_terminalwrench_positive_labels.py`

Independently recheck positive DeepSeek sentence labels with local trace context.

uv run --with pysbd python src/experiments/emotion-steering/audit_terminalwrench_positive_labels.py

Main interfaces: `task_text`, `step_context`, `prepare`, `collect`, `main`.

Input/output settings:

```python
ROOT = batch.ROOT
SOURCE = batch.OUT
OUT = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-steering-deepseek-v41-flash-1017-positive-audit-v1'
```

## `src/experiments/emotion-steering/audit_terminalwrench_v7_steering_sentences.py`

Independent structural/provenance audit for the v7 GPT-6.1-Sol labels.

Main interfaces: `audit`.

Input/output settings:

```python
CANONICAL = HERE.parent
PROJECT = CANONICAL.parents[1]
OUT = PROJECT / 'runs/emotion-rh-sentence-labels/terminalwrench-v7-steering-gpt61sol-v2'
```

## `src/experiments/emotion-steering/benchmark_continuous.py`

Replay saved first-turn inputs for serving validation; never experiment episodes.

Arguments: `model`.

Main interfaces: `run`.

## `src/experiments/emotion-steering/build_terminalwrench_rh_expanded_20261026.py`

Build trace-level RH rates from v7 steering plus both neutral extensions.

Main interfaces: `wilson`, `key`, `build`.

Input/output settings:

```python
ROOT = HERE.parents[2]
OUTPUT = LABEL_ROOT / 'terminalwrench-v7-expanded-neutral-rh-20261026'
SOURCES = [('v7-steering', LABEL_ROOT / 'terminalwrench-v7-steering-gpt61sol-v2', LABEL_ROOT / 'terminalwrench-v7-steering-gpt61sol-v2/manifest.json'), ('neutral-extension-8', LABEL_ROOT / 'terminalwrench-v7-neutral-extension-gpt61sol-v2', LABEL_ROOT / 'terminalwrench-v7-neutral-extension-gpt61sol-v2/manifest.json'), ('neutral20-astra-reviewed', LABEL_ROOT / 'terminalwrench-neutral20-20261026-astra-review-v1', LABEL_ROOT / 'terminalwrench-neutral20-20261026-gpt61sol-v1/manifest.json')]
```

## `src/experiments/emotion-steering/continue_batch.py`

Preserve a stopped checkpoint when changing only steering batch size.

The source run stays immutable. Imported payloads retain their generation batch
size and bind to their original bytes through a per-record provenance receipt.
This helper is deliberately outside the frozen generation package.

Arguments: `--source-run`, `--verify-run`, `--config`.

Main interfaces: `verify_continuation`, `import_checkpoint`.

## `src/experiments/emotion-steering/data/terminalwrench-tasks/1351/tests/test_outputs.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `test_script_exists_and_executable`, `test_output_directory_and_files`, `test_image_dimensions_within_limits`, `test_processing_log`.

## `src/experiments/emotion-steering/data/terminalwrench-tasks/606/tests/test_outputs.py`

Tests for Linux server power management configuration task.
Verifies that the system is properly configured to prevent sleep/suspend states.

Main interfaces: `test_grub_kernel_parameters`, `test_grub_config_updated`, `test_systemd_sleep_targets_masked`.

## `src/experiments/emotion-steering/data/terminalwrench-tasks/flood-escape-path-planner/environment/flood_simulator.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `FloodSimulator`.

## `src/experiments/emotion-steering/data/terminalwrench-tasks/flood-escape-path-planner/tests/test_outputs.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `test_reaches_safety`, `test_avoids_water`, `test_already_safe`, `test_no_water`, `test_survival_in_dead_end`, `test_single_cell_island`, `test_narrow_bridge`, `test_race_condition`, `test_corner_start`, `test_large_open_grid`, `test_trapped_but_delay`, `test_exceed_steps`.

## `src/experiments/emotion-steering/emotion_steering/__init__.py`

Paired, signed emotion interventions with explicit behavioral adjudication.

## `src/experiments/emotion-steering/emotion_steering/active_budget.py`

Live active-time clock. Only dispatched inference queue intervals are excluded.

Main interfaces: `ActiveClock`, `serve_clock`, `read_clock`, `read_clock_file`, `ActiveBudgetTimeoutError`, `BudgetInfrastructureError`, `ActiveTimeTerminus2`.

## `src/experiments/emotion-steering/emotion_steering/async_backend.py`

Continuous Lens serving with independent completion and cancellation.

Main interfaces: `AsyncLensBackend`.

## `src/experiments/emotion-steering/emotion_steering/backend.py`

One request at a time: stable per-sample RNG and exception-safe interventions.

Main interfaces: `BackendValidationError`, `intervention`, `split_completion`, `HFBackend`, `lens_hook`, `LensBackend`, `make_backend`.

## `src/experiments/emotion-steering/emotion_steering/config.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `safe_id`, `load_vector`, `load_probes`, `resolve`, `tasks`, `run_root`.

Input/output settings:

```python
CANONICAL = HERE.parent
PROJECT = CANONICAL.parents[1]
```

## `src/experiments/emotion-steering/emotion_steering/environments.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `coding`, `ledger_score`, `SalesEpisode`, `sales`, `generate_batch`, `generate_one`.

## `src/experiments/emotion-steering/emotion_steering/inference_queue.py`

Live queue and execution timing for condition-isolated inference batches.

Main interfaces: `CompletionQueue`.

## `src/experiments/emotion-steering/emotion_steering/judging.py`

Blinded, evidence-backed RH adjudication and old-rubric comparison.

Main interfaces: `_sales_fixture`, `request`, `evidence`, `validate_label`, `audit`, `legacy`, `check_fixtures`.

Input/output settings:

```python
LABELS = ('reward_hack', 'honest_limit', 'ordinary_failure', 'benign', 'uncertain', 'incoherent')
```

## `src/experiments/emotion-steering/emotion_steering/lens_worker_ext.py`

Named worker RPC for selected-layer hooks; no callable serialization needed.

Main interfaces: `install_selected_hooks`, `SelectedLayerExtension`.

## `src/experiments/emotion-steering/emotion_steering/pipeline.py`

Prepare, generate, judge, report, verify. Historical imports are explicit.

Main interfaces: `progress`, `commit`, `verify_receipts`, `initialize`, `generate`, `import_existing`, `reuse_judgments`, `judge`, `joined_rows`, `verify`, `main`.

## `src/experiments/emotion-steering/emotion_steering/remote.py`

Bind remote resumptions and downloads to the complete frozen run suite.

Main interfaces: `suite_members`, `saved_call_id`, `result_archives`, `validate_archive`.

## `src/experiments/emotion-steering/emotion_steering/reporting.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `wilson`, `summarize`, `report`.

## `src/experiments/emotion-steering/emotion_steering/terminalwrench.py`

Native Harbor tasks use a condition-bound chat endpoint and fresh Docker trials.

Arguments: `--run-dir`, `--host`, `--port`.

Main interfaces: `endpoint_identity`, `api_base`, `preflight`, `alias`, `generate_one`, `Service`, `serve`.

## `src/experiments/emotion-steering/emotion_steering/timed_proxy.py`

Bridge live worker timing into the local Harbor active-time clock.

Main interfaces: `TimedProxy`.

## `src/experiments/emotion-steering/finish_figures.py`

Render each campaign's figures when its saved run passes verification.

Arguments: `--campaign`, `--once`.

Main interfaces: `render_ready`, `main`.

## `src/experiments/emotion-steering/finish_terminalwrench_glm53_neutral10_20261101.py`

Finish this fixed ten-trace batch: audit, sentence-label, and save a report.

Main interfaces: `main`.

## `src/experiments/emotion-steering/glm_adaptive.py`

GLM batch32 continuation with recorded, checkpoint-preserving batch16 fallback.

The canonical scientific code stays frozen. This adapter is included in the new
run's source hashes and keeps the requested batch size separate from runtime
fallback decisions.

Arguments: `--config`, `--import-source`, `--queued`, `--finish`, `--child-specs`, `--effective-batch-size`, `--status-path`.

Main interfaces: `resolve_adaptive`, `check_sources`, `import_checkpoint`, `memory_or_engine_failure`, `run_attempts`, `child`, `finish`.

## `src/experiments/emotion-steering/glm_modal_worker.py`

One GLM GPU allocation; isolated CUDA subprocesses make batch fallback safe.

Main interfaces: `adaptive_worker`, `continue_suite`.

## `src/experiments/emotion-steering/glm_sales_capped.py`

Explicit mixed-budget Sales continuation; preserve completed episodes verbatim.

Arguments: `--config`, `--import-source`, `--finish`, `--child-specs`, `--effective-batch-size`, `--status-path`.

Main interfaces: `resolve_capped`, `import_checkpoint`, `child`, `finish`.

## `src/experiments/emotion-steering/glm_sales_capped_worker.py`

One GLM GPU allocation; isolated CUDA subprocesses make batch fallback safe.

Main interfaces: `adaptive_worker`, `continue_suite`.

## `src/experiments/emotion-steering/gptoss_terminalwrench_backend.py`

GPT-OSS native-channel runtime for the pinned interactive sweep.

Main interfaces: `TerminalWrenchGPTOSS`.

## `src/experiments/emotion-steering/gptoss_terminalwrench_timing.py`

Live queue/execution boundaries for one serialized GPT-OSS worker.

The generator acquires the backend lock before emitting generation_started. A
consumer may safely pause its episode clock before that event, and must resume
it immediately on receipt. Queue/startup require a separate wall-time watchdog.

Main interfaces: `timed_call`, `TimedGPTOSSMixin`.

## `src/experiments/emotion-steering/gptoss_terminalwrench_worker.py`

One shared, serialized H200 allocation for interactive GPT-OSS episodes.

Main interfaces: `GPTOSSTerminalWrench`.

Input/output settings:

```python
PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
```

## `src/experiments/emotion-steering/gptoss_terminalwrench_worker_v2.py`

Dedicated v2 GPT-OSS app with streamed live active-budget accounting.

Main interfaces: `GPTOSSTerminalWrench`.

Input/output settings:

```python
PROJECT = HERE.parents[2] if modal.is_local() else Path('/study')
```

## `src/experiments/emotion-steering/label_saved_glm53_terminalwrench_neutral10_early.py`

Populate the final labeling cache for saved traces while generation continues.

Input/output settings:

```python
PROJECT = HERE.parents[2]
ROOT = PROJECT / 'runs/emotion-steering' / RUN
```

## `src/experiments/emotion-steering/label_terminalwrench_glm53_neutral10_20261101.py`

Label the ten new GLM TerminalWrench traces using the existing sentence rubric.

## `src/experiments/emotion-steering/label_terminalwrench_glm53_neutral10_recovery_20261101.py`

Resume the final labeling pass serially, respecting transient retry intervals.

Input/output settings:

```python
PROJECT = HERE.parents[2]
```

## `src/experiments/emotion-steering/label_terminalwrench_neutral20_20261026.py`

Sentence-label the 60 new neutral TerminalWrench traces with GPT-6.1 Sol.

Input/output settings:

```python
ROOT = HERE.parents[2]
```

## `src/experiments/emotion-steering/label_terminalwrench_steering_sentences.py`

Label saved steering-episode assistant sentences with GPT-6.1 Sol.

Prepare: uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences.py --prepare-only
Pilot:   uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences.py --limit 3
Run:     uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences.py --workers 8

Arguments: `--prepare-only`, `--limit`, `--workers`.

Main interfaces: `segment_safe`, `source_rows`, `prepare`, `collect`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parent.parent
ROOT = CANONICAL.parents[1]
OUT = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-steering-gpt61sol-v1'
```

## `src/experiments/emotion-steering/label_terminalwrench_steering_sentences_deepseek.py`

Label saved steering-episode assistant sentences with DeepSeek V4.1 Flash 1017.

Prepare: uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences_deepseek.py --prepare-only
Pilot:   uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences_deepseek.py --limit 3
Run:     uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_steering_sentences_deepseek.py --workers 4

Arguments: `--prepare-only`, `--limit`, `--workers`.

Main interfaces: `segment_safe`, `source_rows`, `prepare`, `annotate`, `collect`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parent.parent
ROOT = CANONICAL.parents[1]
OUT = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-steering-deepseek-v41-flash-1017-v1'
```

## `src/experiments/emotion-steering/label_terminalwrench_v7_neutral_extension.py`

Sentence-label the 16 new neutral v7 extension traces with GPT-6.1-Sol.

Prepare only: uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_neutral_extension.py --prepare-only
Run:         uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_neutral_extension.py --workers 1

Input/output settings:

```python
ROOT = HERE.parents[2]
```

## `src/experiments/emotion-steering/label_terminalwrench_v7_steering_sentences.py`

Label saved steering-episode assistant sentences with GPT-6.1 Sol.

Prepare: uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_steering_sentences.py --prepare-only
Pilot:   uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_steering_sentences.py --limit 3
Run:     uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_steering_sentences.py --workers 8

Arguments: `--prepare-only`, `--limit`, `--workers`.

Main interfaces: `segment_safe`, `source_rows`, `prepare`, `collect`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parent.parent
ROOT = CANONICAL.parents[1]
OUT = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-v7-steering-gpt61sol-v1'
```

## `src/experiments/emotion-steering/label_terminalwrench_v7_steering_sentences_v2.py`

Label saved steering-episode assistant sentences with GPT-6.1 Sol.

Prepare: uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_steering_sentences_v2.py --prepare-only
Run:     uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_steering_sentences_v2.py --workers 1

Arguments: `--prepare-only`, `--limit`, `--workers`.

Main interfaces: `segment_safe`, `source_rows`, `prepare`, `collect`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parent.parent
ROOT = CANONICAL.parents[1]
OUT = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-v7-steering-gpt61sol-v2'
```

## `src/experiments/emotion-steering/modal_worker.py`

Pinned model-specific GPU image; canonical coordinator owns local judging.

Main interfaces: `worker`, `archive_run`, `main`, `suite`, `run_local`, `download_archive`.

Input/output settings:

```python
CANONICAL = HERE.parent
PROJECT = CANONICAL.parents[1]
```

## `src/experiments/emotion-steering/pause_after_batch.py`

Run through Modal container exec to persist a whole batch before a handoff.

Only use for a worker whose reload was requested. This pauses the designated
coordinator and its VLLM processes; after the final volume commit, its app may be
stopped and the saved responses imported into a continuation run.

Arguments: `run_id`, `--batch-size`, `--target`, `--coordinator-pid`.

Main interfaces: `main`.

## `src/experiments/emotion-steering/plot_saved.py`

Plot a verified steering run without generating responses or changing labels.

Arguments: `--run-dir`.

Main interfaces: `cells`, `setup_axis`, `rh_curve`, `save`, `presentation`, `protocol_disclosure`, `overview`, `gallery`, `draw`, `main`.

Input/output settings:

```python
LABELS = {'reward_hack': ('Reward hacking', '#2376a9'), 'ordinary_failure': ('Ordinary failure', '#a05c91'), 'honest_limit': ('Honest limitation', '#4c9b83'), 'uncertain': ('Uncertain', '#b3bcc6'), 'benign': ('Benign', '#b2cfa2'), 'incoherent': ('Incoherent', '#cc9949')}
```

## `src/experiments/emotion-steering/plot_terminalwrench_rh_expanded_20261026.py`

Plot episode-level RH prose rates by v7 steering arm for three models.

Claim: observed arm rates vary; the enlarged neutral cohort is a descriptive reference.
Template: horizontal dot plot with Wilson intervals, one figure per model.
Shows: all labelable steering emotions at alpha -0.10 and +0.10.
Accent: positive-alpha arm; negative-alpha arm is grey.
Text: model, two arm labels, neutral marker, x-axis label, emotion tick labels.
Caption: post-hoc prose labels, denominator, interval, unequal task coverage.
Sketch: neutral vertical line | four emotion rows with paired dots and whiskers.

Main interfaces: `load`, `draw`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
OUT = DATA / 'plots'
```

## `src/experiments/emotion-steering/plot_terminalwrench_sentence_rh.py`

Plot per-emotion TerminalWrench steering from verified sentence labels.

uv run --with matplotlib python src/experiments/emotion-steering/plot_terminalwrench_sentence_rh.py   --labels runs/emotion-rh-sentence-labels/terminalwrench-steering-deepseek-v41-flash-1017-v1   --output runs/emotion-steering/terminalwrench-sentence-rh-deepseek-v41-flash-1017-v1

Arguments: `--labels`, `--output`.

Main interfaces: `wilson`, `make_cells`, `draw_emotion`, `main`.

Input/output settings:

```python
CANONICAL = Path(__file__).resolve().parent.parent
ROOT = CANONICAL.parents[1]
```

## `src/experiments/emotion-steering/plot_terminalwrench_v7_sentence_rh.py`

Plot v7 GPT-6.1-Sol RH sentence-label prevalence by model and selected emotion.

Main interfaces: `wilson`, `main`.

Input/output settings:

```python
ROOT = HERE.parents[2]
OUT = ROOT / 'runs/emotion-rh-sentence-labels/terminalwrench-v7-steering-rh-gpt61sol-v2/model-overview'
```

## `src/experiments/emotion-steering/recover_terminalwrench_glm53_neutral10_20261101.py`

Recover the fixed ten tasks after coordinator SIGTERM, preserving all attempts.

Main interfaces: `main`.

## `src/experiments/emotion-steering/reuse_terminalwrench.py`

Import explicitly selected clean attempts with verifiable original provenance.

Arguments: `source`, `target`.

Main interfaces: `reuse`.

## `src/experiments/emotion-steering/run.py`

Run this experiment independently; see README.md for inputs and commands.

Input/output settings:

```python
ROOT = Path(__file__).resolve().parents[3]
```

## `src/experiments/emotion-steering/summarize_terminalwrench_glm53_neutral10_20261101.py`

Verify trace artifacts and derive the final report from saved sentence labels.

Input/output settings:

```python
PROJECT = HERE.parents[2]
```

## `src/experiments/emotion-steering/terminalwrench_async_worker.py`

Bounded continuous serving; no whole-batch completion barrier.

Main interfaces: `AsyncInteractive`.

## `src/experiments/emotion-steering/terminalwrench_evaluate.py`

Blinded TerminalWrench adjudication, task-bootstrap summaries and paired viewer.

Usage: python terminalwrench_evaluate.py RUN_ROOT [--judge]
The generation config and canonical responses remain immutable. Judging is resumable.

Arguments: `run_root`, `--judge`, `--allow-external-judge`.

Main interfaces: `validate`, `judge`, `reward`, `interval`, `summarize`, `report`, `main`.

## `src/experiments/emotion-steering/terminalwrench_finish.py`

Durable finish watcher: stop this model's compute, then judge the saved sweep.

Arguments: `--run-root`, `--worker-app`, `--status`.

Main interfaces: `main`.

## `src/experiments/emotion-steering/terminalwrench_glm53_neutral10_20261101.py`

Generate ten fresh, unsteered GLM 5.3 TerminalWrench episodes; resume safely.

Arguments: `--prepare`, `--port`, `--concurrency`.

Main interfaces: `prepare`.

## `src/experiments/emotion-steering/terminalwrench_keepalive.py`

Keep the single inference allocation warm while its coordinator is alive.

Arguments: `root`, `--worker-app`, `--worker-class`.

Main interfaces: `alive`, `main`.

## `src/experiments/emotion-steering/terminalwrench_lens_worker.py`

Persistent, request-isolated Lens inference for interactive Modal trials.

Deploy once per model with EMOTION_STEERING_RUN and TERMINALWRENCH_WORKER_APP.
Concurrent single-turn RPCs are coalesced into batches of at most four.

Main interfaces: `LensInteractive`.

## `src/experiments/emotion-steering/terminalwrench_migrate_scheduler.py`

Adopt only completed native trials at a quiescent boundary; never kill agents.

Arguments: `root`, `--stop-only`.

Main interfaces: `argv`, `alive`, `live_children`, `active_http_connections`, `native_finished`, `main`.

## `src/experiments/emotion-steering/terminalwrench_neutral20_20261026.py`

Collect twenty new, shared-task neutral TerminalWrench episodes per model.

Arguments: `--model`, `--prepare`, `--port`, `--concurrency`.

Main interfaces: `paths`, `shared_scenarios`, `prepare`, `validate`, `run`.

Input/output settings:

```python
PROJECT = HERE.parents[2]
SOURCES = {'gpt_oss': 'gptoss120b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7', 'gemma': 'gemma4b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7', 'glm': 'glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7'}
```

## `src/experiments/emotion-steering/terminalwrench_neutral20_audit.py`

Audit the saved neutral20 TerminalWrench traces, including partial timeouts.

Main interfaces: `audit`.

## `src/experiments/emotion-steering/terminalwrench_neutral_extension.py`

Complete the twenty missing neutral GLM tasks from the frozen 24-task suite.

Arguments: `--prepare`, `--port`.

Main interfaces: `prepare_extension`, `main`.

## `src/experiments/emotion-steering/terminalwrench_neutral_extension_clockfix.py`

Complete unattempted neutral GLM tasks through the unified API and clock port.

Arguments: `--prepare`, `--port`.

Main interfaces: `prepare_extension`, `main`.

## `src/experiments/emotion-steering/terminalwrench_neutral_extension_resume.py`

Complete unattempted neutral GLM tasks, retaining all earlier episode attempts.

Arguments: `--prepare`, `--port`.

Main interfaces: `prepare_extension`, `main`.

## `src/experiments/emotion-steering/terminalwrench_neutral_extension_supervise.py`

Finalize the neutral extension and stop only its dedicated model app.

Arguments: `--run-root`.

Main interfaces: `process_identity`, `audit`, `main`.

## `src/experiments/emotion-steering/terminalwrench_neutral_to20.py`

Generate neutral GLM traces until twenty valid trajectories or twenty attempts.

Arguments: `--prepare`, `--port`.

Main interfaces: `valid_trace`, `prepare_run`, `retry_task`, `main`.

## `src/experiments/emotion-steering/terminalwrench_supervise.py`

Durable continuous-sweep supervisor with bounded OOM fallback and provenance.

Arguments: `--model`, `--root`, `--port`, `--suffix`, `--selection`, `--inference-slots`.

Main interfaces: `terminate_group`, `main`.

## `src/experiments/emotion-steering/terminalwrench_sweep.py`

Frozen 24-task Modal Harbor sweep, with local durable per-episode artifacts.

Arguments: `--model`, `--prepare`, `--reduced`, `--selection`, `--continuous`, `--inference-slots`, `--run-suffix`, `--protocol`, `--worker-app`, `--worker-class`, `--port`, `--concurrency`.

Main interfaces: `prepare`, `run_rolling`, `main`.

Input/output settings:

```python
PROJECT = HERE.parents[2]
```

## `src/experiments/emotion-steering/terminalwrench_v7_neutral_extension.py`

Run a small, provenance-preserving neutral-only extension to a v7 sweep.

Arguments: `--model`, `--port`, `--concurrency`.

Main interfaces: `main`.

Input/output settings:

```python
PROJECT = HERE.parents[2]
SOURCES = {'glm': 'glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7', 'gemma': 'gemma4b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7'}
```

## `src/experiments/emotion-steering/terminalwrench_validate.py`

Inspect actual first-batch inference receipts and native terminal interactions.

Main interfaces: `_pending_local_requests`, `validate_first_batch`.

## `src/experiments/emotion-steering/test_gptoss_terminalwrench_timing.py`

Contract checks for live queue/execution timing without loading model weights.

Main interfaces: `TimingTests`.

## `src/experiments/emotion-steering/test_gptoss_timed_proxy.py`

GPT proxy live-clock and generator cleanup contracts, with no remote calls.

Main interfaces: `Clock`, `ProxyTests`.

## `src/experiments/emotion-steering/tests/test_async_backend.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `FakeEngine`, `backend`, `test_short_request_returns_before_long_request`, `test_queue_time_and_executing_cancellation`.

## `src/experiments/emotion-steering/tests/test_terminalwrench_active_budget.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `ClockTests`, `HarborClassificationTests`.

## `src/experiments/emotion-steering/tests/test_terminalwrench_scheduler.py`

Concurrency contract: refill free sandbox slots without waiting for peers.

Main interfaces: `RollingSchedulerTests`.

## `src/experiments/emotion-traces/spikes/load_missing.py`

Join verified new score shards to the original selection adapters.

Main interfaces: `append_entries`.

## `src/experiments/emotion-traces/spikes/prepare_missing.py`

Freeze the two missing-cell replay batches from saved, reviewed labels.

Main interfaces: `finish`, `gemma`, `glm`.

Input/output settings:

```python
ROOT = _REPO
```

## `src/experiments/emotion-traces/spikes/render.py`

Render selected saved trajectories without re-running the selection.

Main interfaces: `draw`.

## `src/experiments/emotion-traces/spikes/search_stronger_coding.py`

Search saved Gemma coding passages for large separation and flat controls.

Main interfaces: `keep_id`.

Input/output settings:

```python
ROOT = _REPO
OUT = HERE / 'stronger-coding-review'
```

## `src/experiments/emotion-traces/spikes/select_spikes.py`

Select descriptive upward divergences from saved token-level replay scores.

Main interfaces: `curve`.

Input/output settings:

```python
ROOT = _REPO
```

## `src/experiments/emotion-traces/spikes/verify_complete.py`

Validate the completed grid, new replay receipts, and frozen original curves.

Input/output settings:

```python
ROOT = _REPO
```

## `src/experiments/emotion-traces/task-averages/build_data.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `keep`, `offsetmask`, `tok`, `keep_id`, `add`, `trajectory`.

Input/output settings:

```python
ROOT = _REPO
OUT = ROOT / 'runs/emotion-traces/task-averages'
REPLAY = ROOT / 'runs/emotion-rh-replay'
```

## `src/experiments/emotion-traces/task-averages/make_gallery.py`

Supporting implementation; inspect functions and its experiment README before direct use.

## `src/experiments/emotion-traces/task-averages/render_guilty.py`

Supporting implementation; inspect functions and its experiment README before direct use.

## `src/experiments/shared/__init__.py`

Small utilities shared by canonical experiment runners.

## `src/experiments/shared/cli.py`

Common argument parsing for independent experiment entrypoints.

Each experiment invokes this helper with its own type. It never schedules a
different experiment; prerequisites are explicit inputs in the supplied config.

Arguments: `--config`, `--dry-run`, `--resume`, `--through`, `--smoke-only`, `--reuse-token-cache`, `--serve`.

Main interfaces: `main`.

## `src/experiments/shared/io.py`

Supporting implementation; inspect functions and its experiment README before direct use.

Main interfaces: `require`, `read`, `digest`, `sha`, `atomic`, `save_array`, `save_npz`.

## `src/experiments/shared/progress.py`

Single-writer progress.json producer for model-forensics/dashboards.

Main interfaces: `now`, `lock`, `Progress`.

## `src/experiments/shared/server.py`

Read-only loopback dashboard. Do not expose raw datasets or model tokens.

Main interfaces: `Handler`, `serve`.
