# Reproduce results with Gemma 3 4B

Run commands from the repository root. Each command is an independent experiment
or preparation step. Use the frozen data and labels to compare with published
results; fresh generations and judge calls define a new realization.

## Setup and downloads

```bash
uv sync --locked --group dev
uv run python tools/artifacts.py fetch --category datasets
uv run python tools/artifacts.py fetch --category probes
uv run python tools/artifacts.py fetch --category traces
uv run python tools/artifacts.py fetch --category results
```

Default downloads use the URLs in `manifests/releases.json` for release v0.1.0.
For locally downloaded archives, add `--from dist`. Set `HF_TOKEN`
in your environment after accepting the Gemma checkpoint terms. Modal commands
use your own authenticated account and create empty durable caches when needed.
API collection/judging uses `OPENROUTER_API_KEY`. Keep credentials outside Git.

The checkpoint is `google/gemma-3-4b-it`, revision
`093f9f388b31de276ce2de164bdc2081324b9767`, BF16. Probe extraction and the Lens
replays use one H100 with the pinned remote runtime in `modal_worker.py`.
Tokenization downloads tokenizer metadata; GPU commands download weights remotely.
The development box does not need CUDA. Local replay alternatives need the
appropriate CUDA PyTorch environment; do not substitute its numerical backend
without a new run identity and replay checks.

## 1. Train, calibrate and evaluate emotion probes

```bash
uv run python src/experiments/emotion-probes/run.py --config configs/gemma4b/probes.json --dry-run
uv run python src/experiments/emotion-probes/run.py --config configs/gemma4b/probes.json --through tokenize
uv run python src/experiments/emotion-probes/run.py --config configs/gemma4b/probes.json --resume --smoke-only
uv run python src/experiments/emotion-probes/run.py --config configs/gemma4b/probes.json --resume
```

The inputs contain 2,700 training cases and 2,258 unique sales replays, split into
1,129 calibration and 1,129 held-out examples. Five source-grouped training folds
select layers; calibration selects one contrast per emotion before held-out
evaluation. All 36 candidate directions are fitted internally. The release
contains only the selected twelve. Results go to
`runs/emotion-probes/replication-gemma-3-4b-it-diversity-v1/`.

Later commands default to the released selected probes in `data/probes/gemma4b/`.
To use your newly trained probes, change each relevant script's `PROBES` setting
or the config's `probe_bundle` to the new run's `probes/` directory. Do not replace
the frozen released probes or edit old checksums.

## 2. Generate traces or use the supplied traces

The trace archive already includes the original Coding and Sales responses, sales
feedback, fixed cohorts and labels. Skip new collection for archived reproduction.
For a new collection, the following commands issue paid API requests:

```bash
uv run python src/experiments/emotion-rh-forecasting/generate_openrouter_traces.py --model gemma3-4b --dataset coding --count 100 --output runs/new-gemma4b-coding
uv run python src/experiments/emotion-rh-forecasting/generate_openrouter_traces.py --model gemma3-4b --dataset sales --count 150 --output runs/new-gemma4b-sales
```

Those fresh output folders need their own frozen cohorts and labels before
analysis. `label_gemma4b_sales.py` contains the original 1,200-prose-unit selection
and GPT-5.5 rubric; `label_gemma4b_coding_lines.py` labels code lines rather than
prose. Their `SOURCE`/`OUT` settings identify archived populations. Point them at
new inputs explicitly instead of overwriting the supplied evidence.

## 3. Regenerate Coding and Sales projections and forecasts

```bash
uv run python src/experiments/emotion-rh-forecasting/prepare_gemma4b_sales_replay.py
EMOTION_PROBE_MODEL_FAMILY=gemma3 EMOTION_PROBE_GPU=H100:1 uv run modal run src/experiments/emotion-rh-forecasting/modal_gemma4b_sales_replay.py
uv run python src/experiments/emotion-rh-forecasting/plot_gemma4b_sales_forecast.py

uv run python src/experiments/emotion-rh-forecasting/prepare_gemma4b_coding_line_replay.py
EMOTION_PROBE_MODEL_FAMILY=gemma3 EMOTION_PROBE_GPU=H100:1 uv run modal run src/experiments/emotion-rh-forecasting/modal_gemma4b_coding_line_replay.py
uv run python src/experiments/emotion-rh-forecasting/plot_gemma4b_coding_line_forecast.py
```

Preparation verifies source text and sentence spans, then downloads the pinned
tokenizer. GPU workers regenerate missing score shards and verify replay receipts.
No token scores are downloaded from the release. Sales evaluates held-out
scenarios; Coding evaluates code lines. Horizons mean original unit `i+k`, not
warning time before the first hack. Sampling bands are balanced-subsample variation.

The combined twelve-feature logistic forecasts are separate commands:

```bash
uv run python src/experiments/emotion-rh-forecasting/combined_emotions.py runs/emotion-rh-forecasting/gemma3-4b-sales-gpt55-v1
uv run python src/experiments/emotion-rh-forecasting/combined_emotions.py runs/emotion-rh-forecasting/gemma3-4b-coding-lines-v1
```

## 4. TerminalWrench forecasts

The supplied label files preserve native episode text and task-group splits.
These commands replay the frozen neutral traces; they do not launch new terminal
tasks or require a live Harbor environment.

```bash
uv run python src/experiments/emotion-rh-forecasting/preflight_terminalwrench_neutral_v7_20261026.py
uv run python src/experiments/emotion-rh-forecasting/prepare_terminalwrench_neutral_replay.py 'Gemma 4B'
uv run modal run src/experiments/emotion-rh-forecasting/modal_gemma4b_terminalwrench_replay.py --run-dir runs/emotion-rh-replay/gemma4b-terminalwrench-neutral-v7-best12-20261026
uv run python src/experiments/emotion-rh-forecasting/run.py --config configs/gemma4b-terminalwrench-neutral-v7-cached-rh-forecasting.json
uv run python src/experiments/emotion-rh-forecasting/combined_emotions.py runs/emotion-rh-forecasting/gemma4b-terminalwrench-neutral-v7-rh-20261026-v3
```

## 5. Emotion traces and RH/control comparisons

After the relevant replay commands have completed:

```bash
uv run python src/experiments/emotion-traces/task-averages/build_data.py
uv run python src/experiments/emotion-traces/task-averages/make_gallery.py
```

The adapter uses available completed replay sets. It preserves the original
128-token window, punctuation exclusion, per-trace centering, matching, and
reused-control counts. Missing controls remain missing. Selected-spike analysis
is separate in `emotion-traces/spikes/`; it selects examples after inspecting
their curves and must not be interpreted as a population-level effect.

## 6. Emotion steering

```bash
uv run python src/experiments/emotion-steering/run.py --config configs/gemma4b/steering-coding.json --dry-run
uv run python src/experiments/emotion-steering/run.py --config configs/gemma4b/steering-coding.json
uv run python src/experiments/emotion-steering/run.py --config configs/gemma4b/steering-sales.json
```

Each full config schedules 588 episodes: three scenarios, four replicates,
twelve emotions and four signed strengths, plus a shared baseline. These commands
use GPU generation and paid judging. Use `--through generate` to stop before
judging and `--resume` to continue. Steering is residual-norm-relative at one
selected post-block layer. Generation and truncation settings are frozen in the
configs; inspect the dry-run before launching.

## 7. Self-report correlations

Use the released whole-trace selections and saved ratings to regenerate only
activation readouts, then correlations:

```bash
uv run python src/experiments/emotion-self-report/prepare_gemma4b_activations.py
uv run python src/experiments/emotion-self-report/modal_gemma4b_activations.py
uv run python src/experiments/emotion-self-report/finalize_gemma4b_activations.py
uv run python src/experiments/emotion-self-report/finalize_activations.py runs/emotion-self-report-activations/gemma4b-complete-traces-v1
```

Each checkpoint uses the last causally completed assistant token. Ratings use
the recorded 0–10 question, not a newly inferred 1–10 scale. For new paid rating
collection, use `emotion-self-report/run.py` with a config identifying visible
traces, checkpoints and question; see that experiment's README.

The pressure and behavioral-denial studies use Gemma 27B/GPT-OSS. Their scripts
are included, but they are not additional Gemma 4B reproduction commands.

## Obtain all published results

```bash
uv run python tools/artifacts.py fetch --category results
uv run python tools/artifacts.py verify --category results
uv run python tools/results.py
```

This reads the saved metrics and figure data across all models without GPU/API
calls. It does not recompute scientific estimates. Full model replay and fresh
generation have not been launched as part of preparing this repository; consult
`docs/validation.md` for the checks actually performed.
