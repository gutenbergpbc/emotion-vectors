# Emotion probes

Train mean-difference directions on the frozen diversity corpus, calibrate them
on edited sales examples, and evaluate held-out emotion specificity.

```bash
uv run python src/experiments/emotion-probes/run.py --config configs/gemma4b/probes.json --dry-run
uv run python src/experiments/emotion-probes/run.py --config configs/gemma4b/probes.json --through tokenize
uv run python src/experiments/emotion-probes/run.py --config configs/gemma4b/probes.json --resume
```

`run.py` owns only this experiment. Stages are prepare, tokenize, extract, fit,
calibrate, evaluate, report and verify. Modal GPU extraction uses a pinned H100
runtime for Gemma 4B; `--smoke-only` checks real capture before full extraction.
`--resume` refuses changed source/config/input identities. `--serve PORT` serves
that run's read-only report.

Inputs are the four files in `data/datasets/diversity-sales/`. The immutable split
has five training folds and separate calibration/held-out sales source groups.
`emotion_probes/data.py` defines schemas and input hashes; `gemma_rendering.py`
preserves native rendering and marked assistant spans; `extraction.py` captures
post-block residuals; `fitting.py` selects layers using training folds;
`evaluation.py` selects thresholds/contrasts on calibration only;
`reporting.py` verifies saved scientific artifacts.

All 36 candidates are internal fitting outputs; only twelve selected directions
are released. New output goes under `runs/emotion-probes/<run-id>/`. The released
best probes and their mixes are indexed in `manifests/probes.json`.

`adapters/gemma27b` and `adapters/gptoss120b` preserve model-specific extraction,
rendering and pinned requirement files used by larger-model replay scripts.
They are supporting code for adapting experiments, not additional required
Gemma 4B walkthroughs. Historical story-generation helpers require their own
explicit corpus preparation and are not substituted for the diversity experiment.

See [Gemma 4B instructions](../../../docs/reproduce-gemma4b.md) and the
[script reference](../../../docs/scripts.md) for per-file entrypoints.
