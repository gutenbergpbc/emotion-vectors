# Emotion vectors

Experiment-based replication code for emotion probes, reward-hacking monitoring,
and emotion steering. Start with the [Gemma 3 4B reproduction guide](docs/reproduce-gemma4b.md).
It gives independent commands; there is no full-reproduction coordinator.

```bash
uv sync --locked --group dev
uv run python tools/artifacts.py fetch --category datasets
uv run python tools/artifacts.py fetch --category probes
uv run python src/experiments/emotion-probes/run.py --config configs/gemma4b/probes.json --dry-run
```

Download links are recorded in `manifests/releases.json` and point to
[release v0.1.0](https://github.com/gutenbergpbc/emotion-vectors/releases/tag/v0.1.0). The release contains datasets, selected probes,
text traces, results, and `SHA256SUMS`. Already materialized files are verified
and reused. To restore from locally downloaded archives instead, add
`--from dist` to a fetch command.

## Experiments

| Directory | Question |
|---|---|
| [emotion-probes](src/experiments/emotion-probes/README.md) | Can directions trained on emotion text distinguish held-out sales examples? |
| [emotion-traces](src/experiments/emotion-traces/README.md) | How do projections differ around risky actions and matched controls? |
| [emotion-steering](src/experiments/emotion-steering/README.md) | Does adding/subtracting emotion directions change behavior? |
| [emotion-rh-forecasting](src/experiments/emotion-rh-forecasting/README.md) | Can frozen emotion readouts predict reward hacking at future offsets? |
| [emotion-self-report](src/experiments/emotion-self-report/README.md) | Do sentence-level self-reports correlate with probe readouts? |
| [emotion-pressure](src/experiments/emotion-pressure/README.md) | Do emotion-denial prompts change readouts under scripted pressure? |
| [emotion-denial](src/experiments/emotion-denial/README.md) | Do emotion-denial prompts change behavioral reward hacking? |

Each directory has independent entrypoints. [The script reference](docs/scripts.md)
indexes their purposes, arguments, and literal input/output settings. Model configs
live in `configs/`; the tested walkthrough configs live in `configs/gemma4b/`.

## Artifacts

- Complete available catalog text data, controls, reviews, and frozen splits.
- Twelve selected probes per model, with each probe's training family, contrast,
  layer, centering, checkpoint and original selection rule. Historical selection
  bias is retained, including GPT-OSS selection using held-out labels.
- Optional original text traces and annotations; small saved self-report ratings.
- Published metrics, aggregate figure data and figures for all available models.

Activations, residual arrays, token scores, token caches, model weights and local
virtual environments are excluded from release archives. Experiments regenerate
their intermediates under ignored `runs/`. [Artifact documentation](docs/data-and-probes.md)
explains provenance, remaining catalog gaps, downloading and release preparation.

Gemma 4B covers probe validation, trace analyses, forecasting, steering and
self-report. The original pressure/denial experiments use Gemma 27B and GPT-OSS;
they are separate scripts and are not presented as completed Gemma 4B experiments.
Fresh stochastic generations need not reproduce the exact archived rates.

## Development

```bash
uv run pytest
uv run python tools/build_docs.py
```

CPU tests cover scientific rules, causal replay, resumability and portability.
GPU checks require optional dependencies and actual hardware. No inference or
paid API calls are made by the CPU suite. Source provenance is recorded in
`manifests/source-code.json`; upstream notices are in [THIRD_PARTY.md](THIRD_PARTY.md).
