# Replication repository

Experiments live in `src/experiments/`, organized like the original canonical
experiments. Keep independent entrypoints; do not add a full-reproduction
coordinator. Document commands, inputs, outputs, scientific assumptions and
compute requirements in each experiment's README and `docs/scripts.md`.

Start with `docs/reproduce-gemma4b.md`. Shared scientific utilities live in
`src/experiments/shared/`; environment implementations live in
`src/environments/`. Preserve model revisions, vector orientation, training
families, calibration selection, source grouping and held-out splits.

The release contains complete catalog text data and only selected best probes,
with per-probe training mix recorded. Never add activations, token-score arrays,
model checkpoints, credentials, or local virtual environments to Git or release
archives. Runtime caches under `runs/` may contain those intermediates. Build
release bundles using `tools/package_artifacts.py`; verify with
`tools/artifacts.py verify`.

Use `uv run pytest` for CPU scientific/portability checks. GPU/API execution
costs money: use dry-run previews for validation; do not launch full experiments
as an incidental check. Existing frozen input bytes and historical originals
must not be rewritten to make old receipts pass. New code gets new run IDs.
