# Data, probes, traces and results

`manifests/artifacts.json` enumerates every release payload with category, size,
SHA-256, and original research path. Original source files remain byte-identical.
Relocated label indexes are separate exports with `transformed: true`; only
directory references change. Character spans and source trace hashes remain fixed.

## Datasets

`data/datasets/diversity-sales/` contains the frozen, materialized training,
evaluation, pair-audit, split and protocol files. Preserve their bytes: probe
preparation checks the original four scientific input hashes. Source paths in
these JSON files are historical provenance, not dependencies to open.

`data/datasets/catalog/` retains both source inventories and their available
text/review payloads. `manifests/catalog-coverage.json` accounts for each inventory
link as included, derived artifact excluded, external reference, or missing source.
Do not interpret this as every historical report having a recoverable raw dataset.
The complete available catalog is broader than the fixed twelve-emotion paper
input packet. Historical development flags and unresolved reviews are preserved.

`data/tasks/` contains exact Coding prompts and Sales scenario specifications.
The edited sales emotion benchmark and the interactive sales environment are
different inputs and serve different roles.

## Selected probes

`manifests/probes.json` records twelve directions per model. Each bundle contains
portable `probes.json` and a compact `vectors.npz`/`manifest.json`/`selected-12.json`
interface. The release does not contain every candidate or all-layer probe.
Probe fitting still computes all contrasts internally before selection.

Each emotion is labeled with training family, contrast, centering, layer and
checkpoint revision. Gemma 4B and GLM selections use calibration mixed balanced
accuracy; Gemma 27B uses its recorded diversity selection. GPT-OSS's historical
mixed diversity/self-story selection uses held-out accuracy and is explicitly
selection-biased. Preserve the recorded selection rather than presenting it as
calibration-only or selecting a new winner during packaging. Qwen's selected
bundle is retained for historical compatibility with pressure/steering sources.

Never mix a direction across checkpoint precision variants. Raw projection
magnitudes are not comparable across layers, models or training families.

## Traces

Text-only traces materialize under `runs/` with original experiment names so
existing preparation scripts can find them. They include prompts, visible output,
reasoning where available, tool feedback, ledger states and labels. Source-model
API traces replayed on a pinned local checkpoint are not claimed to contain that
checkpoint's original serving activations. Self-report selections retain exact
visible prefixes, checkpoint spans and observed ratings; per-request API receipt
duplicates are omitted from the compact trace release.

Some historical paths in archived metadata are informational. New replay specs
and receipts must be generated in this repository; old token-score receipts are
not portable evidence for newly computed projections. No token scores, tokenizer
arrays or residual tensors are part of the release. Runtime experiments may save
these under ignored `runs/` for resume and independent verification.

## Release preparation

```bash
# Maintainer only: read the original research checkout.
uv run python tools/export_source.py --source /path/to/model-forensics
uv run python tools/package_artifacts.py
uv run python tools/artifacts.py verify
```

The exporter includes original source corpora, controls, reviews, selected probes,
text traces, labels and saved result tables/figures. It omits numeric intermediates
and records catalog gaps. The packager uses the explicit manifest, splits archives
into bounded shards, sets deterministic archive metadata, and rejects arrays
outside the probe category. It never archives an entire local run directory.

Archives are prepared in ignored `dist/`. `manifests/releases.json` has null URLs
until publication; there are no invented public links. After choosing a tag, run
the packager with `--base-url` for that release and upload the exact archives.
The fetcher verifies archive and member hashes, rejects links/path traversal and
refuses to overwrite changed local artifacts. A release tag should bind code,
manifests, and archive versions together.
