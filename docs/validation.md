# Validation of the initial replication checkout

Validated on November 3, 2026 with the repository's locked Python 3.12 environment.

- `uv run pytest`: 113 passed, 5 skipped. Skips are optional runtime/model checks
  unavailable in the CPU environment. The tests cover calibration/split isolation,
  causal token boundaries, replay receipts, steering semantics, Sales ledgers,
  self-report parsing/correlation, corruption detection, and archive extraction.
- Probe training and both Gemma 4B steering configs pass workload dry runs.
  Frozen dataset hashes and selected-bundle loading are checked.
- Fresh trace generation passes its no-API dry run; a self-report collection
  config passes its workload preview without issuing requests.
- The original pressure-denial setup passes its forward-count/prompt dry run.
- TerminalWrench neutral preflight reads the packaged text/labels and reconstructs
  task folds: Gemma 4B has 33 traces and seven h=10-positive task groups. GLM's
  insufficient positive-group coverage remains reported, not filled in.
- Every enumerated release payload passes its SHA-256 check. Dataset and probe
  archives were restored into a separate temporary checkout containing only this
  repository's source/configs/manifests. Probe and Sales-steering dry runs pass
  there with the research checkout absent from import/input paths.
- Python source compiles; primary documentation links resolve locally.

The initial TerminalWrench synthetic test fixture predates the source reader's
current receipt schema. Its annotation kind and independent-check tolerance were
updated to the current schema; production replay validation was not weakened.

No GPU extraction, model generation, paid judging or new self-report collection
was launched during this migration. The Gemma guide preserves the source methods
and commands but a full GPU/API execution in the new checkout remains to be done.

The available text datasets are packaged. Four historical catalog references
(three unique guide/review files) have no matching source file; their exact links
are recorded in `manifests/catalog-coverage.json`. The primary frozen training,
evaluation and split files are present and verified. Two stale corpus references
were resolved to their actual preserved sources with resolution notes.

Release archives are prepared locally in `dist/` and are not yet attached to a
public release. `manifests/releases.json` deliberately has no public URLs.
