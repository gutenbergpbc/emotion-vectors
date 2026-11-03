# Neutral TerminalWrench OpenRouter self-report collection

The live read-only dashboard is served on the devbox's port 8797 by
`terminalwrench_dashboard.py`. From another machine, forward port 8797 in the
[devbox dashboard](https://devbox.gutenberg.ai/) and open the forwarded URL;
`http://127.0.0.1:8797/` works only inside the devbox. It polls the three canonical run directories
every ten seconds and shows committed numeric/null/nonnumeric responses,
failures, OpenRouter-reported cost, collection rate, and ETA. It does not serve
raw prompts, trace text, or API credentials.

Run IDs:

| Model | Run ID | Sentence checkpoints | Requests |
|---|---|---:|---:|
| GPT-OSS 120B | `gpt-oss-120b-terminalwrench-neutral-self-report-v2` | 850 | 10,200 |
| Gemma 3 4B | `gemma-3-4b-terminalwrench-neutral-self-report-v1` | 787 | 9,444 |
| GLM 5.3 | `glm-5.3-terminalwrench-neutral-self-report-v1` | 360 | 4,320 |

The inputs were frozen with `prepare_terminalwrench_v7.py` under
`inputs/terminalwrench-neutral-v7-20261026/`. The `provenance.json` in each
model directory records local source episodes, hashes, model revisions, and
the ATIF-to-chat normalization. Six preflight requests (shortest and longest
prefix for each model) succeeded and their metadata is saved under
`experiment-results/emotion-self-report-preflight/terminalwrench-neutral-v7-20261026/transport.json`.

Run from `model-forensics` with the canonical virtual environment:

```sh
experiments/canonical/.venv/bin/python experiments/canonical/emotion-self-report/terminalwrench_dashboard.py --port 8797
bash experiments/canonical/emotion-self-report/run_terminalwrench_v7.sh gpt-oss-120b-terminalwrench-neutral-self-report-v2
bash experiments/canonical/emotion-self-report/run_terminalwrench_v7.sh gemma-3-4b-terminalwrench-neutral-self-report-v1
bash experiments/canonical/emotion-self-report/run_terminalwrench_v7.sh glm-5.3-terminalwrench-neutral-self-report-v1
```

The dashboard and three collection commands run in detached `tmux` sessions
named `tw-self-report-dash`, `tw-self-report-gptoss`, `tw-self-report-gemma`,
and `tw-self-report-glm`, so they continue after a Codex chat turn ends.
The three collection commands run separately. `run_terminalwrench_v7.sh`
retries a stopped canonical run at most twice, reusing verified responses.
Logs are in `/scratch/terminalwrench-self-report-v7-20261026/`; durable progress,
individual responses, and verification receipts remain in
`experiment-results/emotion-self-report/<run-id>/`. A completed collection
produces `verification.json`. The dashboard is read-only and does not control
the workers.

OpenRouter ratings use the same named model families, while the source traces
were produced by locally served quantized checkpoints. This serving difference
is retained in provenance and limits interpretation. No local activation-probe
scores have been attached to these runs; those require a separate replay and
exact checkpoint join after ratings are collected.

GPT-OSS v1 was stopped after Groq returned five identical successful HTTP
completions with `content=null` for one checkpoint, so the canonical collector
could not obtain a parseable rating and marked collection failed. The v1
responses remain frozen in their original run directory. A same-model request
to Crusoe returned a numeric rating for that exact checkpoint and for the
longest prefix. GPT-OSS v2 uses Crusoe only, a new run identity, and the same
frozen traces, checkpoints, and scientific prompt.
