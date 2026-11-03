# Emotion steering

Generate task responses while adding a signed, residual-norm-relative direction
at one selected post-block layer. Keep one unsteered baseline shared by emotions.

```bash
uv run python src/experiments/emotion-steering/run.py --config configs/gemma4b/steering-coding.json --dry-run
uv run python src/experiments/emotion-steering/run.py --config configs/gemma4b/steering-coding.json --through generate
uv run python src/experiments/emotion-steering/run.py --config configs/gemma4b/steering-coding.json --resume
```

Inputs: config JSON, selected probe bundle and task fixture. Sales additionally
uses `src/environments/sales_env/`. Output: frozen config, condition/seed plan,
responses, intervention receipts, judgments, plots and verification under
`runs/emotion-steering/<run-id>/`. A full 12-emotion Gemma config has 588 episodes.

`emotion_steering/config.py` validates checkpoint/vector identity and workload;
`backend.py` and `async_backend.py` implement generation; `environments.py` owns
Coding/Sales episodes; `judging.py` owns outcome criteria; `pipeline.py` owns
this experiment's resumable stages. `models.json` pins model-specific runtimes.
`modal_worker.py` launches GPU generation; judgments use paid API calls.

Steering applies h' = h + strength * norm(h) * unit(v) at a selected layer.
Positive direction means the stored orientation. Keep the configured output cap,
sampling settings, incomplete denominators, uncertainty and seed pairing.
Fresh generated outcomes need not match archived counts exactly.

TerminalWrench live steering requires its Harbor/Docker task infrastructure and
endpoint setup; archived TerminalWrench replay in the forecasting experiment is
a separate operation. Historical scheduler/recovery helpers are indexed in
[the script reference](../../../docs/scripts.md); the Gemma walkthrough uses
`run.py`, not those incident-specific helpers.
