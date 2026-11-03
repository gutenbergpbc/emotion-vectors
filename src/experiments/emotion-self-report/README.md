# Emotion self-report

Ask one emotion question at each visible assistant checkpoint and compare saved
ratings to causal probe readouts. The recorded question uses a 0–10 scale.

`run.py --config CONFIG --dry-run` previews collection. `--through collect`
collects ratings without needing projection scores. Full analysis requires the
configured checkpoint-matched scores and their provenance. `config.example.json`
is a schema illustration; supply actual local input paths before running it.

For the paper's Gemma 4B results, the trace archive supplies whole-trace selections,
ratings and checkpoint spans. Run `prepare_gemma4b_activations.py`,
`modal_gemma4b_activations.py`, `finalize_gemma4b_activations.py`, then
`finalize_activations.py RUN_DIR`, as documented in the
[Gemma guide](../../../docs/reproduce-gemma4b.md). These regenerate last-completed-
assistant-token projections and correlations without new rating calls.

`emotion_self_report/inputs.py` defines visible-text inputs; `collection.py`
handles API requests and checkpoints; `analysis.py` computes correlations and
trace bootstrap intervals; `segmentation.py` preserves prose versus code units.
`select_activation_traces.py` freezes complete rated traces. `pooled/calculate.py`
is the historical cross-model Coding/Sales pooling analysis and requires the
corresponding per-model regenerated results; do not use it to fabricate missing
models. Use the per-dataset finalizer for the single-model walkthrough.

Raw projection is the primary comparison. Trace grouping, missing ratings,
checkpoint kind, provider identity and replay checkpoint mismatches must remain
visible. Code lines and prose checkpoints are not interchangeable.
