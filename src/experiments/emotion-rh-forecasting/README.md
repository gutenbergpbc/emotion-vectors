# Emotion reward-hacking forecasting

Predict label i+k using twelve frozen emotion readouts at source unit i. The
individual experiment fits scalar cutoffs/polarities in training folds; the
separate combined experiment uses training-fold standardization and fixed L2
logistic regression. Neither fits the emotion directions on RH labels.

Start with sections 2–4 of [the Gemma 4B guide](../../../docs/reproduce-gemma4b.md).
Inputs are selected probes, original text traces, RH labels and source groups.
Scores are regenerated locally; no token-score archive is distributed.

- `generate_openrouter_traces.py`: optional paid fresh Coding/Sales collection;
  `--output` separates it from archived evidence; `--dry-run` previews inputs.
- `label_gemma4b_sales.py`, `label_gemma4b_coding_lines.py`: original cohort/rubric
  and paid labeling. Coding units are code lines, Sales units are prose.
- `prepare_gemma4b_*_replay.py`: verify text/spans and create native replay jobs.
- `modal_gemma4b_*_replay.py`: GPU projection capture with runtime receipts.
- `plot_gemma4b_*_forecast.py`: individual forecasts using regenerated projections.
- `run.py --config CONFIG`: independent configurable individual forecast runner.
- `combined_emotions.py RUN_DIR`: all-twelve logistic forecasts after individual
  results. `render_combined.py` and `plot_combined_roc.py` render saved results.

`emotion_rh_forecasting/inputs.py` validates portable and historical contracts;
`features.py` pools only causally completed tokens; `analysis.py` owns group
isolation, horizon pairing, balancing and cutoffs. `terminalwrench_replay.py`
implements native template replay, not live terminal task execution.

Historical model-specific preparation scripts retain explicit input/output
constants and require the corresponding text artifacts and newly generated
replays. `configs/` includes reference settings for those models. An input path
under `runs/.../scores` is a runtime prerequisite, not a missing release asset.

Horizon zero is contemporaneous. Later labels can be continued hacking rather
than first onset. Bands quantify majority-class subsampling, not confidence over
new tasks. Full-trajectory judge labels are retrospective. Preserve these
qualifications and the recorded judge/model identities when extending the code.

See [the script reference](../../../docs/scripts.md) for every helper.
