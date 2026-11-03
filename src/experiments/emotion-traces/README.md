# Emotion traces around risky actions

Descriptive RH/control comparisons use regenerated per-token projections from
`emotion-rh-forecasting/`. They do not train additional emotion directions.

```bash
uv run python src/experiments/emotion-traces/task-averages/build_data.py
uv run python src/experiments/emotion-traces/task-averages/make_gallery.py
```

`task-averages/build_data.py` adapts available completed replay sets, uses the
original 128-token pre-event window, excludes whitespace/punctuation from rolling
means, centers each trace and matches controls within source/model. It writes
`runs/emotion-traces/task-averages/averages.json` and a pairing audit; `make_gallery.py`
renders those aggregates. Check available control counts before interpreting a
cell. Gemma 4B Coding has no labeled non-hack controls in the original sample.

`spikes/select_spikes.py` is a separate post-inspection example selector;
`spikes/render.py` renders its selected examples. These are not estimates of
population-average effects. `prepare_missing.py`, `load_missing.py` and
`search_stronger_coding.py` preserve historical replay/selection extensions and
need their explicit source artifacts. Do not synthesize absent scores or label
missing observations as zero.

The paper's aggregate figure data and figures are in the results download.
Original screenshot-digitized pressure redraws are deliberately not used as
measurements. See [the script reference](../../../docs/scripts.md).
