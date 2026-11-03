# Neutral TerminalWrench self-report baseline: full-trace cohort

## Question and scope

At fixed points in unsteered TerminalWrench trajectories, how do each model's
0–10 task-stance ratings relate to its own emotion-probe scores? The ratings are
a comparison measure, not ground-truth emotion. Analyze sentence trajectories
within traces before interpreting pooled correlations.

Use **44 completed neutral trajectories** from the frozen v7 base runs and the
eight-task neutral extensions: 21 GPT-OSS 120B, 12 Gemma 3 4B, and 11 GLM 5.3.
Take every eligible assistant sentence in every selected trajectory, including
its final turn. This broadens the initial four-task set to support within-trace
trend plots. The model cohorts have different task sets and sample numbers, so
cross-model comparisons are descriptive. No steered trajectory is eligible.

| Model | Full traces | Sentence checkpoints | 12-emotion rating requests | Probe values |
|---|---:|---:|---:|---:|
| GPT-OSS 120B | 21 | 850 | 10,200 | 10,200 |
| Gemma 3 4B | 12 | 787 | 9,444 | 9,444 |
| GLM 5.3 | 11 | 360 | 4,320 | 4,320 |
| **Total** | **44** | **1,997** | **23,964** | **23,964** |

All paths begin with `experiment-results/emotion-steering/`. The `v7` run is
`<model>-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7`,
where `<model>` is `gptoss120b`, `gemma4b`, or `glm53`; append
`-neutral-extension-8-20261025` for Gemma/GLM extension episodes. Resolve each
episode by `task.id`, rather than by the opaque episode filename. Select only
baseline tasks with `emotion=null`, `strength=0`, no `generation_error`,
`termination=complete`, at least two agent turns and at least ten canonical
`sentence_or_fragment` spans covering at least two turns. Deduplicate by
`(model, scenario, sample)` and reject conflicting duplicate content. Freeze
the full source path, SHA-256, task ID,
source run, model checkpoint/quantization, termination, and native artifact
hashes in the selection manifest. The source files currently have no
`generation_error`, report `termination=complete`, and contain agent turns;
recheck these and the native artifact hashes before freezing. Among the v7
sources, Gemma's `mnist-learning-fix` has zero eligible spans because its turns
are fenced JSON; `fix-sentiment-cli-text-processing` and
`flood-escape-path-planner` expose eligible spans in only one of 30 turns.
Those three are excluded. Gemma's `recover-corrupted-sqlite-data` is included:
19 of 21 turns have eligible spans, and the two gaps must be visible in its
trajectory plot. Earlier v2-v6 archives and the separate sample-2 cohort
are outside this frozen selection; they can be separate future cohorts.

## Execution plan

1. **Normalize exact visible histories.** Export one canonical `traces.json`
   per model from the ATIF `trajectory_text`. Preserve the initial user task,
   each assistant's *visible* `message`, its submitted tool calls, and the
   corresponding observations in chronological order. Do not include
   `reasoning_content`, future observations, task outcomes, labels, rewards,
   steering metadata, or probe values in requests. ATIF observations need an
   explicit adapter to the canonical `messages` format; do not flatten them
   into an invented conversation without recording the transformation. Verify
   that a checkpoint after assistant turn `j` contains exactly the prior turns
   and no result of turn `j`'s tool call. Keep `model` metadata tied to the
   actual query checkpoint, with local source-model identity separately noted.

2. **Freeze one checkpoint per eligible assistant sentence or terminal
   fragment.** Apply the canonical `segmentation.segment()` to every assistant
   message and keep its `sentence_or_fragment` spans, with exact original
   `message_index`, `char_start`, and `char_end`. Code blocks and headings stay
   in the replay prefix but are not targets. This yields **850 GPT-OSS, 787
   Gemma, and 360 GLM sentence checkpoints: 1,997 total**, hence **23,964
   self-report requests and 23,964 probe-score values** for all 12 emotions
   before any context-limit exclusions. These counts were computed from the
   selected ATIF visible messages using the current canonical segmenter; freeze
   the segmenter revision and recompute during preparation. Each request ends
   immediately after the target sentence, before later sentences or any
   observation from that assistant turn. Audit one early, middle, and final
   request per model by hand. If a provider cannot accept a full exact prefix,
   report that checkpoint as unavailable; do not silently truncate or switch
   models.

3. **Preflight model identity and transport.** The canonical collector calls
   OpenRouter and strictly checks the returned model ID. The steering traces
   came from local served model variants, so an OpenRouter rating is a
   *same-family black-box comparison*, unless the exact serving checkpoint,
   quantization, chat template, and reasoning policy can be matched. Record
   that distinction in each run's provenance. Verify that a live endpoint is
   available for `openai/gpt-oss-120b`, the actual Gemma 3 4B variant, and
   `z-ai/glm-5.3`, then send one frozen example request per model. Record
   provider, returned model, context fit, refusal/null behavior, and actual
   price. If the exact variant is only available through local serving, add a
   provider adapter with the same durable response and identity checks before
   collection; do not label a substituted model as that model's self-report.

4. **Run one small preflight and then the frozen collection.** For each model, create
   a separate canonical config and run ID under `experiments/canonical/configs/`.
   Use `environment=terminalwrench`, `prompt=task_stance`, all 12 emotions,
   temperature 0, and the same reasoning policy that the chosen endpoint
   supports. Use `run.py --dry-run`, then `--through prepare` to inspect the
   checkpoint list and example request. A separate new run ID with the first
   checkpoint only is the transport preflight. Run the full frozen collection with
   `--resume`; retain numeric zero, explicit null, and nonnumeric responses as
   distinct outcomes. Set an explicit cost ceiling before dispatch, based on
   the prepared payload sizes and preflight usage; stop rather than exceeding
   it. Do not reuse a pilot run ID for changed inputs or settings.

5. **Replay activations and pair exact checkpoints.** For each source model,
   replay the same normalized prefix through its local unsteered checkpoint
   and score all 12 existing probes at the last completed assistant token at
   or before `char_end`. Freeze probe bundle/layer, centering, quantization,
   tokenizer/chat template, source trace hashes, and token-to-character
   alignment. The probe's endpoint may lag the sentence endpoint but must
   never pass it. Export the canonical score CSV and provenance JSON; join on
   `(trace_id, unit_id, emotion)` and `message_index`. Do not interpret a
   provider-generated continuation as a byte-identical local activation.

6. **Report coverage before association.** Per model and emotion, show
   attempted/numeric/null/nonnumeric/context-unavailable counts, paired-score
   coverage, and rating distribution. Then report raw-projection Pearson and
   Spearman correlations with whole-trace bootstrap intervals using the
   canonical baseline. Plot each full trace in sentence order, with self-report
   and probe scores on separate axes; mark assistant-turn boundaries and
   sentence-coverage gaps. Add a normalized 0–1 trace-progress view for
   cross-trace trends, with the raw sentence count retained. Show task-level rows;
   compare models descriptively because traces, checkpoints, and provider
   conditions differ. An early/middle/final-turn sensitivity view can reveal
   whether a pooled association merely tracks task progress. Report trace
   counts and effective paired coverage with every coefficient; these are
   small, selected, unequal task cohorts.

7. **Validate and decide on expansion.** Run the canonical verifier, inspect
   hashes and exact prefix examples, independently spot-check score alignment,
   and retain response/provider receipts. A separate matched-task comparison
   should use the shared task intersection only. Expansion to the shared
   20-task sample-2 cohorts requires completed, validated episodes for all
   three models and a new frozen run ID; that cohort is outside this plan.

## Deliverables

Frozen selection manifest and normalization adapter; three trace and
checkpoint files; three canonical configs and verified self-report runs; three
probe-score/provenance exports; per-model coverage and association reports;
and a short cross-model limitations note. The existing canonical runner and
format are documented in `experiments/canonical/emotion-self-report/README.md`.
