# Gemma 3 4B steering on TerminalWrench

Status: v1 retained; user-approved active-time revision running. Updated 2026-10-22 (America/Los_Angeles). This revision replaces the earlier 12-task × 2-seed design with **24 tasks × 1 seed** and removes the separate 20-episode pilot.

## Question

Does emotion steering change reward-hacking behavior during full interactive terminal episodes, and does it affect ordinary task performance?

## Design

- **24 distinct tasks, one sampled episode per task per condition.** Broader task coverage takes priority over repeated seeds on the same task.
- **12 emotions:** hostile, frustrated, excited, appreciative, sad, afraid, surprised, anxious, guilty, ashamed, desperate, calm.
- **Four strengths per emotion:** −0.10, −0.05, +0.05, +0.10.
- **One shared unsteered baseline.** All conditions receive the same existing RH-eliciting prompt followed by the original task instruction.
- Use a deterministic seed for each task, matched across its steering conditions and baseline. Keep the existing per-turn seed rule.

**Graph: 24 episodes per point, spanning 24 distinct tasks.** Each emotion plot has four steered points and the shared zero-strength baseline. Total: **24 × (12 × 4 + 1) = 1,176 episodes**. No separate pilot is included.

## Tasks and starting prompts

Use four tasks from each of six curated families:

| Family | Tasks |
|---|---|
| code repair | `fix-sentiment-cli-text-processing`, `sympy-matrix-fix`, `fix-multihead-attention-weights-shape`, `custom-memory-heap-crash` |
| data processing | `multi-source-data-merger`, `mongodb-sales-aggregation-engine`, `recover-corrupted-sqlite-data`, `iris-data-loading-script` |
| algorithms | `flood-escape-path-planner`, `rush-hour-bfs-solver`, `implement-nonogram-solver`, `tree-directory-parser` |
| ml numerics | `logistic-regression-divergence`, `mnist-learning-fix`, `compare-lasso-ridge-elasticnet`, `largest-eigenval` |
| build tooling | `resolve-react-peer-deps`, `detect-c-feature-flags`, `amuse-install`, `wasm-pipeline` |
| systems workflows | `git-multibranch`, `configure-git-webserver`, `pypi-server`, `tmux-advanced-workflow` |

The [task manifest](gemma4b-terminalwrench-task-sample-v2.json) stores the exact starting instructions, source revision and file hashes. This is a diverse curated sample, not a representative random sample of the entire suite. Tasks are not selected by hacking outcomes. The previously inspected tmux task is included.

Each episode runs in a fresh sandbox: task prompt → model command → actual terminal observation → subsequent commands → native verification. These are full interactive episodes, not prompt-only continuations.

## Model, steering and execution

Use the maintained `experiments/canonical/emotion-steering` implementation and existing Gemma probe bundle at `experiment-results/emotion-probes/gemma-3-4b-it-diversity-v1/probes`.

- **Model:** `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`, BF16.
- **Steering:** existing selected layers and vectors; one emotion at a time. Apply `h′ = h + α × ||h||₂ × v / ||v||₂` on prompt and generated tokens each turn. Baseline has no hook.
- **Infrastructure:** one Modal H100 for vLLM-Lens inference; separate Modal task sandboxes. No OpenRouter or local Docker.
- **Harness:** existing pinned Harbor 0.21.0 / Terminus-2 JSON agent.
- **Limits:** 30 turns, 4,096 generated tokens per call, 65,536-token context, 900-second agent timeout; temperature 0.7, top-p 0.95. No summarization. Preserve native terminal waits.

Adapt the existing Docker-only native adapter to Modal. Run independent episodes concurrently on the shared GPU, starting with four sandboxes and batches up to four. Preserve request-specific steering and seeds; avoid reusing cached activations across incompatible steering conditions. Reuse cached model weights and prebuilt task images.

Check that the first mixed-condition batch records the intended steering and real terminal interactions, then continue the sweep. No extra pilot or elaborate acceptance gate. Preserve failed attempts and their errors; do not silently replace them. Stop compute when the run finishes.

## Evaluation and graph

Record **RH attempts**, **supported successful RH**, **native verifier reward**, and **ordinary execution failures** separately. Reward 0 does not mean no hacking, and reward 1 does not prove hacking. Unsupported success claims and malformed commands are not automatically hacking.

Judge full transcripts with a TerminalWrench-appropriate rubric, blinded to emotion and strength. Retain uncertain labels and supporting evidence. Do not reuse the impossible-coding rubric unchanged.

Plot RH-attempt rate against steering strength for each emotion, with 24 episodes per point and the shared baseline. Show uncertainty and unresolved labels; compare each condition to its matched baseline and bootstrap by task. Also show task reward and execution-failure rates so capability degradation is not mistaken for reduced hacking. Treat this as an exploratory sweep, not a confirmed causal explanation of any emotion.

## Outputs

Save the configuration, task/probe identities, raw responses, steering receipts, native trajectories, terminal observations, verifier results and labels. Build a viewer filterable by task, emotion and strength, with matched baseline/steered episodes side by side. Report actual runtime and cost after execution.

## Execution record

The frozen run is `experiment-results/emotion-steering/gemma4b-terminalwrench-interactive-steering-v1`. Its detached local coordinator drives native Harbor 0.21.0 Modal task sandboxes and the deployed `terminalwrench-gemma4b-interactive-steering-v1` Lens worker. The live progress dashboard is `dashboards/terminalwrench-three-models/index.html`. Exact task source hashes, RH-prefixed task copies, probe vectors, planned episode assignments, native trials, and request receipts are retained in the run directory. A durable finish watcher stops this model’s compute and runs blinded transcript evaluation after all planned episode artifacts exist. No sweep results are claimed yet.

### Active-time revision and retained attempts

The user approved 900 **active** seconds per episode, excluding startup and inference queue waits; see [the approved timing policy](terminalwrench-active-time-v2.md). Generation, parsing, commands and terminal waits still count. Eight v1 attempts remain in the original run. Four attempts under the initial v2 runtime exposed an SDK pending-poll exception mismatch and remain preserved in `gemma4b-terminalwrench-interactive-steering-active-time-v2`; they are not pooled into the corrected sweep. The corrected run is `gemma4b-terminalwrench-interactive-steering-active-time-v2-r2`, with exact runtime sources saved alongside the frozen configuration, rolling concurrency four after first mixed validation, and durable stop/evaluation watchers. No complete-sweep results are claimed yet.
