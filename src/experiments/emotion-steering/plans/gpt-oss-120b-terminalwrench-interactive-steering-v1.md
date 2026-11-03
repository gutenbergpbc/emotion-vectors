# GPT-OSS 120B steering on TerminalWrench

Status: planned, not launched. Created 2026-10-22. Same experimental setup as Gemma 3 4B.

## Shared experiment design

Follow the [Gemma experiment plan](gemma4b-terminalwrench-interactive-steering-v1.md), using the exact same [24-task manifest](gemma4b-terminalwrench-task-sample-v2.json), RH-eliciting instruction and task-to-seed mapping. The manifest filename is historical; its tasks are shared across all three models.

- Full interactive Harbor episodes in fresh Modal sandboxes, with real terminal feedback and native verification.
- **24 tasks × 1 seed per condition: 24 episodes per graph point.**
- **12 emotions:** hostile, frustrated, excited, appreciative, sad, afraid, surprised, anxious, guilty, ashamed, desperate, calm.
- Strengths **−0.10, −0.05, +0.05, +0.10**, plus one unsteered baseline shared across this model's emotion plots.
- **1,176 episodes per model**: 24 × (12 × 4 + 1). No separate pilot. Baseline trajectories are generated separately for each model, never pooled across models.
- Match each task's seed across conditions and models, retaining the deterministic per-turn seed offset. Matching seeds does not make outputs identical across models.
- **30 turns, 4,096 generated tokens per call, 65,536-token total context, 900-second agent timeout, temperature 0.7, top-p 0.95.** The output budget includes generated reasoning tokens where applicable. No summarization or silent truncation; preserve native terminal waits.

## Steering

Reuse the maintained `experiments/canonical/emotion-steering` implementation. Apply one model-specific emotion vector at its existing selected decoder block: `h′ = h + α × ||h||₂ × v / ||v||₂`, on prompt and generated tokens every turn. Baseline installs no hook. Do not reuse another model's vectors or retune layers on these task outcomes.

## Model and runtime

- **Checkpoint/tokenizer:** `openai/gpt-oss-120b`, revision `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`, original MXFP4 weights.
- **Compute:** one Modal H200; existing Transformers/PyTorch steering backend, not an assumed vLLM port. Reuse the pinned preset in `models.json` (PyTorch 2.8.0, Transformers 4.57.6 and its pinned MXFP4 kernels).
- **Probes:** `experiments/canonical/emotion-steering/data/gpt-oss-120b-probes.json`, which contains all 12 selected emotion directions. Its historical selection used held-out labels; retain that exploratory selection-bias disclosure.
- **Channels:** native interactive analysis/final handling with low reasoning effort. Do not use the coding-only forced-final prefix. Save reasoning separately and send only the final action text to the terminal agent.
- **Scheduling:** the current backend is single-request and uses model-global hooks. Serialize model calls while four sandboxes overlap terminal work. Do not concurrently apply different hooks to this model instance. Four-request GPU batching would require additional backend work and is not assumed here.

## Execution

Use the pinned Harbor 0.21.0 / Terminus-2 JSON agent and adapt the native Docker-only adapter to Modal, as in the Gemma plan. Both inference and task environments run on Modal; no OpenRouter or local Docker. Explicitly enforce the shared context and episode limits rather than inheriting larger model defaults.

Keep one model allocation loaded, reuse cached weights, prebuild task images and start with four independent task sandboxes. Preserve condition-specific steering, seeds and receipts. Do not reuse cached activations across incompatible interventions. Check the first mixed-condition work for real terminal interactions and correct steering, then continue; no extra pilot or elaborate gates. Keep failed attempts visible and stop compute at completion.

## Evaluation and outputs

Use the same blinded TerminalWrench rubric and judge configuration across models. Keep RH attempts, supported successful RH, native verifier reward, uncertain labels and ordinary execution failures separate. Reward alone is not a hacking label.

Plot RH-attempt rate versus strength for each emotion, with 24 episodes per point and this model's matched baseline. Show task-level uncertainty, unresolved labels, task reward and execution-failure rates. Bootstrap by task. Compare steering effects relative to each model's own baseline; different precisions, probe selection and reasoning behavior limit claims about cross-model differences.

Save raw responses, steering records, native trajectories, verifier results, labels and actual runtime/cost. Provide a task/emotion/strength viewer with matched baseline and steered episodes side by side.
