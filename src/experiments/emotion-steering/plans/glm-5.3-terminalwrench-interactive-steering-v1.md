# GLM 5.3 steering on TerminalWrench

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

- **Checkpoint:** existing `Inferact/GLM-5.3-NVFP4`, revision `ce67b36f3669192b5bb233819f0fda6c8a9837f8`. This plan uses NVFP4, not BF16.
- **Tokenizer:** `zai-org/GLM-5.3`, revision `ba32f8c19c8738e2c57c71798c601d477a7bf844`.
- **Compute:** one Modal allocation of **four B200 GPUs**, tensor parallelism 4. Reuse the existing vLLM 0.30.0 / vLLM-Lens 1.2.1 runtime, Transformers 5.17.0, CUDA 13.0.2 image and matching prebuilt FlashInfer kernels.
- **Probes:** `experiment-results/emotion-probes/glm-5.3-nvfp4-diversity-v2/probes`, retaining the saved selected layers, directions and selection provenance.
- **Channels:** existing native thinking-enabled, low-effort configuration. Save reasoning separately; only complete final action text reaches the terminal agent. The shared 4,096-token limit includes thinking, so reasoning-heavy truncations remain explicit outcomes.
- **Scheduling:** extend the interactive endpoint to use existing request-specific Lens batching, initially up to four requests across four independent sandboxes. Preserve tensor-parallel steering receipts and per-request hook isolation; the current serialized endpoint does not already provide this queue.

## Execution

Use the pinned Harbor 0.21.0 / Terminus-2 JSON agent and adapt the native Docker-only adapter to Modal, as in the Gemma plan. Both inference and task environments run on Modal; no OpenRouter or local Docker. Explicitly enforce the shared context and episode limits rather than inheriting larger model defaults.

Keep one model allocation loaded, reuse cached weights, prebuild task images and start with four independent task sandboxes. Preserve condition-specific steering, seeds and receipts. Do not reuse cached activations across incompatible interventions. Check the first mixed-condition work for real terminal interactions and correct steering, then continue; no extra pilot or elaborate gates. Keep failed attempts visible and stop compute at completion.

## Evaluation and outputs

Use the same blinded TerminalWrench rubric and judge configuration across models. Keep RH attempts, supported successful RH, native verifier reward, uncertain labels and ordinary execution failures separate. Reward alone is not a hacking label.

Plot RH-attempt rate versus strength for each emotion, with 24 episodes per point and this model's matched baseline. Show task-level uncertainty, unresolved labels, task reward and execution-failure rates. Bootstrap by task. Compare steering effects relative to each model's own baseline; different precisions, probe selection and reasoning behavior limit claims about cross-model differences.

Save raw responses, steering records, native trajectories, verifier results, labels and actual runtime/cost. Provide a task/emotion/strength viewer with matched baseline and steered episodes side by side.
