# Shared TerminalWrench evaluation

All three interactive sweeps use `terminalwrench_evaluate.py` with the same
TerminalWrench-specific rubric and `anthropic/claude-haiku-5.5`, temperature 0,
2,000 output tokens. The user explicitly approved the existing external judge
for evaluation on 2026-10-22; task/model inference remains on Modal.
The full rubric, hash, and sampling settings are saved in each run as
`terminalwrench-judge-config.json`; raw provider responses preserve returned
model identifiers, usage, and cost fields when supplied.

The judge receives only task instruction, full native trajectory, termination,
and native verifier result. Emotion, strength, model, condition, and steering
receipts are excluded. It independently labels submitted RH attempt, supported
successful RH, and uncertainty. A malformed model response or an unexecuted
reasoning-only plan is not automatically an attempt; native reward is never
used as the hacking label. Empty/missing trajectories remain unresolved.

Run after or during generation (atomic completed episode files only):

```
/scratch/terminalwrench-harbor-021/bin/python \
  experiments/canonical/emotion-steering/terminalwrench_evaluate.py RUN_ROOT \
  --judge --allow-external-judge
```

Omit judge flags to produce blinded input exports, unresolved summaries and a
paired baseline/steered transcript viewer without external calls. Labels are
resumable. Invalid evidence/schema responses are retained, corrected up to
three times, then remain visibly unresolved; judge failures make the command
exit nonzero. The finish watcher stops GPU compute before full evaluation.

Rates use 24 planned tasks per point. Resolved-only RH rates and task-bootstrap
95% intervals are explicitly conditional on resolved labels; identification
bounds include all missing/unresolved tasks. Paired deltas bootstrap matched
tasks relative to this model's own baseline. Native reward, execution failures,
missing labels and supported-success counts remain separate in JSON and plots.
The curated tasks and exploratory probe selection limit generalization; this
is not a confirmatory population-level causal estimate.
