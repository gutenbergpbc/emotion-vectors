# Emotion-denial prompts under scripted pressure

This is a forward-pass experiment on twenty authored scenarios, four cumulative
pressure checkpoints and three prompt conditions. It generates no task responses
and measures no behavioral reward-hacking rate. Original completed models are
Gemma 27B and GPT-OSS 120B, not Gemma 4B.

```bash
uv run python src/experiments/emotion-pressure/run.py --inputs data/pressure/gemma-3-27b-it.json --model gemma-3-27b-it --output runs/pressure/gemma27b --dry-run
```

Remove `--dry-run` only in a compatible local CUDA environment. The archived
source runtime uses Torch 2.8.0, Transformers 4.57.6, Accelerate 1.13.0 and NumPy
2.4.4; GPT-OSS additionally uses Triton 3.4.0 and kernels 0.11.7. Use a separate
pinned environment rather than replacing the repository's Lens runtime.

`run.py` loads the original prompt conditions from the immutable source input;
`pressure_core.py` renders native prefixes, checks baseline token identities,
measures fixed readouts and computes scenario-paired bootstrap estimates.
`verify.py RUN --model MODEL` independently recomputes from runtime residuals.
Outputs include 240 forwards and 2,880 emotion scores per model, runtime details,
summary and verification. Local residuals stay out of the release.

The copied historical `build_report.py` and `check_viewer.py` target the later
resistance-prompt presentation and are supporting references, not the original
paper's runner. Original denial wording must not be replaced with later prompts.
Raw projections are uncalibrated and this prefix-token readout differs from the
marked-sentence means used for training. Scripted pressure also changes wording,
history length and stakes.
