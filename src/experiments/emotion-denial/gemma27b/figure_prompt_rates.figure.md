# Prompt comparison figure

Claim: The emotion-denial prompts have lower judged RH rates than the session-label control in sales; coding rates are similar.

Template: Two-panel horizontal bar plot with a shared percentage scale.

Shows: Four prompt conditions for Gemma 3 27B, once for sales and once for coding. Every cell has 100 generated traces.

Accent: Amber for the two emotion prompts; slate for the no-prompt and session-label references.

Image text: “GEMMA 27B / REWARD HACKING”, “Reward hacking by prompt”, “Sales”, “Coding”, four prompt labels (including “Neutral / no prompt”), eight percentages, “Judge-labeled RH rate”.

Caption: Whole-trace evidence-based judge labels among 100 traces per cell, across three fixed scenarios. The no-prompt cohort contains 12 local vLLM and 88 OpenRouter traces; the other cohorts use OpenRouter. Rates are descriptive and do not measure uncertainty across tasks.

Sketch:

```text
                  Sales                  Coding
No prompt         ███████ 73%            ██ 26%
Session label     ██████  63%            ███ 33%
Named emotions    ████    44%            ███ 29%
General emotions  █████   49%            ███ 28%
                  0–100%                 0–100%
```
