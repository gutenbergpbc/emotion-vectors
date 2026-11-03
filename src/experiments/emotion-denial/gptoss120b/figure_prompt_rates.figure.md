# GPT-OSS 120B prompt comparison

Claim: Observed reward-hacking rates differ across the three prompts, with coding baseline generation settings differing materially from the prompt arms.

Template: Two-panel horizontal bar plot with a shared percentage scale.

Shows: The existing unsteered baseline and the two emotion prompts for sales and coding.

Accent: Amber for the two prompt conditions; slate for the existing baseline.

Text: Model, title, two panel names, prompt labels, six rates, axis label, and a short coding-baseline note.

Caption: Whole-trace judge labels across three repeated scenarios per task. Baseline: 156 sales traces and 96 coding traces; each prompt arm: 100 traces per task. Coding baseline used local forced-final generation with a 400-token cap, reached by 84/96 traces; new coding arms used OpenRouter reasoning and a 4,096-token cap. Sales and coding baseline providers and dates also differ from the prompt arms.

Sketch:

```text
                  Sales                    Coding
No prompt         ██████ 58.3%             ███ 27.1%*
Named emotions    ██████ 62%               █ 11%
General emotions  █████ 53%                2%
                  0–100%                   0–100%
```
