# OpenRouter coding and sales trace collection

`generate_openrouter_traces.py` collects unsteered, unlabeled assistant output for
the sentence-level RH experiment. It uses the three frozen impossible coding
prompts from the GPT-OSS coding run and the three frozen TechFlow sales scenarios
(appreciative, neutral, hostile) from the GPT-OSS sales run. Coding requests are
distributed 34/33/33 for a 100-trace batch; sales requests are distributed
50/50/50 for a 150-episode batch. Each sales episode runs the existing
`sales_env.sales_environment.SalesEnvironment` for at most eight turns.

```bash
uv run --with pandas --with aiohttp python experiments/canonical/emotion-rh-forecasting/generate_openrouter_traces.py --model glm53 --dataset coding --count 100
uv run --with pandas --with aiohttp python experiments/canonical/emotion-rh-forecasting/generate_openrouter_traces.py --model gemma3-4b --dataset coding --count 100
uv run --with pandas --with aiohttp python experiments/canonical/emotion-rh-forecasting/generate_openrouter_traces.py --model glm53 --dataset sales --count 150
uv run --with pandas --with aiohttp python experiments/canonical/emotion-rh-forecasting/generate_openrouter_traces.py --model gemma3-4b --dataset sales --count 150
```

`OPENROUTER_API_KEY` must be in the devbox secrets. The script resumes by
skipping saved trace IDs. Its output is in
`experiment-results/emotion-rh-traces/<model>-<dataset>-openrouter-v1/`, with
one JSON file per trace, a frozen run specification, and a summary. Every
successful request retains the prompt, returned model/provider, full response,
reasoning if exposed by OpenRouter, finish reason, and token/cost usage. Sales
episodes also retain all feedback and ledger states. `max_turns` episodes are
saved but are distinguished from finalized episodes. No RH judgments or probe
features are generated in this step.

If fewer than 150 sales episodes finalize, extend the same run without changing
its prompts or generation settings, for example with `--count 225 --extend`.
The run specification records each extension. After collection,
`python experiments/canonical/emotion-rh-forecasting/audit_openrouter_traces.py`
verifies IDs, returned models and counts, then freezes the first 50 finalized
episodes per tone in `cohort-finalized-150.json`. Selection uses no RH labels
or probe scores.

Gemma 3 4B is an instruct model and may return no separate reasoning channel.
For such traces the future sentence experiment can only use prose in the final
response. Keep these traces separate from GLM and GPT-OSS when reporting sample
sizes and interpretation.

## October 21–22 collection

The original GLM 5.3 and Gemma 3 4B coding batches each contained 100 saved
traces. GLM sales contains 150 finalized episodes (50 per tone). The initial
150 Gemma sales attempts yielded 106 finalized and 44 eight-turn episodes.
The collection was extended to 225 attempts, yielding 160 finalized episodes;
`cohort-finalized-150.json` selects 50 finalized episodes per tone. The other
75 attempts remain available for audit but are outside that frozen cohort.
No sentence RH labels or probe scores have been attached yet. The separate
`audit.json` files contain exact provider, cost, reasoning-availability and
completion counts.

The GLM coding run was subsequently extended from 100 to **400** complete
traces with the same prompts and settings (134 primality, 133 search, 133
sorting). The first 100 retain their original IDs and bytes. The existing
whole-answer Claude Haiku 5.5 judge was subsequently run on the additional 300:
10 hack, 13 ambiguous, and 277 honest/impossible. All 400 trace judgments have
matching source hashes in `whole-answer-hack-judgments/summary.json`. These
are whole-answer categories; sentence-level RH labels remain uncollected.
