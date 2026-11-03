# TerminalWrench active-time revision

User-approved revision, October 22, 2026 (America/Los_Angeles).

The three original plans used a 900-second wall-clock agent timeout. The first
four saved attempts for each of Gemma and GPT-OSS reached that timeout. Those
attempts and their native rewards remain v1 artifacts; they are not overwritten
or pooled into the revised sweep.

## Approved timing policy

Each v2 episode receives **900 seconds of active agent time**. Model startup and
inference-queue waits are excluded. Model generation, agent parsing, terminal
commands, and native terminal waits consume the active budget. Task sandbox setup
and verification retain their separate harness limits.

The 30-turn limit, 4,096 generated tokens per call, 65,536-token context, model
revisions, steering directions, task manifest, sampling parameters, and matched
task seeds remain as specified in the original model plans.

Queue exclusion must operate while requests are pending: subtracting a duration
only after a response arrives cannot prevent premature cancellation. A live
generation-start signal must resume active-time accounting before generation
finishes. Cross-machine monotonic timestamps must not be subtracted from each
other. Persist timing receipts and classify active-budget exhaustion separately
from infrastructure watchdog failures and native task reward.

Use a distinct run identity for each v2 model. Warm the model before starting
episode timers. After the first mixed-condition validation, use a rolling bounded
pool that admits a new episode whenever a slot becomes available. Preserve
request-specific steering and seeds. Keep independent infrastructure watchdogs
for startup, stuck queues, and total job lifetime; these are not extensions to the
900-second active budget.

Before launch, test that queue waits do not consume active budget, generation and
terminal work do consume it, generation is charged even when it never returns,
and a freed episode slot admits new work before a slower peer completes. The
resolved configuration and runtime receipts record concrete watchdog settings.

External judging remains authorized for evaluation only, with the same blinded
judge configuration across all three models.
