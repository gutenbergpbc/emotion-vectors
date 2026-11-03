# GPT-OSS v2 live timing contract

Deployment: `terminalwrench-gptoss120b-interactive-steering-v2`, class
`GPTOSSTerminalWrench`. GPU inference remains serialized on one H200.

Call `complete_timed.remote_gen(messages, probe, strength, seed, max_tokens,
request_id, active_seconds_remaining)` and consume every event:

- `generation_started`: emitted only after obtaining the same reentrant lock
  used by canonical inference. Resume the local active-time clock immediately.
  No further model-lock wait can occur. Startup and Modal admission queue occur
  before this event and must have an independent infrastructure watchdog.
- `generation_finished`: contains `output` and `active_generation_seconds`.
- `generation_failed`: contains error and active elapsed duration, followed by
  the original raised exception. A missing final result never pauses a clock
  that already received `generation_started`.

Events carry remote wall and monotonic timestamps for audit. Do not subtract
remote monotonic timestamps from local monotonic timestamps. Remote duration
comes from two readings of the same worker clock. The coordinator must preserve
its own receipt/clock transitions separately.

A per-token stopping criterion bounds generation by the supplied remaining
active budget, including prompt preparation after the start event. Generation
can overrun by one indivisible model forward pass. Method timeout is 1,200
seconds as a separate infrastructure backstop. Close/cancel the remote generator
when its consumer ends; even if cancellation is delayed during a forward pass,
the decode deadline prevents unbounded orphan generation. The generator releases
the lock on close/error and restores the original `model.generate` in `finally`;
canonical intervention hooks also remove themselves in `finally`.

`test_gptoss_terminalwrench_timing.py` verifies live queue exclusion, serialization,
cancellation before compute, failure timing, and remaining-budget decode cutoff
with restoration, without allocating a GPU.
