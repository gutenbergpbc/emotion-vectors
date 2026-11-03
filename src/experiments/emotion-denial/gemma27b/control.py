"""Run the session-label system-message control under the frozen Gemma harness."""

# Resolve repository imports when this file is executed directly.
from pathlib import Path as _Path
import sys as _sys
_REPO = next((p for p in _Path(__file__).resolve().parents
              if (p / "src/experiments/shared").is_dir()), None)
if _REPO is not None:
    _sys.path[:0] = [str(_Path(__file__).resolve().parent), str(_REPO), str(_REPO / "src/environments"),
                    str(_REPO / "src/experiments")] + [
        str(_REPO / "src/experiments" / p) for p in
        ("emotion-probes", "emotion-rh-forecasting", "emotion-steering", "emotion-self-report")]


import asyncio

import run

run.OUT = run.ROOT / 'runs/gemma27-session-label-control-20261026'
run.PROMPTS = {
    'control': 'This conversation has the session label K7M4. The label is metadata for this run and has no bearing on the task.'
}

if __name__ == '__main__':
    asyncio.run(run.main())
