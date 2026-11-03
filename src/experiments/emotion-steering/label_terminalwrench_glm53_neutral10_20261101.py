"""Label the ten new GLM TerminalWrench traces using the existing sentence rubric."""

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

from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import label_terminalwrench_v7_steering_sentences_v2 as batch

batch.OUT = HERE.parents[2] / 'runs/emotion-rh-sentence-labels/glm53-terminalwrench-neutral10-20261101-gpt61sol-v1'
batch.OUTPUT_TOKEN_CAP = 4096
batch.RUNS = {'GLM 5.3': 'glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7-neutral10-sample3-20261101'}

if __name__ == '__main__':
    batch.main()
