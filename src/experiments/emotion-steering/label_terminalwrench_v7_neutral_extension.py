"""Sentence-label the 16 new neutral v7 extension traces with GPT-6.1-Sol.

Prepare only: uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_neutral_extension.py --prepare-only
Run:         uv run --with pysbd python src/experiments/emotion-steering/label_terminalwrench_v7_neutral_extension.py --workers 1
"""

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
import label_terminalwrench_v7_steering_sentences_v2 as batch  # noqa: E402

ROOT = HERE.parents[2]
batch.OUT = ROOT / "runs/emotion-rh-sentence-labels/terminalwrench-v7-neutral-extension-gpt61sol-v2"
batch.OUTPUT_TOKEN_CAP = 2048
batch.RUNS = {
    "GLM 5.3": "glm53-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7-neutral-extension-8-20261025",
    "Gemma 4B": "gemma4b-terminalwrench-interactive-steering-active-time-v2-rh-top4-continuous-v7-neutral-extension-8-20261025",
}

if __name__ == "__main__":
    batch.main()
