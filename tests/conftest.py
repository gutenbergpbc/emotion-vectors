from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "src/experiments"
sys.path[:0] = [str(ROOT), str(ROOT / "src/environments"), str(EXPERIMENTS)] + [
    str(EXPERIMENTS / p) for p in ("emotion-probes", "emotion-rh-forecasting", "emotion-steering", "emotion-self-report")]
