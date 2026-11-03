#!/usr/bin/env python3
"""Run this experiment independently; see README.md for inputs and commands."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
EXPERIMENTS = ROOT / 'src/experiments'
sys.path[:0] = [str(ROOT), str(ROOT/'src/environments'), str(EXPERIMENTS)] + [
    str(EXPERIMENTS/p) for p in ('emotion-probes', 'emotion-steering', 'emotion-rh-forecasting', 'emotion-self-report')]
from shared.cli import main
if __name__ == '__main__':
    main('emotion-probes')
