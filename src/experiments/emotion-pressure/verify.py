#!/usr/bin/env python3
"""Recheck pressure-run residuals and statistics: verify.py RUN --model MODEL.
Residuals are runtime intermediates, deliberately absent from release archives.
"""
import argparse
import json
from pathlib import Path
import pressure_core as core

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--model', required=True, choices=['gemma-3-27b-it', 'gpt-oss-120b'])
    args = parser.parse_args()
    data = json.loads((args.run/'inputs.json').read_text())
    core.CONDITIONS = data['conditions']
    rows = [json.loads(s) for s in (args.run/'measurements.jsonl').read_text().splitlines()]
    assert len(rows) == 240
    receipt = core.verify_rows(rows, data, args.model, args.run)
    summary = core.summarize(rows, data['emotion_probes'][args.model], True)
    assert summary == json.loads((args.run/'summary.json').read_text())
    print(json.dumps(receipt, indent=2))
