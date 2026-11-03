#!/usr/bin/env python3
"""Replay the paper's frozen pressure histories on a local GPU.

Inputs: data/pressure/<model>.json, including original denial wording,
twenty scenarios, selected probes, revisions and archived baseline token IDs.
Outputs: measurements, local residuals, runtime and verification in --output.
This experiment makes forward passes only. It never generates task responses.
"""
import argparse
import json
from pathlib import Path
import pressure_core as core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--model',choices=['gemma-3-27b-it','gpt-oss-120b'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    data=json.loads(args.inputs.read_text())
    core.CONDITIONS=data['conditions']
    assert set(core.CONDITIONS)=={'baseline','named_emotions','general_emotions'}
    count=len(data['fixtures']['sets'])*4*len(core.CONDITIONS)
    assert count==240 and len(data['emotion_probes'][args.model])==12
    if args.dry_run:
        print(json.dumps(dict(model=args.model,forwards=count,emotion_scores=count*12,
            conditions=core.CONDITIONS,revision=data['probes'][args.model]['revision']),indent=2))
        return
    core.run_worker(args.model,data,args.output.resolve(),lambda:None)


if __name__=='__main__':main()
