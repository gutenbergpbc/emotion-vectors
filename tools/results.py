#!/usr/bin/env python3
"""List downloaded published metrics and figures, without recomputation."""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contains',default='',help='Filter result paths')
    args=parser.parse_args()
    files=json.loads((ROOT/'manifests/artifacts.json').read_text())['files']
    for name,item in sorted(files.items()):
        if item['category']=='results' and args.contains in name:
            print(('available' if (ROOT/name).exists() else 'download required')+'\t'+name)
