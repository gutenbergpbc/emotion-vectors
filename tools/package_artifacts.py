#!/usr/bin/env python3
"""Build deterministic release archives from manifests/artifacts.json.

python tools/package_artifacts.py [--base-url https://.../releases/download/v1]
Archives contain only enumerated payloads, never a recursive copy of runs/.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', help='Published release asset base URL; omit while preparing locally')
    parser.add_argument('--category', choices=['all','datasets','probes','traces','results'], default='all')
    parser.add_argument('--shard-mb', type=int, default=512, help='Approximate uncompressed archive size')
    args = parser.parse_args()
    if args.shard_mb <= 0:
        parser.error('--shard-mb must be positive')
    files = json.loads((ROOT/'manifests/artifacts.json').read_text())['files']
    out = ROOT/'dist'; out.mkdir(exist_ok=True)
    previous = json.loads((ROOT/'manifests/releases.json').read_text())['assets'] if (ROOT/'manifests/releases.json').exists() else []
    assets = [a for a in previous if args.category != 'all' and a['category'] != args.category]
    for category in ('datasets', 'probes', 'traces', 'results'):
        if args.category != 'all' and category != args.category:
            continue
        groups, batch, size = [], [], 0
        for name, item in sorted(files.items()):
            if item['category'] != category:
                continue
            path = ROOT/name
            if sha(path) != item['sha256']:
                raise ValueError(f'Changed payload: {name}')
            if path.suffix in {'.npy', '.npz', '.pt', '.safetensors'} and category != 'probes':
                raise ValueError(f'Regenerable array in release: {name}')
            if batch and size + item['bytes'] > args.shard_mb*1024*1024:
                groups.append(batch); batch, size = [], 0
            batch.append(name); size += item['bytes']
        if batch:
            groups.append(batch)
        for index, names in enumerate(groups, 1):
            name = f'{category}-v1-{index:03d}.tar.gz'
            target = out/name
            with target.open('wb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|') as archive:
                    for relative in names:
                        path = ROOT/relative
                        info = tarfile.TarInfo(relative)
                        info.size = path.stat().st_size
                        info.mode = 0o644
                        info.mtime = info.uid = info.gid = 0
                        with path.open('rb') as stream:
                            archive.addfile(info, stream)
            if target.stat().st_size >= 2*1024**3:
                raise ValueError(f'Archive exceeds release asset limit: {target}')
            assets.append(dict(name=name, category=category, bytes=target.stat().st_size,
                sha256=sha(target), files=names,
                url=args.base_url.rstrip('/')+'/'+name if args.base_url else None))
            print(f'{name}: {target.stat().st_size:,} bytes; {len(names)} files', flush=True)
    manifest = dict(schema_version=1, published=bool(args.base_url), assets=assets)
    (ROOT/'manifests/releases.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
