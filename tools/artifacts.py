#!/usr/bin/env python3
"""Fetch or verify release artifacts without model downloads or paid compute.

python tools/artifacts.py fetch --category datasets --from dist
python tools/artifacts.py fetch --category results --base-url RELEASE_ASSET_URL
python tools/artifacts.py verify --category probes
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def extract(path, asset, files, root=ROOT):
    """Reject unlisted members, links, traversal, duplicate names and corruption."""
    expected = set(asset['files'])
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        if len({m.name for m in members}) != len(members):
            raise ValueError('Duplicate archive members')
        if {m.name for m in members} != expected:
            raise ValueError('Archive members differ from release manifest')
        for member in members:
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not member.isfile():
                raise ValueError(f'Unsafe member: {member.name}')
            if member.size != files[member.name]['bytes']:
                raise ValueError(f'Unexpected member size: {member.name}')
        # Validate the complete archive before materializing any member.
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            for member in members:
                destination = stage/member.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, destination.open('wb') as output:
                    shutil.copyfileobj(source, output)
                if sha(destination) != files[member.name]['sha256']:
                    raise ValueError(f'Corrupt member: {member.name}')
                target = (root/member.name).resolve()
                if not target.is_relative_to(root.resolve()):
                    raise ValueError('Destination escapes repository through a symlink')
                if target.exists() and sha(target) != files[member.name]['sha256']:
                    raise ValueError(f'Refusing to replace changed local file: {target}')
            for member in members:
                target = root/member.name
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(stage/member.name, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fetch', 'verify'])
    parser.add_argument('--category', choices=['all', 'datasets', 'probes', 'traces', 'results'], default='all')
    parser.add_argument('--from', dest='local', type=Path, help='Local directory of release archives')
    parser.add_argument('--base-url', help='Override the published release download URL')
    args = parser.parse_args()
    files = json.loads((ROOT/'manifests/artifacts.json').read_text())['files']
    selected = {p:i for p,i in files.items() if args.category == 'all' or i['category'] == args.category}
    if args.command == 'fetch':
        releases = json.loads((ROOT/'manifests/releases.json').read_text())
        for asset in releases['assets']:
            if args.category != 'all' and asset['category'] != args.category:
                continue
            if all((ROOT/p).is_file() and sha(ROOT/p) == files[p]['sha256'] for p in asset['files']):
                continue
            with tempfile.TemporaryDirectory() as temp:
                path = Path(temp)/asset['name']
                if args.local:
                    shutil.copyfile(args.local/asset['name'], path)
                else:
                    url = args.base_url.rstrip('/')+'/'+asset['name'] if args.base_url else asset['url']
                    if not url:
                        parser.error('Release assets are prepared locally but not published; use --from dist or --base-url')
                    with urllib.request.urlopen(url) as response, path.open('wb') as output:
                        shutil.copyfileobj(response, output)
                if path.stat().st_size != asset['bytes'] or sha(path) != asset['sha256']:
                    raise ValueError(f'Archive checksum mismatch: {asset["name"]}')
                extract(path, asset, files)
                print('Fetched', asset['name'])
    failures = []
    for name, item in selected.items():
        path = ROOT/name
        if not path.is_file() or sha(path) != item['sha256']:
            failures.append(name)
    if failures:
        raise SystemExit('Missing/changed artifacts:\n'+'\n'.join(failures))
    print(f'Verified {len(selected)} {args.category} artifact files')


if __name__ == '__main__':
    main()
