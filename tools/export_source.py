#!/usr/bin/env python3
"""Materialize release inputs from the original research checkout.

Maintainer command: python tools/export_source.py --source /path/to/model-forensics
Writes portable datasets, selected probe bundles, text traces, published results,
and source/coverage manifests. Never exports activation or token-score arrays.
Original evidence bytes are retained; relocated indexes are recorded separately.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
FILES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def retain(source, destination, category, source_root):
    source, destination = Path(source), Path(destination)
    if not source.is_file() or source.name.startswith('._'):
        return
    assert source.resolve().is_relative_to(source_root.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    checksum = sha(source)
    if not destination.exists() or sha(destination) != checksum:
        shutil.copy2(source, destination)
    FILES[str(destination.relative_to(ROOT))] = dict(
        category=category, bytes=destination.stat().st_size, sha256=checksum,
        source=str(source.relative_to(source_root)))


def relocated(value):
    if isinstance(value, dict):
        return {k: relocated(v) for k, v in value.items()}
    if isinstance(value, list):
        return [relocated(v) for v in value]
    if isinstance(value, str):
        return value.replace('experiment-results/', 'runs/')
    return value


def generated(path, value, category, source=None):
    write(path, value)
    FILES[str(path.relative_to(ROOT))] = dict(category=category,
        bytes=path.stat().st_size, sha256=sha(path), source=source,
        transformed=True)


def datasets(source):
    inputs = source / 'probe-harness/unified-twelve-model-replication-v1/inputs'
    for p in inputs.glob('*.json'):
        retain(p, ROOT / 'data/datasets/diversity-sales' / p.name, 'datasets', source)
    inventories = [source / 'sandbox/ayush/emotion_vectors/stories/astra/phase1b/emotion_probe_pipeline_v1/DATA_INVENTORY.md',
                   source / 'probe-harness/unified-twelve-model-replication-v1/guide/DATA_INVENTORY.md']
    coverage = []
    excluded = re.compile(r'(token_trace|token.trace|activations|scores|calibrat|evaluation_results|plot_data|nested_cv|models|vectors|cache)', re.I)
    for number, inventory in enumerate(inventories):
        retain(inventory, ROOT / 'data/datasets/catalog' / f'inventory-{number}.md', 'datasets', source)
        for link in dict.fromkeys(re.findall(r'\]\(([^)]+)\)', inventory.read_text())):
            link = link.strip('<>').split('#')[0]
            original_link = link
            if link.startswith(('http://127.0.0.1:', 'http://localhost:')):
                link = urlparse(link).path.lstrip('/')
            if not link or '://' in link:
                coverage.append(dict(inventory=number, link=link, status='external_reference'))
                continue
            candidates = [inventory.parent / link, source / link,
                source / 'sandbox/ayush/emotion_vectors' / link,
                inventories[0].parent / link]
            # The historical prose inventory contains two stale relative names.
            # Package actual preserved sources, never fabricate the missing file.
            corrected = None
            if 'hostile_diversity_regenerated_v2/episodes.jsonl' in link:
                corrected = source/'sandbox/ayush/emotion_vectors/stories/astra/phase1b/hostile_diversity_regenerated_v2'
            elif 'full_v2_token_length/stories.jsonl' in link:
                corrected = source/'sandbox/ayush/emotion_vectors/runs/qwen35-9b-hostility-20260924/phase1/full_v2_token_length/stories.jsonl'
            if corrected is not None:
                candidates.append(corrected)
            target = next((p.resolve() for p in candidates if p.exists() and p.resolve().is_relative_to(source)), None)
            if target is None:
                coverage.append(dict(inventory=number, link=link, status='missing_source'))
                continue
            if excluded.search(link) or target.suffix in {'.svg', '.png', '.pdf'}:
                coverage.append(dict(inventory=number, link=link, status='derived_artifact_excluded'))
                continue
            paths = [target] if target.is_file() else sorted(target.rglob('*'))
            retained = []
            for p in paths:
                if not p.is_file() or p.suffix not in {'.json', '.jsonl', '.md', '.txt', '.html'}:
                    continue
                if excluded.search(p.name) or any(k in p.parts for k in ['scores', 'activations', '__pycache__']):
                    continue
                destination = ROOT / 'data/datasets/catalog/source' / p.relative_to(source)
                retain(p, destination, 'datasets', source)
                if not p.name.startswith('._'):
                    retained.append(str(destination.relative_to(ROOT)))
            row=dict(inventory=number, link=link,
                status='included' if retained else 'derived_artifact_excluded', files=retained)
            if original_link != link:
                row['original_link']=original_link
            if corrected is not None and target==corrected.resolve():
                row['resolution_note']='Stale inventory reference; original preserved source files are included at their actual names.'
            coverage.append(row)
    # Story corpora used by mixed-family best probes are later than the inventory.
    for family in ['gemma_self_story_topicmix', 'gpt_oss_self_story_topicmix']:
        for p in (source / 'output' / family).rglob('*'):
            if p.is_file() and p.suffix in {'.json', '.jsonl'} and any(k in p.name for k in ['stories', 'corpus', 'topic', 'generation', 'review', 'prompt']):
                if not excluded.search(p.name):
                    retain(p, ROOT / 'data/datasets/catalog/source' / p.relative_to(source), 'datasets', source)
    write(ROOT / 'manifests/catalog-coverage.json', coverage)
    for old, new in [
        ('experiments/gpt_oss_sales_sentence_labels_20261016/source-cache/gpt-oss-120b-sales-four-emotions-b32-r2-20261016/run_spec.json', 'data/tasks/sales-source.json'),
        ('experiments/gpt_oss_sentence_rh_12probes_20261016/cases.json', 'data/tasks/coding-source.json')]:
        retain(source/old, ROOT/new, 'datasets', source)
    return Counter(row['status'] for row in coverage)


def probes(source):
    import numpy as np
    exports = []
    primary = {
        'gemma4b': 'gemma-3-4b-it-diversity-v1',
        'glm52': 'glm-5.2-nvfp4-diversity-v2',
        'glm53': 'glm-5.3-nvfp4-diversity-v2',
    }
    bundles = []
    for slug, run in primary.items():
        directory = source / 'experiment-results/emotion-probes' / run / 'probes'
        manifest = json.loads((directory / 'manifest.json').read_text())
        selected = json.loads((directory / 'selected-12.json').read_text())
        rows = []
        with np.load(directory / 'vectors.npz', allow_pickle=False) as z:
            for emotion, key in selected['probes'].items():
                m = manifest['models'][key]
                rows.append(dict(emotion=emotion, probe=key, layer=m['layer'],
                    vector=z[key].tolist(), intercept=m.get('intercept', 0),
                    threshold=m.get('threshold'), centering=manifest.get('centering'),
                    training_family='diversity', contrast=m.get('recipe', key.split('__')[-1]),
                    source=str(directory.relative_to(source)), source_sha256=sha(directory / 'vectors.npz')))
        bundles.append((slug, dict(schema_version=1,
            model={k: manifest.get(k, manifest['model'] if k == 'tokenizer_model' else manifest['revision'])
                   for k in ['model', 'revision', 'tokenizer_model', 'tokenizer_revision']},
            probes=rows, selection=selected['method'])))
    for slug, catalog_name in [('gemma27b', 'gemma-3-27b'), ('gptoss120b', 'gpt-oss-120b'), ('qwen35-9b', 'qwen35-9b')]:
        p = source / f'experiments/canonical/emotion-steering/data/{catalog_name}-probes.json'
        catalog = json.loads(p.read_text()); rows = []
        for item in catalog['probes']:
            pth = source / item['source']; key = item['probe'].split('::')[-1]
            assert sha(pth) == item['source_sha256']
            if pth.suffix == '.npy':
                v = np.load(pth, allow_pickle=False)
                metadata = {}
            elif pth.suffix == '.npz':
                with np.load(pth, allow_pickle=False) as z:
                    v = z[key]
                    if v.ndim == 2:
                        v = v[item['layer_zero_based']]
                metadata = {}
            else:
                metadata = json.loads(pth.read_text())['models'][key]
                v = np.asarray(metadata['vector'])
            rows.append(dict(emotion=item['emotion'], probe=key,
                layer=item['layer_zero_based'], vector=v.tolist(), intercept=metadata.get('intercept', 0),
                threshold=metadata.get('threshold'), centering=item.get('centering'),
                training_family='self_story' if 'story' in item['source'] or 'story' in item.get('catalog_probe', '') else 'diversity',
                contrast=metadata.get('recipe', key.split('__')[-1]), provenance=item))
        bundles.append((slug, dict(schema_version=1,
            model=dict(name=catalog['model'], revision=catalog['revision'],
                       tokenizer_model=catalog['model'], tokenizer_revision=catalog['revision']),
            probes=rows, selection=catalog['selection'])))
    for slug, bundle in bundles:
        model = bundle['model']
        model['name'] = model.pop('model', model.get('name'))
        assert len(bundle['probes']) == 12 and len({p['emotion'] for p in bundle['probes']}) == 12
        folder = ROOT / 'data/probes' / slug
        generated(folder / 'probes.json', bundle, 'probes')
        selected = {p['emotion']: p['probe'] for p in bundle['probes']}
        generated(folder / 'selected-12.json', {'probes': selected, 'method': bundle['selection']}, 'probes')
        models = {p['probe']: {k: v for k, v in p.items() if k != 'vector'} for p in bundle['probes']}
        np.savez_compressed(folder / 'vectors.npz', **{p['probe']: np.asarray(p['vector'], np.float32) for p in bundle['probes']})
        FILES[str((folder/'vectors.npz').relative_to(ROOT))] = dict(category='probes', bytes=(folder/'vectors.npz').stat().st_size, sha256=sha(folder/'vectors.npz'))
        generated(folder / 'manifest.json', dict(schema_version=1, synthetic=False,
            model=model['name'], revision=model['revision'], tokenizer_model=model['tokenizer_model'],
            tokenizer_revision=model['tokenizer_revision'], centering='per-probe; see models',
            models=models, files={n: sha(folder/n) for n in ['vectors.npz','selected-12.json']}), 'probes')
        exports.append(dict(id=slug, path=f'data/probes/{slug}', model=model,
            emotions=12, selection=bundle['selection'], mixes={p['emotion']: {k:p.get(k) for k in ['training_family','contrast','centering','layer']} for p in bundle['probes']}))
    write(ROOT / 'manifests/probes.json', exports)


def traces(source):
    categories = ['emotion-rh-traces', 'emotion-rh-sentence-labels', 'emotion-rh-code-line-labels']
    for category in categories:
        paths = ((source / 'experiment-results' / category).rglob('*.json')
                 if category == 'emotion-rh-traces' else
                 [p for name in ['labels.json','selection.json','manifest.json','summary.json']
                  for p in (source / 'experiment-results' / category).glob('*/'+name)])
        for p in paths:
            if any(k in p.parts for k in ['replay', 'scores', 'inputs']) or p.stat().st_size > 150_000_000:
                continue
            rel = p.relative_to(source / 'experiment-results')
            retain(p, ROOT / 'data/traces/original' / rel, 'traces', source)
            # Source paths move, while raw assistant/trace bytes remain unchanged.
            d = json.loads(p.read_text())
            if p.name in {'labels.json', 'cohort-finalized-150.json'}:
                generated(ROOT / 'runs' / rel, relocated(d), 'traces', str(p.relative_to(source)))
            else:
                retain(p, ROOT / 'runs' / rel, 'traces', source)
    for category in ['emotion-self-report', 'emotion-steering']:
        for run in (source / 'experiment-results' / category).iterdir():
            if not run.is_dir():
                continue
            for directory in (['responses', 'traces', 'judgments'] if category == 'emotion-steering' else []):
                for p in (run / directory).rglob('*.json'):
                    if not p.name.startswith('._'):
                        retain(p, ROOT / 'runs' / p.relative_to(source / 'experiment-results'), 'traces', source)
            for name in ['cases.json', 'checkpoints.json', 'ratings.json', 'combined-ratings.json', 'manifest.json', 'config.resolved.json']:
                p = run / name
                if p.exists():
                    generated(ROOT / 'runs' / p.relative_to(source / 'experiment-results'), relocated(json.loads(p.read_text())), 'traces', str(p.relative_to(source)))
    for name in ['gemma27-emotion-denial-twoarms-20261026','gptoss120b-emotion-denial-lowreason-20261027','gemma27-session-label-control-20261026']:
        for p in (source/'experiment-results'/name).rglob('*.json'):
            retain(p,ROOT/'runs'/p.relative_to(source/'experiment-results'),'traces',source)
    # TerminalWrench source traces and frozen folds are embedded in these inputs.
    for p in (source/'experiments/canonical/emotion-self-report/inputs').rglob('*.json'):
        retain(p, ROOT/'data/traces/terminalwrench'/p.relative_to(source/'experiments/canonical/emotion-self-report/inputs'), 'traces', source)
    for p in (source/'experiment-results/emotion-rh-replay').glob('*/dataset-traces.json'):
        retain(p,ROOT/'runs'/p.relative_to(source/'experiment-results'),'traces',source)
    for slug in ['gemma-3-27b-it','gpt-oss-120b']:
        p=source/f'experiments/emotion_pressure_resistance_20261102/source/{slug}--inputs.json'
        retain(p,ROOT/f'data/pressure/{slug}.json','datasets',source)
    # Whole-trace self-report selections include ratings and exact checkpoint spans.
    for p in (source/'experiment-results/emotion-self-report-activations').glob('*/selected-traces.json'):
        retain(p,ROOT/'runs'/p.relative_to(source/'experiment-results'),'traces',source)
        for name in ['cases.json','checkpoints.json','ratings.json','selection.json']:
            retain(p.with_name(name),ROOT/'runs'/p.with_name(name).relative_to(source/'experiment-results'),'traces',source)


def results(source):
    names = {'metrics.json','metrics.csv','summary.json','result-bundle.json','results.json',
             'baseline-results.json','selected-probes.csv','plotted-values.csv','source-data.csv',
             'simple-plotted-data.csv','summary.csv','plot-data.csv','averages.json',
             'verification.json','independent-verification.json'}
    for category in ['emotion-probes','emotion-rh-forecasting','emotion-self-report','emotion-steering']:
        base=source/'experiment-results'/category
        for p in base.rglob('*'):
            if p.is_file() and p.name in names and not any(k in p.parts for k in ['scores','inputs','activations']):
                retain(p,ROOT/'results'/p.relative_to(source/'experiment-results'),'results',source)
    for category in ['gemma27-emotion-denial-twoarms-20261026','gptoss120b-emotion-denial-lowreason-20261027','gemma27-session-label-control-20261026','gemma27-neutral100-whole-trace-audit-20261026']:
        for name in names:
            retain(source/'experiment-results'/category/name,ROOT/'results'/category/name,'results',source)
    output_sets=['emotion-probe-model-comparison-20261026','model-dataset-rh-grid-20261102',
        'steering-counterintuitive-four-panel-20261102','self-report-correlations-pooled-20261102',
        'model-dataset-emotion-spikes-20261101','task-average-emotion-traces-20261027']
    for category in output_sets:
        for p in (source/'output'/category).rglob('*'):
            if p.is_file() and (p.name in names or p.suffix in {'.png','.svg','.pdf'}):
                if not any(k in p.parts for k in ['replays','scores','_kit']):
                    retain(p,ROOT/'results/paper'/category/p.relative_to(source/'output'/category),'results',source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve()
    if not (source/'experiments/canonical').is_dir():
        parser.error('Expected a model-forensics research checkout')
    previous = json.loads((ROOT/'manifests/artifacts.json').read_text())['files'] if (ROOT/'manifests/artifacts.json').exists() else {}
    coverage=datasets(source);probes(source);traces(source);results(source)
    # Prune only unchanged exports made by this tool, never arbitrary run files.
    for name, item in previous.items():
        path=ROOT/name
        if name not in FILES and path.is_file() and sha(path)==item['sha256']:
            path.unlink()
    write(ROOT/'manifests/artifacts.json', {'schema_version':1,'files':FILES})
    write(ROOT/'manifests/export-summary.json',dict(files=len(FILES),
        counts=dict(Counter(f['category'] for f in FILES.values())),
        bytes_by_category={c:sum(f['bytes'] for f in FILES.values() if f['category']==c) for c in ['datasets','probes','traces','results']},
        catalog_coverage=dict(coverage), excludes=['activations','token scores','token caches','model weights']))
    print(json.dumps(json.loads((ROOT/'manifests/export-summary.json').read_text()),indent=2))


if __name__=='__main__':main()
