"""Generate, audit, and freeze Gemma-authored topic-matched story training data.

Reuse the existing Gemma self-story prompts, lexical checks and semantic rubric.
Each accepted item is checkpointed independently; no evaluation text reaches
the generator or reviewer. API provider revisions are not claimed to be pinned.
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

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import importlib.util
from pathlib import Path
import re
import shutil
import sys

import numpy as np

CANONICAL = Path(__file__).resolve().parents[1]
PROJECT = CANONICAL.parents[1]
sys.path[:0] = [str(CANONICAL), str(CANONICAL / 'emotion-probes')]
from shared.io import atomic, digest, read, require, sha
from shared.progress import Progress, lock
from emotion_probes.data import EMOTIONS, LABELS, validate_dataset
from emotion_probes.pipeline import install_dashboard

MODEL = 'google/gemma-3-4b-it'
FOLD_SEED = 20261014


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def accepted(decision, kind):
    if kind == 'emotion':
        for field in ('fit', 'topic_fit'):
            require(type(decision.get(field)) is int and 1 <= decision[field] <= 5, 'Invalid semantic audit rating')
        require(type(decision.get('explicit_wording')) is bool, 'Missing explicit-wording audit')
        return decision['fit'] >= 3 and decision['topic_fit'] >= 3 and not decision['explicit_wording']
    require(type(decision.get('neutrality')) is int and 1 <= decision['neutrality'] <= 5, 'Invalid neutrality audit')
    require(type(decision.get('role_format_ok')) is bool, 'Missing role-format audit')
    return decision['neutrality'] == 5 and decision['role_format_ok']


def collect_item(item, generator, reviewer, folder, generation_identity):
    path = folder / 'accepted' / f"{item['id']}.json"
    if path.exists():
        record = read(path)
        require(record['generation_identity'] == generation_identity, 'Generation configuration changed')
        require(accepted(record['semantic_audit_final'], record['kind']), 'Cached item no longer passes audit')
        require(not generator.qa(record, record['text'], {'finish_reason': record.get('revision_finish_reason', record['finish_reason'])}), 'Cached lexical audit failed')
        return record
    attempt_path = folder / 'attempts' / f"{item['id']}.jsonl"
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    record = generator.run_one(item, attempt_path)
    require(not record.get('qa_flags') and not record.get('error'), f"Generation failed: {item['id']}: {record.get('error')}")
    record['generation_identity'] = generation_identity
    lineage = []
    for attempt in range(4):
        audit = reviewer.audit_batch(f"{item['id']}-audit-{attempt}", [record], item['emotion'])
        decision = audit['items'][0]
        lineage.append({'audit': audit, 'text_sha256': hashlib.sha256(record['text'].encode()).hexdigest()})
        atomic(folder / 'reviews' / f"{item['id']}.json", lineage)
        if accepted(decision, item['kind']):
            record['semantic_audit_final'] = decision
            record['review_count'] = attempt + 1
            atomic(path, record)
            return record
        require(attempt < 3, f"Semantic audit unresolved: {item['id']}: {decision}")
        issue = decision.get('issue') or str(decision)
        original = record['text']
        for repair_attempt in range(3):
            revised = reviewer.revise(record, issue)
            flags = generator.qa(item, revised['text'], {'finish_reason': revised['revision_finish_reason']})
            lineage.append({'prior_text': original, 'revision': revised, 'lexical_flags': flags})
            atomic(folder / 'reviews' / f"{item['id']}.json", lineage)
            if not flags:
                record.update(revised)
                break
            issue = 'Repair failed these structural/lexical checks: ' + ', '.join(flags)
        else:
            raise ValueError(f"Repair unresolved: {item['id']}: {flags}")
    raise AssertionError('Unreachable')


def materialize(records, dataset, source, generation_spec, folder):
    topics = sorted({r['topic_index'] + 1 for r in records})
    shuffled = np.random.default_rng(FOLD_SEED).permutation(topics)
    folds = {int(topic): fold for fold, group in enumerate(np.array_split(shuffled, 5)) for topic in group}
    cases = []
    for record in records:
        text = record['text'].strip()
        label = record['emotion'].lower() if record['kind'] == 'emotion' else 'neutral'
        if record['kind'] == 'emotion':
            text = re.sub(r'^\s*\[story\s+1\]\s*', '', text, count=1, flags=re.I).strip()
        else:
            text = re.sub(r'(?m)^Person:\s*', 'Human: ', text)
            text = re.sub(r'(?m)^AI:\s*', 'Assistant: ', text)
        topic = record['topic_index'] + 1
        cases.append({'id': f'gemma4b-story-{label}-{topic:02d}', 'label': label,
                      'partition': 'training', 'render_mode': 'raw_story', 'topic_index': topic,
                      'topic': record['topic'], 'group_id': f'story-topic-{topic:02d}', 'fold': folds[topic],
                      'text': text, 'text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                      'generation_id': record['id']})
    cases.sort(key=lambda c: (LABELS.index(c['label']), c['topic_index']))
    dataset.mkdir(parents=True, exist_ok=True)
    atomic(dataset / 'training_cases.json', cases)
    for name in ('eval_cases.json', 'split_manifest.json'):
        shutil.copyfile(source / name, dataset / name)
    atomic(folder / 'topic_folds.json', {'seed': FOLD_SEED, 'topic_to_fold': folds, 'fold_count': 5})
    atomic(folder / 'qa_report.json', {'status': 'passed', 'semantic_review_items': len(records),
        'unresolved': 0, 'counts': {label: sum(c['label'] == label for c in cases) for label in LABELS},
        'review_model': MODEL, 'review_method': 'same-model semantic audit with fixed pre-probe repair criteria',
        'provenance': 'API model/provider revision is not independently verified against the pinned extraction checkpoint'})
    atomic(dataset / 'protocol.json', {
        'schema_version': 1, 'training_family': 'story', 'generation': generation_spec,
        'training_cases_sha256': sha(dataset / 'training_cases.json'),
        'accepted_record_hashes': {r['id']: sha(folder / 'accepted' / f"{r['id']}.json") for r in records},
        'qa_report_sha256': sha(folder / 'qa_report.json'), 'fold_seed': FOLD_SEED,
        'topic_to_fold': folds, 'neutral_pool': '20 shared neutral dialogues, each stored once',
        'training_pooling': 'all raw-text tokens including tokenizer special tokens; no chat template',
        'evaluation_source': str(source),
        'evaluation_hashes': {name: sha(source / name) for name in ('eval_cases.json', 'split_manifest.json')},
        'limitations': ['same-model semantic review, not human-certified labels',
                       'story versus neutral-dialogue format confound', '20 topics, four validation topics per fold',
                       'reused retrospective evaluation benchmark'],
    })
    validate_dataset(dataset, training_family='story')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    folder = root / 'generation'
    folder.mkdir(parents=True, exist_ok=True)
    legacy = PROJECT / 'experiments/gemma_self_story_topicmix'
    generator = load_module('story_generator', legacy / 'generate_openrouter.py')
    reviewer = load_module('story_reviewer', legacy / 'review_openrouter.py')
    generator.MODEL = reviewer.MODEL = MODEL
    reviewer.generator = generator
    jobs = generator.make_jobs()
    require(len(jobs) == 260 and [e.lower() for e in generator.qwen.EMOTIONS] == EMOTIONS, 'Story inventory changed')
    source = PROJECT / 'probe-harness/unified-twelve-model-replication-v1/inputs'
    dataset = root / 'story-inputs'
    spec = {'model': MODEL, 'api_url': generator.URL, 'topics': generator.qwen.TOPICS,
            'emotions': EMOTIONS, 'count': 260, 'source_hashes': {
                'collector': sha(Path(__file__)), 'generator': sha(legacy / 'generate_openrouter.py'),
                'reviewer': sha(legacy / 'review_openrouter.py'), 'prompts': sha(generator.SOURCE)},
            'prompt_adjustments': 'existing Gemma topicmix one-item prompts, perspective mix and neutral task rotation'}
    spec['identity'] = digest(spec)
    cfg_path = folder / 'config.json'
    if cfg_path.exists():
        require(read(cfg_path) == spec, 'Generation source/config changed; use a new run')
    else:
        atomic(cfg_path, spec)
    install_dashboard(root)
    with lock(root), Progress(root, root.name) as progress:
        completed = len(list((folder / 'accepted').glob('*.json')))
        progress.stage('prepare', completed, 260, 'accepted stories/controls', 'Generating and auditing Gemma 3 4B story training data')
        records, errors = [], []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(collect_item, job, generator, reviewer, folder, spec['identity']): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    records.append(future.result())
                    progress.stage('prepare', len(list((folder / 'accepted').glob('*.json'))), 260, 'accepted stories/controls', f"Generated and reviewed {job['id']}")
                    print(f"Accepted {job['id']}", flush=True)
                except Exception as exc:
                    errors.append({'id': job['id'], 'error': str(exc)})
                    atomic(folder / 'errors' / f"{job['id']}.json", errors[-1])
                    print(f"Failed {job['id']}: {exc}", flush=True)
        require(not errors, f'{len(errors)} generation/review items failed; rerun to resume accepted items')
        materialize(records, dataset, source, spec, folder)
        progress.complete_stage('prepare')
        progress.finish(False)
    print(f'Frozen story dataset: {dataset}', flush=True)


if __name__ == '__main__':
    main()
