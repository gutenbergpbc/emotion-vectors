"""Materialized inventory loader. Historical source paths are provenance only."""
from collections import defaultdict
from pathlib import Path

from shared.io import digest, read, require, sha

EMOTIONS = ['hostile', 'frustrated', 'excited', 'appreciative', 'sad', 'afraid',
            'surprised', 'anxious', 'guilty', 'ashamed', 'desperate', 'calm']
RECIPES = ['neutral', 'other11_neutral', 'other11']
LABELS = EMOTIONS + ['neutral']
INPUTS = ['training_cases.json', 'eval_cases.json', 'split_manifest.json', 'protocol.json']
FROZEN = {
    'training_cases.json': 'fca9cc2407965efa7a925ee392bb3ef588c8847c9cb320aac37403b8b014174e',
    'eval_cases.json': 'a4175f24f31868a34281b233c9f35a740452ca0ba3f85f8aee9eb0ce9de08d04',
    'split_manifest.json': '5aaed7460b54dcf81088957d37d7adaaeb3b1117404a81656476fb1414188779',
    'protocol.json': 'b90e2e476fe3ad6f136d0be0f803360659579918e453da714f6ea192e120d68b',
}


def training_conditions(cases):
    """A story neutral is one shared observation, reused in each contrast mean."""
    story = all(c['render_mode'] == 'raw_story' for c in cases)
    return {key: [i for i, c in enumerate(cases)
                  if c['label'] == key[1] and (story or c['emotion'] == key[0])]
            for e in EMOTIONS for key in [(e, e), (e, 'neutral')]}


def validate_dataset(root, frozen=False, training_family='diversity'):
    root = Path(root)
    hashes = {name: sha(root / name) for name in INPUTS}
    if frozen:
        require(hashes == FROZEN, 'Frozen twelve-emotion inventory changed')
    training, evaluation, split = [read(root / name) for name in INPUTS[:3]]
    require(training and evaluation, 'Empty dataset')
    for cases in (training, evaluation):
        require(len({c['id'] for c in cases}) == len(cases), 'Duplicate case ID')
        for case in cases:
            if cases is training and training_family == 'story':
                require(case['render_mode'] == 'raw_story' and case['label'] in LABELS, 'Invalid story training case')
                require(isinstance(case['text'], str) and case['text'].strip(), 'Empty story')
                import hashlib
                require(hashlib.sha256(case['text'].encode()).hexdigest() == case['text_sha256'], 'Story text checksum mismatch')
                require(type(case['topic_index']) is int and case['topic_index'] in range(1, 21), 'Expected twenty story topics')
                require(case['group_id'] == f"story-topic-{case['topic_index']:02d}", 'Story source group must identify its topic')
                continue
            target = case['target']
            require(case['label'] in LABELS and case['emotion'] in EMOTIONS, 'Unknown label')
            require(target['stream'] in ('content', 'reasoning'), 'Unsupported target stream')
            require(target['message_index'] == len(case['messages']) - 1, 'Target must be in final supplied message')
            message = case['messages'][target['message_index']]
            require(message['role'] == 'assistant', 'Only assistant targets are supported')
            body = message.get(target['stream'], '')
            if case['render_mode'] == 'sales_with_reasoning':
                body = body.strip()
            a, b = target['char_start'], target['char_end']
            require(0 <= a < b <= len(body) and body[a:b] == target['text'], f"Invalid target: {case['id']}")
    groups = defaultdict(set)
    for case in training:
        require(type(case['fold']) is int and case['fold'] in range(5), 'Expected five saved folds')
        if training_family == 'diversity':
            require(case['label'] in (case['emotion'], 'neutral'), 'Training labels must be target or matched neutral')
        groups[case['group_id']].add(case['fold'])
    require(all(len(v) == 1 for v in groups.values()), 'Training source group crosses folds')
    if training_family == 'story':
        require(len(training) == 260 and len({c['text_sha256'] for c in training}) == 260, 'Expected 260 unique story/control texts')
        for label in LABELS:
            subset = [c for c in training if c['label'] == label]
            require(len(subset) == 20 and {c['topic_index'] for c in subset} == set(range(1, 21)), 'Each story label must cover twenty topics once')
            require(all(sum(c['fold'] == fold for c in subset) == 4 for fold in range(5)), 'Story folds must hold out four topics per label')
    for condition, indices in training_conditions(training).items():
        require({training[i]['fold'] for i in indices} == set(range(5)), f'Missing training condition/fold: {condition}')
    by_id = {c['id']: c for c in evaluation}
    # IDs are partition-local in the historical bundle. Detect actual replay
    # leakage below; an identical string ID is not evidence of shared content.
    train_texts = {digest({k: c[k] for k in ('messages', 'target', 'render_mode')}) for c in training if c['render_mode'] != 'raw_story'}
    require(all(digest({k: c[k] for k in ('messages', 'target', 'render_mode')}) not in train_texts for c in evaluation),
            'Exact training replay occurs in evaluation')
    story_texts = {c['text'].strip() for c in training if c['render_mode'] == 'raw_story'}
    require(all(c['target']['text'].strip() not in story_texts for c in evaluation), 'Exact training story occurs in evaluation')
    rows = split['rows']
    require(len({r['id'] for r in rows}) == len(rows), 'Duplicate split row')
    require(set(split['aliases']) == {r['id'] for r in rows}, 'Alias keys do not match retained IDs')
    used, partitions, exact = set(), defaultdict(set), defaultdict(set)
    for row in rows:
        require(row['split'] in ('calibration', 'heldout'), 'Invalid evaluation partition')
        require(row['id'] in by_id and row['id'] in split['aliases'][row['id']], 'Missing canonical replay')
        partitions[row['group_id']].add(row['split'])
        require(row['group_id'] not in groups, 'Training source group occurs in evaluation')
        for alias in split['aliases'][row['id']]:
            require(alias in by_id and alias not in used, 'Unknown/duplicate alias')
            used.add(alias)
            case = by_id[alias]
            require(case['label'] == row['label'], 'Alias label mismatch')
            key = digest({k: case[k] for k in ('messages', 'target', 'render_mode')})
            exact[key].add(row['split'])
    require(used == set(by_id), 'Incomplete alias coverage')
    require(all(len(v) == 1 for v in partitions.values()), 'Source conversation crosses evaluation partitions')
    require(all(len(v) == 1 for v in exact.values()), 'Exact replay crosses evaluation partitions')
    for part in ('calibration', 'heldout'):
        require({r['label'] for r in rows if r['split'] == part} == set(LABELS), f'Missing labels in {part}')
    if frozen:
        require((len(training), len(evaluation), len(rows)) == (2700, 2300, 2258), 'Frozen counts changed')
    # Extract canonical eval rows only: aliases retain their provenance in the manifest.
    return training, [by_id[r['id']] for r in rows], split, hashes
