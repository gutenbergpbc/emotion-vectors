import pytest

from shared.io import atomic, read
from emotion_probes.pipeline import prepare
from test_pipeline import fixture_config
from resume_from_extract import execute


@pytest.mark.parametrize('complete', [True, False])
def test_cached_resume_skips_tokenization_and_requires_completed_manifests(tmp_path, monkeypatch, complete):
    spec = fixture_config(tmp_path)
    root = tmp_path / 'run'
    prepare(root, spec)
    atomic(root / 'tokenization.json', {'identity': spec['identity'],
           'cases': spec['counts']['training'] + spec['counts']['eval']})
    for part in ('training', 'eval'):
        count = spec['counts'][part] - (1 if part == 'eval' and not complete else 0)
        atomic(root / 'tokens' / part / 'manifest.json',
               {'identity': spec['identity'], 'records': [{} for _ in range(count)]})
    def forbidden(*args, **kwargs):
        pytest.fail('Cached resume must not invoke tokenization')
    monkeypatch.setattr('emotion_probes.extraction.tokenize', forbidden)
    calls = []
    def extract(*args, **kwargs):
        calls.append('extract')
        return True
    monkeypatch.setattr('emotion_probes.extraction.extract', extract)
    if not complete:
        with pytest.raises(ValueError, match='Incomplete token manifest'):
            execute(root, through='extract')
        assert calls == []
        return
    execute(root, through='extract')
    assert calls == ['extract']
    progress = read(root / 'progress.json')
    stage = next(s for s in progress['stages'] if s['id'] == 'tokenize')
    assert stage['state'] == 'complete'
    assert 'scan skipped' in stage['message']
    assert len(list((root / 'logs').glob('cached-token-resume-*.json'))) == 1


def test_explicit_policy_skips_tolerance_smoke_without_claiming_pass(tmp_path):
    from emotion_probes import extraction
    from resume_from_extract import extract_with_policy
    from types import SimpleNamespace
    spec = {'identity': 'frozen-run'}
    atomic(tmp_path / 'execution-policy.json', {
        'identity': spec['identity'], 'capture_smoke': 'skip_tolerance_checks',
        'reason': 'User requested extraction without tolerance checks'})
    original = extraction.smoke
    progress = SimpleNamespace(stage=lambda *args, **kwargs: None)
    def extract(root, spec, progress, smoke_only):
        extraction.smoke(root, spec, object(), progress, {})
        return True
    assert extract_with_policy(tmp_path, spec, progress, False, extract)
    assert extraction.smoke is original
    evidence = read(tmp_path / 'smoke.json')
    assert evidence['status'] == 'skipped' and evidence['passed'] is None
    assert read(tmp_path / 'probes/execution-policy.json') == evidence
    assert 'finite activation values' in evidence['checks_retained']
    with pytest.raises(ValueError, match='another run'):
        extract_with_policy(tmp_path, {'identity': 'different'}, progress, False, extract)
    assert extraction.smoke is original
