"""Exercise the combined frozen-emotion model against canonical synthetic data."""
import importlib.util
from pathlib import Path
from test_rh_forecasting import source
from emotion_rh_forecasting.pipeline import load_config, resolve, execute
from shared.io import read

def test_all_emotion_combiner_preserves_pairs_and_baseline(source, tmp_path):
    config=load_config(source)
    config['horizons']=list(range(11))
    spec,bundle,traces,jobs=resolve(config)
    root=tmp_path/'forecast'
    execute(root,spec,bundle,traces,jobs)
    path=(Path(__file__).resolve().parents[1] / "src/experiments")/'emotion-rh-forecasting/combined_emotions.py'
    module_spec=importlib.util.spec_from_file_location('combined_emotions',path)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    module.main(root)
    result=read(root/'combined-all12/result-bundle.json')
    assert len(result['estimates'])==99
    assert all(r['accuracy']==1 for r in result['estimates'] if r['model']=='all_12')
    assert all(r['accuracy']==.5 for r in result['estimates'] if r['model']=='always_not_hacking')
    assert read(root/'combined-all12/verification.json')['passed']
