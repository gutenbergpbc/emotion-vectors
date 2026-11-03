"""Check the completed report in a real browser and inspect PDF coverage."""

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
import json
from pathlib import Path

import pymupdf
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent


def check(root):
    viewer = root / 'viewer'
    data = json.loads((viewer / 'plot-data.json').read_text())
    assert len(data['models']) == 2 and len(data['emotions']) == 12
    with sync_playwright() as p:
        executable = next((Path.home() / '.cache/ms-playwright').glob('chromium_headless_shell-*/chrome-headless-shell-linux64/chrome-headless-shell'))
        browser = p.chromium.launch(headless=True, executable_path=str(executable))
        page = browser.new_page(viewport={'width': 1440, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto((viewer / 'index.html').as_uri())
        assert page.locator('section.emotion').count() == 12
        assert page.locator('.chart svg').count() == 24
        for mode, dots in [('raw', 288), ('change', 288), ('paired', 192)]:
            page.select_option('#mode', mode)
            assert page.locator('circle.point').count() == dots
            assert page.locator('polygon').count() == {'raw': 72, 'change': 0, 'paired': 48}[mode]
            assert not page.locator('svg').evaluate_all('(nodes)=>nodes.some(n=>/NaN|Infinity|undefined/.test(n.outerHTML))')
        page.select_option('#mode', 'raw')
        for emotion in ('desperate', 'frustrated', 'appreciative'):
            page.locator('#' + emotion).screenshot(path=str(viewer / (emotion + '-html.png')))
        page.select_option('#jump', 'appreciative')
        page.wait_for_timeout(600)
        page.set_viewport_size({'width': 390, 'height': 844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        assert not errors, errors
        browser.close()
    document = pymupdf.open(viewer / 'all-emotions.pdf')
    assert len(document) == 13
    assert 'Emotions such as frustration' in document[0].get_text()
    assert 'Emotions do not affect' in document[0].get_text()
    for i, emotion in enumerate(data['emotions']):
        assert data['titles'][emotion] in document[i+1].get_text()
    result = dict(emotions=12, model_panels=24, views_tested=3, pdf_pages=13,
                  exact_prompts_in_pdf=True, javascript_errors=errors,
                  mobile_no_horizontal_overflow=True)
    (root / 'presentation-verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    check(parser.parse_args().root.resolve())
