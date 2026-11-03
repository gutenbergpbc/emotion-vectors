"""Run GPT-OSS through the shared one-run coordinator and local judge."""
from . import run_once
from pathlib import Path

run_once.MODAL_RUNNER = Path(__file__).resolve().with_name('modal_runner_gpt_oss_once.py')
run_once.main()
