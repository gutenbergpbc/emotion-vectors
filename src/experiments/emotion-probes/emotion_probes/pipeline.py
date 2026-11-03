from pathlib import Path
import shutil

from shared.io import atomic, read, require, sha
from shared.progress import Progress, STAGES, lock
from .config import CANONICAL, PROJECT, source_hashes
from .data import INPUTS


def prepare(root, spec, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'config.resolved.json'
    if path.exists():
        require(resume, 'Run already exists; use --resume or a new run_id')
        require(read(path) == spec, 'Configuration or code changed; choose a new run_id')
    else:
        atomic(path, spec)
    (root / 'inputs').mkdir(exist_ok=True)
    for name in INPUTS:
        target = root / 'inputs' / name
        if target.exists():
            require(sha(target) == spec['input_hashes'][name], f'Changed frozen input {name}')
        else:
            shutil.copyfile(Path(spec['dataset']) / name, target)
    install_dashboard(root)


def install_dashboard(root):
    # The shared viewer consumes the existing dashboard progress contract.
    for source, dest in [('dashboard.html', 'index.html'), ('dashboard.css', 'dashboard.css'), ('dashboard.js', 'dashboard.js')]:
        shutil.copyfile(PROJECT / 'dashboards' / source, root / dest)


def execute(root, through='verify', smoke_only=False):
    from .extraction import tokenize, extract
    from .fitting import fit
    from .evaluation import calibrate, evaluate
    from .reporting import report, verify
    root = Path(root)
    spec = read(root / 'config.resolved.json')
    require(spec['source_hashes'] == source_hashes(), 'Scientific code changed since preparation; create a new run')
    with lock(root), Progress(root, f"{'SYNTHETIC DEMO · ' if spec['backend'] == 'synthetic' else ''}{spec['run_id']}") as progress:
        progress.stage('prepare', 4, 4, 'input files', 'Frozen input files verified')
        for name, checksum in spec['input_hashes'].items():
            require(sha(root / 'inputs' / name) == checksum, 'Prepared input changed')
        progress.complete_stage('prepare')
        tasks = {'tokenize': tokenize, 'extract': extract, 'fit': fit, 'calibrate': calibrate,
                 'evaluate': evaluate, 'report': report, 'verify': verify}
        through = 'extract' if smoke_only else through
        for name in STAGES[1:STAGES.index(through)+1]:
            progress.stage(name, message=f'Starting {name}')
            if name == 'extract':
                complete = extract(root, spec, progress, smoke_only)
                if not complete:
                    next(s for s in progress.data['stages'] if s['id'] == name)['state'] = 'pending'
                    progress.finish(False)
                    return
            else:
                tasks[name](root, spec, progress)
            progress.complete_stage(name)
        progress.finish(through == 'verify')
