#!/usr/bin/env python3
"""Generate the agent-facing script reference from source docstrings and flags.

This statically parses files; it never imports workers or launches experiments.
"""
import ast
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    lines=['# Script reference', '',
        'Run scripts from the repository root with `uv run python PATH`. Each experiment',
        'README identifies the recommended entrypoints. The Gemma walkthrough selects',
        'the files needed for that model; historical scheduler and recovery helpers',
        'retain explicit constants and should be inspected before use.', '',
        'This reference is generated without importing code. Input/output constants',
        'are source expressions, not assertions that their files already exist.',
        'Paths into local `runs/` may require a prior independent experiment. GPU',
        'and API scripts do not necessarily implement a no-cost `--help`/dry-run;',
        'use the documented primary entrypoints for workload previews.', '']
    for p in sorted((ROOT/'src/experiments').rglob('*.py')):
        if '_kit' in p.parts or '__pycache__' in p.parts:
            continue
        tree=ast.parse(p.read_text())
        relative=str(p.relative_to(ROOT))
        description=ast.get_docstring(tree) or 'Supporting implementation; inspect functions and its experiment README before direct use.'
        flags=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':
                flags.extend(a.value for a in node.args if isinstance(a,ast.Constant) and isinstance(a.value,str))
        constants=[]
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id in {'ROOT','PROJECT','CANONICAL','OUT','OUTPUT','BASE','SOURCE','SOURCES','LABELS','PROBES','REPLAY','SELECTION','DEST','LOCAL'} for n in node.targets):
                expression=ast.unparse(node)
                if len(expression)<700:constants.append(expression)
        lines.extend([f'## `{relative}`', '', description.strip(), ''])
        if flags:lines.extend(['Arguments: '+', '.join('`'+s+'`' for s in dict.fromkeys(flags))+'.',''])
        functions=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))]
        if functions:lines.extend(['Main interfaces: '+', '.join('`'+s+'`' for s in functions)+'.',''])
        if constants:lines.extend(['Input/output settings:', '', '```python', *constants, '```', ''])
    (ROOT/'docs/scripts.md').write_text('\n'.join(lines).rstrip()+'\n')
    print('Wrote docs/scripts.md')


if __name__=='__main__':main()
