#!/usr/bin/env python3
"""Run frozen-backbone evaluation for all configured severities/seeds with one live log."""
from pathlib import Path
import argparse
import datetime
import os
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(os.environ.get('SPEED_LLM_RAG_ROOT', ROOT.parent)).expanduser().resolve()
sys.path.insert(0, str(ROOT))
from backbone.utils import load_yaml

_TS_RE = re.compile(r'^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]')


def _stamp(line: str) -> str:
    line = line.rstrip('\r\n')
    if _TS_RE.match(line):
        return line
    return f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {line}"


def _emit(fh, line: str) -> None:
    x = _stamp(line)
    print(x, flush=True)
    fh.write(x + '\n')
    fh.flush()


def _run(cmd, fh, env):
    _emit(fh, 'RUN ' + subprocess.list2cmdline(cmd))
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace', bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        _emit(fh, line)
    rc = proc.wait()
    _emit(fh, f'COMMAND_EXIT_STATUS={rc}')
    if rc != 0:
        raise SystemExit(rc)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reference-only',action='store_true')
    ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml'))
    ap.add_argument('--run-stamp',default=None)
    a=ap.parse_args()
    cfg=load_yaml(a.config)
    seeds=[int(cfg['experiment']['reference_seed'])] if a.reference_only else [int(x) for x in cfg['experiment']['formal_seeds']]
    stamp=a.run_stamp or datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag='reference' if a.reference_only else 'allseeds'
    log_root=Path(cfg['project']['log_root']); log_root.mkdir(parents=True, exist_ok=True)
    log_path=log_root/f'PatchSTG.evaluation.{tag}.{stamp}.live.log'
    env=os.environ.copy(); env['PYTHONUNBUFFERED']='1'; env.setdefault('SPEED_LLM_RAG_ROOT', str(REPO_ROOT))
    with log_path.open('a',encoding='utf-8',buffering=1) as fh:
        _emit(fh,f'LOG_FILE={log_path}')
        _emit(fh,f'START frozen-backbone evaluation seeds={seeds} severities={cfg["experiment"]["evaluation_severities"]}')
        _emit(fh,'EVALUATION_ACCESS_POLICY=unlocked')
        for sev in [int(x) for x in cfg['experiment']['evaluation_severities']]:
            for seed in seeds:
                _run([sys.executable,str(ROOT/'scripts/run_evaluation.py'),'--severity',str(sev),'--seed',str(seed),'--config',a.config,'--run-stamp',stamp,'--quiet-summary'],fh,env)
        _run([sys.executable,str(ROOT/'scripts/summarize_evaluation.py'),'--config',a.config],fh,env)
        _emit(fh,'FROZEN BACKBONE EVALUATION PASS')
        _emit(fh,'SCRIPT_EXIT_STATUS=0')

if __name__=='__main__': main()
