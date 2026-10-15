#!/usr/bin/env python3
"""Submit exactly one explicit checkpoint/Run evaluation, on one GPU."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
from catalog import BB_MODELS, BL_METHODS, NATIVE, REFERENCES, STATUS_RE


def abs_path(value: str, root: Path) -> Path:
    p = Path(value).expanduser()
    return (p if p.is_absolute() else root / p).resolve()


def main():
    p = argparse.ArgumentParser(description='One explicit Evaluate-only job, no training.')
    p.add_argument('--project', required=True, choices=['backbone', 'baseline'])
    p.add_argument('--target', required=True, choices=['speed', 'flow'])
    p.add_argument('--model', help='Backbone model: STLAformer, PatchSTG, etc')
    p.add_argument('--method', help='Baseline method: e.g. RAFT-STLAformer')
    p.add_argument('--backbone', help='Baseline reference model, when applicable')
    p.add_argument('--checkpoint', help='Path to best.pt (both projects)')
    p.add_argument('--run-dir', help='Baseline formal run directory, alternative to --checkpoint')
    p.add_argument('--mode', choices=['incomplete','complete'], default=None)
    p.add_argument('--seed', type=int, default=int(os.environ.get('SEED', '10')))
    p.add_argument('--root', default=os.environ.get('SPEED_LLM_RAG_ROOT'))
    p.add_argument('--submit', action='store_true', help='Actually submit to qsub; default previews only')
    a = p.parse_args()
    root = Path(a.root).resolve() if a.root else Path(__file__).resolve().parents[2]
    if a.project == 'backbone':
        if not a.model or a.model not in BB_MODELS: p.error(f'--model required and must be one of {BB_MODELS}')
        if a.method or a.backbone or a.run_dir: p.error('Backbone accepts --model and --checkpoint only')
        if not a.checkpoint: p.error('--checkpoint is required')
        ckpt = abs_path(a.checkpoint, root)
        if not ckpt.is_file(): p.error(f'checkpoint missing: {ckpt}')
        mode = a.mode or ('complete' if 'formal_complete' in ckpt.parts else 'incomplete')
        if 'formal_complete' in ckpt.parts and mode != 'complete': p.error('formal_complete checkpoint requires --mode complete')
        if 'formal' in ckpt.parts and mode != 'incomplete': p.error('formal checkpoint requires --mode incomplete')
        if ckpt.parts[-1] != 'best.pt': p.error('checkpoint should be best.pt')
        if a.model not in ckpt.parts: print('WARNING: checkpoint path has no matching model directory; evaluator verifies metadata', file=sys.stderr)
        if a.target not in ckpt.parts: print('WARNING: checkpoint path has no matching target directory', file=sys.stderr)
        name = f'BB_One_{a.target[0]}_{a.model[:4]}_{mode[0]}'
        envvars = dict(MODEL=a.model,TARGET=a.target,MODE=mode,SEED=str(a.seed),CKPT=str(ckpt))
        pbs = root/'EvalOnly/pbs/backbone_eval_only.pbs'
    else:
        if a.model: p.error('Baseline uses --method (not --model)')
        if not a.method or a.method not in BL_METHODS: p.error(f'--method required and must be one of {BL_METHODS}')
        if bool(a.checkpoint) == bool(a.run_dir): p.error('Provide exactly one of --checkpoint or --run-dir')
        run_dir = abs_path(a.run_dir, root) if a.run_dir else abs_path(a.checkpoint, root).parent
        if a.checkpoint and abs_path(a.checkpoint, root).name != 'best.pt': p.error('Baseline checkpoint must be best.pt')
        if not (run_dir/'best.pt').is_file(): p.error(f'best.pt missing under {run_dir}')
        summary = run_dir/'summary.json'
        if not summary.is_file() or not STATUS_RE.search(summary.read_text(errors='replace')):
            p.error(f'completed summary.json missing under {run_dir}')
        mode = a.mode or ('complete' if 'complete' in run_dir.parts else 'incomplete')
        if a.method in NATIVE:
            if a.backbone or mode != 'incomplete': p.error('Native Baseline must use no --backbone and incomplete mode')
        else:
            if a.backbone not in REFERENCES[a.method]: p.error(f'--backbone required; valid values: {REFERENCES[a.method]}')
        if a.target not in run_dir.parts: print('WARNING: run path lacks matching target directory', file=sys.stderr)
        if a.method not in run_dir.parts: print('WARNING: run path lacks matching method directory', file=sys.stderr)
        name = f'BL_One_{a.target[0]}_{a.method[:8].replace("-", "")}'
        envvars = dict(METHOD=a.method,TARGET=a.target,SEED=str(a.seed),MODE=mode,FORMAL_RUN_DIR=str(run_dir))
        if a.backbone: envvars['BACKBONE']=a.backbone
        pbs = root/'EvalOnly/pbs/baseline_eval_only.pbs'
    if not pbs.is_file(): p.error(f'PBS file missing: {pbs}')
    envvars['SPEED_LLM_RAG_ROOT'] = str(root)
    envroot = os.environ.get('TFP_ENV_ROOT') or os.environ.get('CONDA_PREFIX')
    if envroot:
        envvars['TFP_ENV_ROOT'] = str(Path(envroot).resolve())
    vars_str = ','.join(f'{k}={v}' for k,v in envvars.items())
    if ',' in str(root) or any(',' in str(v) or '\n' in str(v) for v in envvars.values()):
        p.error('qsub -v does not support commas/newlines in checkpoint paths')
    cmd = ['qsub','-N',name,'-v',vars_str,str(pbs)]
    print(f'PROJECT={a.project}  TARGET={a.target}  MODE={mode}  GPU=1')
    print('CHECKPOINT='+str(ckpt if a.project=='backbone' else run_dir/'best.pt'))
    print('QSUB_ARGUMENTS='+repr(cmd))
    if not a.submit:
        print('DRY RUN: add --submit to queue this single evaluation')
        return
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if proc.returncode:
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(proc.returncode)
    print('[SUBMITTED] '+proc.stdout.strip())

if __name__ == '__main__': main()
