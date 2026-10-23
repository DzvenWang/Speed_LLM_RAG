#!/usr/bin/env python3
"""Submit a READ-ONLY validation innovation audit for an EXISTING v6 run.

Reads the small run/manifest.json; does NOT load the big checkpoint on the
login node. Only the GPU worker loads best.pt/OOF Memory and runs validation.
"""
import _bootstrap
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from fare.config import root,project_path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target',required=True,choices=['speed','flow'])
    p.add_argument('--run-dir',required=True)
    p.add_argument('--config',default='FARE/configs/formal.yaml')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--check-only',action='store_true')
    args=p.parse_args()
    d=project_path(args.run_dir,must_exist=True)
    manifest_path=d/'manifest.json';ck=d/'best.pt'
    if not manifest_path.is_file() or not ck.is_file():
        raise FileNotFoundError('Expected existing FARE run manifest.json and best.pt')
    manifest=json.loads(manifest_path.read_text(encoding='utf8'))
    md=manifest.get('metadata',{})
    if manifest.get('version')!='6.0.0' or md.get('target')!=args.target:
        raise ValueError('Run is not a v6.0.0 checkpoint for requested target')
    if not md.get('backbone_checkpoint') or not md.get('memory_path'):
        raise ValueError('Missing run provenance for frozen backbone or paired OOF Memory')
    if not (Path(md['memory_path'])/'manifest.json').is_file():
        raise FileNotFoundError('OOF memory manifest not found')
    config=project_path(args.config,must_exist=True)
    if not Path(md['backbone_checkpoint']).is_file():
        raise FileNotFoundError('Original frozen STLAformer checkpoint not found')
    out=d/('innovation_audit_v601_smoke' if args.smoke else 'innovation_audit_v601')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Audit output already exists. Choose a new version/path: {out}')
    env={
      'SPEED_LLM_RAG_ROOT':str(root()),
      'FARE_PYTHON':str(Path(sys.executable).resolve()),
      'TARGET':args.target,
      'FARE_AUDIT_CHECKPOINT':str(ck),
      'BACKBONE_CHECKPOINT':str(Path(md['backbone_checkpoint']).resolve()),
      'FARE_MEMORY':str(Path(md['memory_path']).resolve()),
      'FARE_AUDIT_CONFIG':str(config),
      'FARE_AUDIT_OUTPUT':str(out),
      'FARE_AUDIT_SMOKE':'1' if args.smoke else '0'
    }
    if any(',' in v or '\n' in v for v in env.values()):
        raise ValueError('PBS -v cannot safely contain commas/newlines')
    print('FARE6_AUDIT_READ_ONLY_PRECHECK_PASS '+json.dumps({'run':str(d),'output':str(out),
          'target':args.target,'checkpoint':str(ck),'oof_memory':env['FARE_MEMORY'],
          'python':env['FARE_PYTHON'],'smoke':args.smoke}),flush=True)
    if args.check_only:return
    script=root()/'FARE/pbs/audit_checkpoint.pbs'
    if not script.is_file():raise FileNotFoundError(script)
    job=subprocess.check_output(['qsub','-v',','.join(f'{k}={v}' for k,v in env.items()),str(script)],text=True).strip()
    print('FARE6_AUDIT_JOB='+job+'\nFARE6_AUDIT_OUTPUT='+str(out),flush=True)

if __name__=='__main__':main()
