#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))

from backbone.path_pointers import portable_project_path
from backbone.utils import load_yaml

MODELS = ('STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer')
TARGETS = ('speed','flow')


def newest_checkpoint(formal: Path, seed: int) -> Path | None:
    cands=[p for p in formal.glob(f'*_seed{seed}/best.pt') if p.is_file()]
    if not cands:
        return None
    return max(cands,key=lambda p:p.stat().st_mtime)


def main() -> None:
    ap=argparse.ArgumentParser(description='Rebuild portable LATEST_RUN/LATEST_CHECKPOINT pointers for existing Backbone Formal runs.')
    ap.add_argument('--models',nargs='*',choices=MODELS,default=list(MODELS))
    ap.add_argument('--targets',nargs='*',choices=TARGETS,default=list(TARGETS))
    ap.add_argument('--seed',type=int,default=10)
    args=ap.parse_args()

    project_root=Path(os.environ.get('SPEED_LLM_RAG_ROOT',REPO)).expanduser().resolve()
    for target in args.targets:
        os.environ['TRAFFIDENT_TARGET']=target
        cfg=load_yaml(ROOT/'configs/backbone.yaml')
        for model in args.models:
            formal=Path(cfg['project']['exp_root'])/model/cfg['experiment']['subgraph']/'formal'
            ck=newest_checkpoint(formal,args.seed)
            if ck is None:
                print(f'[SKIP] {target} / {model}: best.pt not found under {formal}',flush=True)
                continue
            run=ck.parent
            run_ptr=portable_project_path(run,project_root)
            ck_ptr=portable_project_path(ck,project_root)
            formal.mkdir(parents=True,exist_ok=True)
            (formal/'LATEST_RUN.txt').write_text(run_ptr+'\n',encoding='utf-8')
            (formal/'LATEST_CHECKPOINT.txt').write_text(ck_ptr+'\n',encoding='utf-8')
            print(f'[OK] {target} / {model}',flush=True)
            print(f'     RUN  = {run_ptr}',flush=True)
            print(f'     CKPT = {ck_ptr}',flush=True)


if __name__=='__main__':
    main()
