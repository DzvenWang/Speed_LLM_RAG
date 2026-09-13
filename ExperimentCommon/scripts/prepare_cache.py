#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not ROOT.exists(): ROOT=Path(__file__).resolve().parents[1]
for p in [ROOT, ROOT.parent/'Protocol', ROOT.parent/'TraffiDent_SupportBench']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from experiment_common.cache import prepare_cache
from experiment_common.utils import load_yaml
p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['development','formal','evaluation'],required=True); p.add_argument('--force',action='store_true'); p.add_argument('--config',default=str(ROOT/'configs/experiment_contract.yaml')); a=p.parse_args()
cfg=load_yaml(a.config); prepare_cache(cfg,a.stage,force=a.force)
