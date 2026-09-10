#!/usr/bin/env python3
from pathlib import Path
import json,sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
cfg=load_yaml(ROOT/'configs/baseline.yaml'); p=Path(cfg['project']['frozen_backbone_json'])
if not p.exists():
    print('WAIT: frozen numerical backbone is not available:',p); raise SystemExit(3)
obj=json.loads(p.read_text()); print('PASS frozen backbone',json.dumps(obj,indent=2,sort_keys=True))
