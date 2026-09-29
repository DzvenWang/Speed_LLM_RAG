#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
ROOT=next((p for p in Path(__file__).resolve().parents if p.name=='Baseline'),Path(__file__).resolve().parents[1])
sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml
from baseline.backbones.frozen_backbone import inspect_frozen_backbone

def main():
    cfg=load_yaml(ROOT/'configs/baseline.yaml')
    obj=inspect_frozen_backbone(cfg)
    print('PASS Backbone-owned FROZEN_BACKBONE.json schema',obj['schema_version'])
    print('PASS selected_model',obj['selected_model'])
    print('PASS Backbone project_version',obj['project_version'])
    print('PASS reference_seed',obj['reference_seed'])
    print('PASS reference_checkpoint',obj['reference_checkpoint'])
    print('PASS reference_checkpoint_sha256',obj['reference_checkpoint_sha256'])
    print('FROZEN BACKBONE VALIDATION PASS')
if __name__=='__main__': main()
