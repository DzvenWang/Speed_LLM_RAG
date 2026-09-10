#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys
import pandas as pd
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone'); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'Protocol')); sys.dont_write_bytecode=True
from backbone.utils import load_yaml
from shared_protocol.consumer import load_frozen_stage
from shared_protocol.utils import canonical_sha256
p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['screening','formal','evaluation'],required=True); a=p.parse_args()
cfg=load_yaml(ROOT/'configs/backbone_selection.yaml'); cache=Path(cfg['project']['exp_root'])/'_cache'/cfg['experiment']['subgraph']/a.stage
mf=cache/'cache_manifest.json'; ready=cache/'READY.json'
if not mf.exists() or not ready.exists(): raise SystemExit(f'FAIL missing cache manifest/READY in {cache}')
j=json.loads(mf.read_text()); fp=j.get('cache_fingerprint'); tmp=dict(j); tmp.pop('cache_fingerprint',None)
if canonical_sha256(tmp)!=fp: raise SystemExit('FAIL Backbone cache fingerprint mismatch')
pstage=cfg['protocol']['stage_map'][a.stage]; pm,pviews=load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],cfg['protocol']['name'],pstage)
if j.get('protocol_stage_hash')!=pm['stage_hash']: raise SystemExit('FAIL Backbone cache protocol hash mismatch')
expected={'screening':{'train','val'},'formal':{'train','val'},'evaluation':{'test'}}[a.stage]
actual=set()
for split in expected:
    p=cache/f'{split}_manifest.csv'
    if not p.exists(): raise SystemExit(f'FAIL missing {p}')
    actual.add(split); meta=pd.read_csv(p); proto=pviews[split]
    cols=['protocol_view_id','view_id','view_group_id','family','severity','distribution']
    if not meta[cols].astype(str).reset_index(drop=True).equals(proto[cols].astype(str).reset_index(drop=True)):
        raise SystemExit(f'FAIL {a.stage}/{split} differs from frozen Shared Protocol')
    print('PASS',a.stage,split,'rows',len(meta),'families',meta.family.value_counts().to_dict())
if actual!=expected: raise SystemExit('FAIL split set')
print('PASS Shared Protocol stage hash',pm['stage_hash']); print('CACHE VALIDATION PASS')
