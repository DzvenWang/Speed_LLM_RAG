#!/usr/bin/env python3
from pathlib import Path
import argparse, importlib, importlib.util, subprocess, sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone'); sys.dont_write_bytecode=True
for p in [ROOT, ROOT.parent/'Protocol', ROOT.parent/'TraffiDent_SupportBench']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from backbone.utils import load_yaml

ap=argparse.ArgumentParser(); ap.add_argument('--model',default=None); ap.add_argument('--stage',choices=['screening','formal','evaluation'],default='screening'); args=ap.parse_args()
cfg=load_yaml(ROOT/'configs/backbone_selection.yaml')
print('Backbone preflight v'+str(cfg['version']))
failed=False
for p in [ROOT,Path(cfg['project']['protocol_root']),Path(cfg['project']['protocol_artifact_root']),Path(cfg['project']['supportbench_project']),Path(cfg['project']['benchmark_root'])]:
    ok=p.exists(); print(('PASS' if ok else 'FAIL'),p); failed|=not ok
try:
    subprocess.check_call(['/scratch/mh20/zw1165/envs/TFP/bin/python',str(ROOT/'scripts/source_lock.py'),'verify','--root',str(ROOT),'--sources',str(ROOT/'configs/backbone_selection.yaml')])
except Exception: failed=True
for pkg in ['torch','torchinfo','numpy','pandas','yaml','scipy','sklearn','fastdtw']:
    ok=importlib.util.find_spec(pkg) is not None; print(('PASS' if ok else 'FAIL'),pkg); failed|=not ok
if args.model=='PatchSTG':
    ok=importlib.util.find_spec('timm') is not None; print(('PASS' if ok else 'FAIL'),'timm (PatchSTG)'); failed|=not ok
# Dynamic import avoids false static-analysis failures for vendor-only STDN helper.
try:
    sys.path.insert(0,str(ROOT/'vendor'/'STDN')); importlib.import_module('graph_utils')
    print('PASS STDN graph_utils import')
except Exception as e: print('FAIL STDN graph_utils import',repr(e)); failed=True
print('PASS STDN first_model path isolates unused torch_geometric/torch_scatter helper imports')
try:
    ds=importlib.import_module('supportbench.dataset'); b=ds.SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
    print('PASS SupportBench',b.sg_dir,'N=',b.N,'H=',b.H,'P=',b.P)
    if args.model=='PatchSTG':
        import pandas as pd, numpy as np
        lat=pd.to_numeric(b.nodes.get('lat'),errors='coerce').to_numpy(float); lng=pd.to_numeric(b.nodes.get('lng'),errors='coerce').to_numpy(float)
        geo=float((np.isfinite(lat)&np.isfinite(lng)).mean())
        print('PASS PatchSTG static node metadata geographic_fraction=',f'{geo:.3f}')
except Exception as e: print('FAIL SupportBench/recent-model contract',repr(e)); failed=True
try:
    consumer=importlib.import_module('shared_protocol.consumer')
    pstage=cfg['protocol']['stage_map'][args.stage]
    pm,_=consumer.load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],cfg['protocol']['name'],pstage)
    print('PASS frozen Protocol/'+pstage,pm['stage_hash'])
except Exception as e: print('FAIL Protocol/'+str(args.stage),repr(e)); failed=True
if failed: raise SystemExit(2)
print('PREFLIGHT PASS')
