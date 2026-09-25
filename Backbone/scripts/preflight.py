#!/usr/bin/env python3
from pathlib import Path
import argparse, importlib.util, json, subprocess, sys
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'TraffiDent_SupportBench'))
from backbone.utils import load_yaml, repository_root
from backbone.shared_cache import cache_path, validate_shared_cache, shared_contract_config
from backbone.identity import load_cache_identity, require_cache_version_compatibility

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',choices=['development','formal','evaluation'],required=True); ap.add_argument('--severity',type=int,choices=[20,40,60,90]); ap.add_argument('--config',default=str(ROOT/'configs/backbone.yaml')); a=ap.parse_args()
    if a.stage=='evaluation' and a.severity is None: raise SystemExit('--severity is required for evaluation')
    cfg=load_yaml(a.config); print('Backbone preflight v'+str(cfg['version'])); failed=False
    paths=[ROOT,repository_root()/'ExperimentCommon',repository_root()/'Protocol',repository_root()/'TraffiDent_SupportBench',Path(cfg['project']['benchmark_root']),Path(cfg['project']['cache_root'])]
    for p in paths:
        ok=p.exists(); print(('PASS' if ok else 'FAIL'),p); failed|=not ok
    try: subprocess.check_call([sys.executable,str(ROOT/'scripts/source_lock.py'),'verify','--config',a.config])
    except Exception: failed=True
    for pkg in ['torch','numpy','pandas','yaml','scipy','sklearn','timm']:
        ok=importlib.util.find_spec(pkg) is not None; print(('PASS' if ok else 'FAIL'),pkg); failed|=not ok
    try:
        scfg=shared_contract_config(cfg,a.stage,a.severity); print('PASS ExperimentCommon contract',scfg['version'],scfg['protocol']['name'],'severity',scfg['experiment']['severity'])
        ident=validate_shared_cache(cfg,a.stage,a.severity,allow_locked_evaluation=True); p=cache_path(cfg,a.stage,a.severity); require_cache_version_compatibility(cfg,ident,a.stage)
        print('PASS cache',p); print('PASS protocol stage hash',ident['protocol_stage_hash']); print('PASS cache fingerprint',ident['cache_fingerprint']); print('PASS time feature fingerprint',ident['time_feature_fingerprint'])
        split={'development':['train','val'],'formal':['train','val'],'evaluation':['test']}[a.stage]
        exp=cfg['protocol_assertions'][a.stage]; fams=cfg['experiment']['families']
        for s in split:
            m=pd.read_csv(p/f'{s}_manifest.csv'); print(f'PASS {a.stage}/{s} rows={len(m)} unique_groups={m.view_group_id.astype(str).nunique()} families={m.family.value_counts().to_dict()}')
            target=exp.get(f'{s}_rows'); per=exp.get(f'per_family_{s}')
            if target is not None and len(m)!=int(target): raise RuntimeError(f'{s} rows {len(m)} != {target}')
            if per is not None:
                counts=m.family.astype(str).value_counts().to_dict(); bad={f:counts.get(f,0) for f in fams if counts.get(f,0)!=int(per)}
                if bad: raise RuntimeError(f'{s} family balance failed {counts}')
    except Exception as e: print('FAIL shared cache',repr(e)); failed=True
    try:
        from supportbench.dataset import SupportBench
        b=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root']); print('PASS SupportBench',b.sg_dir,'N=',b.N,'H=',b.H,'P=',b.P)
        lat=pd.to_numeric(b.nodes.get('lat'),errors='coerce').to_numpy(float); lng=pd.to_numeric(b.nodes.get('lng'),errors='coerce').to_numpy(float); print('PASS PatchSTG static geographic_fraction=',f'{float((np.isfinite(lat)&np.isfinite(lng)).mean()):.3f}')
    except Exception as e: print('FAIL SupportBench',repr(e)); failed=True
    if a.stage=='evaluation':
        print('EVALUATION_ACCESS UNLOCKED (Protocol grant file not required)')
    if failed: raise SystemExit(2)
    print('PREFLIGHT PASS')
if __name__=='__main__': main()
