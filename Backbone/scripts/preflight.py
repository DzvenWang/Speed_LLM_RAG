#!/usr/bin/env python3
from pathlib import Path
import argparse, importlib, importlib.util, subprocess, sys
GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
COMMON=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/ExperimentCommon')
if not COMMON.exists(): COMMON=ROOT.parent/'ExperimentCommon'
for p in [ROOT,COMMON,ROOT.parent/'Protocol',ROOT.parent/'TraffiDent_SupportBench']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from backbone.utils import load_yaml
from experiment_common.constants import CONTRACT_VERSION,DATA_CONTRACT_REVISION,TIME_FEATURE_REVISION,METRICS_REVISION,PREDICTION_CONTRACT_REVISION
from experiment_common.time_features import resolve_time_features,assert_resolved_time_features
from experiment_common.utils import load_yaml as load_shared_yaml

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',default=None); ap.add_argument('--stage',choices=['screening','formal','evaluation'],default='screening'); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/backbone_selection.yaml'); print('Backbone preflight v'+str(cfg['version'])); failed=False
    for p in [ROOT,COMMON,Path(cfg['project']['protocol_root']),Path(cfg['project']['protocol_artifact_root']),Path(cfg['project']['supportbench_project']),Path(cfg['project']['benchmark_root'])]:
        ok=p.exists(); print(('PASS' if ok else 'FAIL'),p); failed|=not ok
    cp=Path(cfg['shared_contract']['config']); cp=cp if cp.exists() else COMMON/'configs/experiment_contract.yaml'
    try:
        sc=load_shared_yaml(cp)
        checks=[('contract version',sc['version'],cfg['shared_contract']['required_version']),('data contract',DATA_CONTRACT_REVISION,cfg['shared_contract']['data_contract_revision']),('time feature',TIME_FEATURE_REVISION,cfg['shared_contract']['time_feature_revision']),('metrics',METRICS_REVISION,cfg['shared_contract']['metrics_revision']),('prediction contract',PREDICTION_CONTRACT_REVISION,cfg['shared_contract']['prediction_contract_revision'])]
        for name,got,exp in checks:
            ok=str(got)==str(exp); print(('PASS' if ok else 'FAIL'),name,got); failed|=not ok
    except Exception as e:
        print('FAIL shared contract',repr(e)); failed=True; sc=None
    try:
        subprocess.check_call(['/scratch/mh20/zw1165/envs/TFP/bin/python',str(ROOT/'scripts/source_lock.py'),'verify','--root',str(ROOT),'--sources',str(ROOT/'configs/backbone_selection.yaml')])
    except Exception: failed=True
    for pkg in ['torch','torchinfo','numpy','pandas','yaml','scipy','sklearn','fastdtw']:
        ok=importlib.util.find_spec(pkg) is not None; print(('PASS' if ok else 'FAIL'),pkg); failed|=not ok
    if a.model=='PatchSTG':
        ok=importlib.util.find_spec('timm') is not None; print(('PASS' if ok else 'FAIL'),'timm (PatchSTG)'); failed|=not ok
    try:
        sys.path.insert(0,str(ROOT/'vendor'/'STDN')); importlib.import_module('graph_utils'); print('PASS STDN graph_utils import')
    except Exception as e: print('FAIL STDN graph_utils import',repr(e)); failed=True
    try:
        from supportbench.dataset import SupportBench
        b=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
        print('PASS SupportBench',b.sg_dir,'N=',b.N,'H=',b.H,'P=',b.P)
        tc=sc['time_features']; tf=resolve_time_features(b,allow_index_fallback=bool(tc.get('allow_index_modulo_fallback',False)),index_origin_dow=int(tc.get('index_origin_dow',0))); audit=assert_resolved_time_features(tf)
        print('PASS time feature source',tf.source,'parser=',tf.parser,'TOD_unique=',audit['tod_unique'],'DOW_unique=',audit['dow_unique'],'fingerprint=',audit['fingerprint'])
        if a.model=='PatchSTG':
            import pandas as pd, numpy as np
            lat=pd.to_numeric(b.nodes.get('lat'),errors='coerce').to_numpy(float); lng=pd.to_numeric(b.nodes.get('lng'),errors='coerce').to_numpy(float); geo=float((np.isfinite(lat)&np.isfinite(lng)).mean()); print('PASS PatchSTG static node metadata geographic_fraction=',f'{geo:.3f}')
    except Exception as e: print('FAIL SupportBench/time feature',repr(e)); failed=True
    try:
        c=importlib.import_module('shared_protocol.consumer'); ps=cfg['protocol']['stage_map'][a.stage]; pm,_=c.load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],cfg['protocol']['name'],ps); print('PASS frozen Protocol/'+ps,pm['stage_hash'])
    except Exception as e: print('FAIL Protocol',repr(e)); failed=True
    if failed: raise SystemExit(2)
    print('PREFLIGHT PASS')
if __name__=='__main__': main()
