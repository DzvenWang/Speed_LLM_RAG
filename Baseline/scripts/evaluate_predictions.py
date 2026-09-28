#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,subprocess
ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd()); sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml, write_json
from baseline.shared_cache import cache_path
from baseline.evaluation import evaluate_prediction_file
from baseline.registry import METHODS
from baseline.run_identity import build_run_manifest
p=argparse.ArgumentParser(); p.add_argument('--stage',default='development',choices=['development','formal','evaluation']); p.add_argument('--split',choices=['val','test'],required=True); p.add_argument('--pred',required=True); p.add_argument('--out',required=True); p.add_argument('--method',default=None); p.add_argument('--seed',type=int,default=10); a=p.parse_args()
cfg=load_yaml(ROOT/'configs/baseline.yaml')
if a.stage in ('formal','evaluation') and not a.method: raise SystemExit('--method is required for formal/evaluation result registration')
if a.method and a.method not in METHODS: raise SystemExit(f'Unknown method {a.method}')
subprocess.check_call([cfg['project']['python'],str(ROOT/'scripts/validate_cache.py'),'--stage',a.stage])
cache=cache_path(cfg,a.stage)
df=evaluate_prediction_file(cache,a.split,a.pred); out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); df.to_csv(out,index=False)
method=a.method or 'UNSPECIFIED'; track=cfg['methods'][method]['formal_track'] if method in METHODS else 'development_smoke'
run=build_run_manifest(cache,None,method,a.seed,track,{'artifact_type':'prediction_evaluation','prediction_file':str(Path(a.pred).resolve()),'split':a.split,'metrics_file':str(out.resolve()),'status':'EVALUATED'})
write_json(out.with_suffix(out.suffix+'.RUN_MANIFEST.json'),run)
print(df.to_string(index=False)); print('PROTOCOL_STAGE_HASH='+run['protocol_stage_hash']); print('wrote',out)
