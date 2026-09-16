#!/usr/bin/env python3
from pathlib import Path
import json,sys,subprocess
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT))
from baseline.utils import load_yaml,write_json
from baseline.registry import METHODS
from baseline.run_identity import build_run_manifest, write_run_manifest
from baseline.shared_cache import cache_path
from baseline.adapters.base import AdapterContext
from baseline.adapters.imputeformer import ImputeFormerAdapter
from baseline.adapters.lrtc_tmcp import LRTCTMCPAdapter
from baseline.adapters.ginar import GinARAdapter
from baseline.adapters.raft import RAFTAdapter
from baseline.adapters.rast import RASTAdapter
from baseline.adapters.saraf import SARAFAdapter
from baseline.adapters.leaf import LEAFAdapter
from baseline.adapters.slpf import SLPFAdapter
from baseline.adapters.sprint import SPRINTAdapter
from baseline.adapters.mogernn import MoGERNNAdapter
from baseline.adapters.remft import ReMFTAdapter

cfg=load_yaml(ROOT/'configs/baseline.yaml'); stage=sys.argv[1] if len(sys.argv)>1 else 'development'
if stage not in ('development','formal'): raise SystemExit('prepare_bridges supports development/formal only; Test is isolated.')
subprocess.check_call([cfg['project']['python'],str(ROOT/'scripts/validate_cache.py'),'--stage',stage])
cache=cache_path(cfg,stage)
ctx=AdapterContext(ROOT,cache,Path(cfg['project']['exp_root']),Path(cfg['project']['log_root']),Path(cfg['project']['frozen_backbone_json']))
out=Path(cfg['project']['exp_root'])/'_bridges'/stage; out.mkdir(parents=True,exist_ok=True)
classes=[ImputeFormerAdapter,LRTCTMCPAdapter,SPRINTAdapter,GinARAdapter,RAFTAdapter,RASTAdapter,SARAFAdapter,LEAFAdapter,ReMFTAdapter,SLPFAdapter,MoGERNNAdapter]
for cls in classes:
    a=cls(ctx); method=a.method; spec=METHODS[method]
    run=build_run_manifest(cache,None,method,cfg['experiment']['seed'],cfg['methods'][method]['formal_track'],{
        'artifact_type':'bridge_contract','requires_frozen_backbone':bool(spec['final_forecast_requires_frozen_backbone']),
        'table_group':spec['table_group'],'runner_project_version':cfg['version']})
    md=out/method; md.mkdir(parents=True,exist_ok=True); write_run_manifest(md/'RUN_MANIFEST.json',run)
    d={'run_identity':run,'compatibility':a.compatibility_audit(),'bridge':a.prepare_bridge()}; write_json(md/'bridge.json',d); write_json(out/f'{method}.json',d)
    print(method,json.dumps({'protocol_stage_hash':run['protocol_stage_hash'],'cache_fingerprint':run['cache_fingerprint'],'track':run['track'],'table_group':spec['table_group']},sort_keys=True))
print('bridge contracts ready:',out)
