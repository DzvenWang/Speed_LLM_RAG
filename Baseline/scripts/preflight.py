#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.util,subprocess,sys
from pathlib import Path
GADI_ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
COMMON=(next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())).parent / 'Base_SupportBench'
if not COMMON.exists(): COMMON=ROOT.parent/'Base_SupportBench'
for p in [ROOT,COMMON,ROOT.parent/'Protocol',ROOT.parent/'Base_SupportBench']:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from baseline.registry import ORDER
from baseline.contracts import audit_repo_contract
from baseline.utils import load_yaml
from baseline.data import resolve_baseline_time_features
from baseline.shared_cache import shared_contract_config, protocol_name, EVALUATION_SEVERITIES
from experiment_common.time_features import assert_resolved_time_features
from experiment_common.constants import CONTRACT_VERSION,CACHE_SCHEMA_VERSION,DATA_CONTRACT_REVISION,TIME_FEATURE_REVISION,METRICS_REVISION,PREDICTION_CONTRACT_REVISION
from shared_protocol.consumer import load_frozen_stage
from baseline.backbones.frozen_registry import load_frozen_backbone_registry

ap=argparse.ArgumentParser()
ap.add_argument('--stage',choices=['development','formal','evaluation'],default='development')
ap.add_argument('--method',default=None)
ap.add_argument('--severity',type=int,choices=EVALUATION_SEVERITIES)
a=ap.parse_args()
if a.stage=='evaluation' and a.severity is None: raise SystemExit('--severity is required for evaluation')
cfg=load_yaml(ROOT/'configs/baseline.yaml')
print(f"Baseline preflight v{cfg['version']} stage={a.stage}"+(f" severity=s{a.severity}" if a.severity else '')+(f" method={a.method}" if a.method else ''))
failed=False
for p in [ROOT,COMMON,Path(cfg['project']['protocol_root']),Path(cfg['project']['protocol_artifact_root']),Path(cfg['project']['supportbench_project']),Path(cfg['project']['benchmark_root']),Path(cfg['project']['backbone_root'])]:
    ok=p.exists(); print(('PASS' if ok else 'FAIL'),p); failed|=not ok
try:
    sc=shared_contract_config(cfg,a.stage,a.severity)
    checks=[
      ('contract version',sc['version'],cfg['shared_contract']['required_version']),
      ('cache schema',CACHE_SCHEMA_VERSION,cfg['shared_contract']['required_cache_schema_version']),
      ('data contract',DATA_CONTRACT_REVISION,cfg['shared_contract']['data_contract_revision']),
      ('time feature',TIME_FEATURE_REVISION,cfg['shared_contract']['time_feature_revision']),
      ('metrics',METRICS_REVISION,cfg['shared_contract']['metrics_revision']),
      ('prediction contract',PREDICTION_CONTRACT_REVISION,cfg['shared_contract']['prediction_contract_revision']),
      ('protocol version',sc['protocol']['version'],cfg['shared_contract']['protocol_version']),
      ('paired group',sc['protocol'].get('paired_group_id',''),cfg['shared_contract']['paired_group_id']),
    ]
    for name,got,exp in checks:
        ok=str(got)==str(exp); print(('PASS' if ok else 'FAIL'),name,got); failed|=not ok
    expected_families=list(cfg['experiment']['families']); got_families=list(sc['experiment']['families'])
    ok=got_families==expected_families; print(('PASS' if ok else 'FAIL'),'Core-5 families',got_families); failed|=not ok
except Exception as e: print('FAIL shared contract',repr(e)); failed=True; sc=None
try: subprocess.check_call([cfg['project']['python'],str(ROOT/'scripts/source_lock.py'),'verify','--root',str(ROOT),'--sources',str(ROOT/'configs/official_sources.yaml')])
except Exception: failed=True
for m in ORDER:
    repo=ROOT/'vendor'/m
    if not repo.exists(): print('FAIL missing repo',m); failed=True; continue
    c=audit_repo_contract(ROOT,m); ok=c['all_entrypoints_present']; print(('PASS' if ok else 'FAIL'),'entrypoint',m); failed|=not ok
for mod in ['torch','numpy','pandas','yaml','scipy','sklearn','tqdm']:
    ok=importlib.util.find_spec(mod) is not None; print(('PASS' if ok else 'FAIL'),mod); failed|=not ok
for mod in ['faiss','tsl','easydict','loguru','transformers','vllm']: print(('PASS' if importlib.util.find_spec(mod) else 'INFO optional missing'),mod)
bench=None; tf=None
try:
    from supportbench.dataset import SupportBench
    bench=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root']); ok=int(bench.N)==388 and int(bench.H)==12 and int(bench.P)==12; print(('PASS' if ok else 'FAIL'),'SupportBench',bench.sg_dir,'N=',bench.N,'H=',bench.H,'P=',bench.P); failed|=not ok
    r=resolve_baseline_time_features(bench,cfg); tf=assert_resolved_time_features(r); print('PASS time feature source',tf['source'],'parser=',tf['parser'],'TOD_unique=',tf['tod_unique'],'DOW_unique=',tf['dow_unique'],'fingerprint=',tf['fingerprint'])
except Exception as e: print('FAIL SupportBench/time features',repr(e)); failed=True
try:
    pname=protocol_name(cfg,a.stage,a.severity)
    pm,_=load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],pname,a.stage)
    print('PASS frozen Protocol',pname,a.stage,pm['stage_hash'])
except Exception as e: print('FAIL Protocol',a.stage,repr(e)); failed=True; pm=None
need_patch=False
if a.method:
    mc=cfg.get('methods',{}).get(a.method)
    if mc is None: print('FAIL unknown method',a.method); failed=True
    else: need_patch=bool(mc.get('requires_frozen_backbone_for_final_forecast',False))
fb=Path(cfg['project']['frozen_backbone_json'])
if need_patch or fb.exists():
    try:
        fpm,_=load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],cfg['protocol']['shared_protocol_name'],'formal'); con=cfg['frozen_backbone_contract']
        obj=load_frozen_backbone_registry(fb,expected_protocol_hash=fpm['stage_hash'],expected_time_feature_revision=cfg['shared_contract']['time_feature_revision'],expected_time_feature_fingerprint=(tf['fingerprint'] if tf else None),expected_time_feature_source=(tf['source'] if tf else None),expected_dataset=cfg['experiment']['subgraph'],expected_target=cfg['experiment']['target'],expected_backbone_project_version=con['required_project_version'],expected_backbone_data_contract_revision=con['required_data_contract_revision'],expected_shared_contract_version=con['required_shared_contract_version'],expected_cache_schema_version=con['required_cache_schema_version'],expected_protocol_version=con['required_protocol_version'],expected_paired_group_id=con['required_paired_group_id'],expected_metrics_revision=cfg['shared_contract']['metrics_revision'],expected_prediction_contract_revision=cfg['shared_contract']['prediction_contract_revision'])
        print('PASS frozen Backbone',obj['selected_model'],'reference',obj['reference_checkpoint'])
    except Exception as e:
        if need_patch: print('FAIL frozen Backbone',repr(e)); failed=True
        else: print('INFO frozen Backbone not ready yet:',repr(e))
if a.stage=='evaluation':
    print('PASS Baseline evaluation policy direct_after_completed_formal_run (no FORMAL_SELECTION/FROZEN_BASELINE_PROTOCOL required)')
if failed: raise SystemExit(2)
print('PREFLIGHT PASS')
