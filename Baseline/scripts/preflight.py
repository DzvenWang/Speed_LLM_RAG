#!/usr/bin/env python3
import importlib.util, subprocess, sys
from pathlib import Path
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'Protocol')); sys.path.insert(0,str(ROOT.parent/'TraffiDent_SupportBench')); sys.dont_write_bytecode=True
from baseline.registry import ORDER
from baseline.contracts import audit_repo_contract
from baseline.utils import load_yaml
from shared_protocol.consumer import load_frozen_stage
cfg=load_yaml(ROOT/'configs/baseline.yaml'); print(f"Baseline preflight v{cfg['version']}"); failed=False
for p in [ROOT,Path(cfg['project']['protocol_root']),Path(cfg['project']['protocol_artifact_root']),Path(cfg['project']['supportbench_project']),Path(cfg['project']['benchmark_root']),Path(cfg['project']['backbone_root'])]:
    ok=p.exists(); print(('PASS' if ok else 'FAIL'),p); failed|=not ok
try:
    subprocess.check_call([cfg['project']['python'],str(ROOT/'scripts/source_lock.py'),'verify','--root',str(ROOT),'--sources',str(ROOT/'configs/official_sources.yaml')])
except Exception: failed=True
for m in ORDER:
    repo=ROOT/'vendor'/m
    if not repo.exists(): print('FAIL missing repo',m); failed=True; continue
    c=audit_repo_contract(ROOT,m)
    if not c['all_entrypoints_present']: print('FAIL official entrypoint contract',m,c['entrypoints']); failed=True
    else: print('PASS entrypoint contract',m)
for mod in ['torch','numpy','pandas','yaml','scipy','sklearn','tqdm']:
    ok=importlib.util.find_spec(mod) is not None; print(('PASS' if ok else 'FAIL'),mod); failed|=not ok
for mod in ['faiss','easydict','loguru','transformers','vllm']:
    print(('PASS' if importlib.util.find_spec(mod) else 'INFO optional missing'),mod)
try:
    from supportbench.dataset import SupportBench
    b=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
    ok=int(b.N)==388 and int(b.H)==int(cfg['experiment']['history']) and int(b.P)==int(cfg['experiment']['horizon'])
    print(('PASS' if ok else 'FAIL'),'SupportBench',b.sg_dir,'N=',b.N,'H=',b.H,'P=',b.P); failed|=not ok
except Exception as e: print('FAIL SupportBench',repr(e)); failed=True
try:
    pm,_=load_frozen_stage(cfg['project']['protocol_artifact_root'],cfg['experiment']['subgraph'],cfg['protocol']['shared_protocol_name'],'development')
    print('PASS frozen Protocol/development',pm['stage_hash'])
except Exception as e: print('FAIL Protocol/development',repr(e)); failed=True
if failed: raise SystemExit(2)
print('PREFLIGHT PASS')
