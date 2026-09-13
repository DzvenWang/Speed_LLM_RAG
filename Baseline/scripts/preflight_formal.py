#!/usr/bin/env python3
from __future__ import annotations
import argparse, importlib.util, json, subprocess, sys
from pathlib import Path

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baseline.utils import load_yaml
from baseline.shared_cache import cache_path
from baseline.backbones.frozen_registry import load_frozen_patchstg_registry

ap = argparse.ArgumentParser()
ap.add_argument('--method', required=True, choices=['GinAR','ImputeFormer','RAFT','RAST','SARAF','RAST-PatchSTG'])
ap.add_argument('--seed', type=int, required=True)
ap.add_argument('--stage', choices=['formal','evaluation'], default='formal')
a = ap.parse_args()
cfg = load_yaml(ROOT / 'configs/baseline.yaml')
if a.seed not in [int(x) for x in cfg['experiment']['formal_seeds']]:
    raise SystemExit(f'Invalid formal seed {a.seed}')
subprocess.check_call([cfg['project']['python'], str(ROOT / 'scripts/validate_cache.py'), '--stage', a.stage])

need_patch = a.method in ('ImputeFormer', 'RAST-PatchSTG')
if need_patch:
    formal_manifest = cache_path(cfg, 'formal') / 'cache_manifest.json'
    if not formal_manifest.exists():
        raise SystemExit('Missing Baseline formal cache; frozen PatchSTG compatibility cannot be checked.')
    cm = json.loads(formal_manifest.read_text())
    contract = cfg.get('frozen_backbone_contract', {})
    load_frozen_patchstg_registry(
        cfg['project']['frozen_backbone_json'],
        expected_protocol_hash=cm['protocol_stage_hash'],
        expected_time_feature_revision=cm['time_feature_revision'],
        expected_time_feature_fingerprint=cm['time_feature_fingerprint'],
        expected_time_feature_source=cm['time_feature_source'],
        expected_dataset=cfg['experiment']['subgraph'],
        expected_backbone_project_version=str(contract.get('required_project_version', '0.5.0')),
    )

for dep in ({'ImputeFormer':['tsl'], 'RAST':['faiss'], 'RAST-PatchSTG':['faiss']}.get(a.method, [])):
    if importlib.util.find_spec(dep) is None:
        raise SystemExit(f'Missing required dependency for {a.method}: {dep}')
runner = {
    'GinAR':'ginar_runner.py', 'ImputeFormer':'imputeformer_runner.py', 'RAFT':'raft_runner.py',
    'RAST':'rast_runner.py', 'RAST-PatchSTG':'rast_runner.py', 'SARAF':'saraf_runner.py',
}[a.method]
if not (ROOT / 'baseline/runners' / runner).exists():
    raise SystemExit(f'Missing runner {runner}')
print(f'FORMAL PREFLIGHT PASS method={a.method} seed={a.seed} stage={a.stage}')
