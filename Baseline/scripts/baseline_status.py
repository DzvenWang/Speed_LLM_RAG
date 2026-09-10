#!/usr/bin/env python3
from pathlib import Path
import sys, json
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline'); sys.path.insert(0,str(ROOT))
from baseline.registry import ORDER, METHODS, MAIN_ORDER, LLM_ORDER, SUBTRACK_ORDER
from baseline.utils import load_yaml
from baseline.run_identity import load_cache_identity, identity_key
cfg=load_yaml(ROOT/'configs/baseline.yaml'); frozen=Path(cfg['project']['frozen_backbone_json']).exists(); exp=Path(cfg['project']['exp_root'])
print('Baseline status v0.4.0 — story-aligned')
identities={}
for stage in ['development','formal','evaluation']:
    mf=exp/'_cache'/cfg['experiment']['subgraph']/stage/'cache_manifest.json'
    if mf.exists():
        try: identities[stage]=load_cache_identity(mf.parent,cfg['version']); print(f'{stage:11s}_cache READY protocol={identities[stage]["protocol_stage_hash"][:12]} cache={identities[stage]["cache_fingerprint"][:12]}')
        except Exception as e: print(f'{stage:11s}_cache STALE/INVALID {e}')
    else: print(f'{stage:11s}_cache '+('LOCKED/NOT_BUILT' if stage=='evaluation' else 'NOT_BUILT'))
runtime=exp/'_audit'/'runtime_contract_audit.json'; leak=exp/'_audit'/'retrieval_leakage_development.json'
print('runtime_audit    ','READY' if runtime.exists() else 'MISSING'); print('retrieval_audit  ','READY' if leak.exists() else 'MISSING'); print('frozen_backbone  ','READY' if frozen else 'WAITING')
print('\nMAIN TABLE')
for m in MAIN_ORDER:
    s=METHODS[m]; repo=(ROOT/'vendor'/m).exists(); gate=s['final_forecast_requires_frozen_backbone'] and not frozen
    if gate: formal_state='WAIT_BACKBONE'
    elif s['execution_state']=='DEV_RUNNER_READY': formal_state='DEV_RUNNER_READY'
    else: formal_state='AUDIT_THEN_RUN'
    bridge=exp/'_bridges'/'development'/m/'RUN_MANIFEST.json'; idstate='MISSING'
    if bridge.exists() and 'development' in identities:
        try:
            x=json.loads(bridge.read_text()); idstate='MATCH' if identity_key(x)==identity_key(identities['development']) else 'STALE'
        except Exception: idstate='INVALID'
    print(f"{m:13s} repo={'READY' if repo else 'MISSING':7s} identity={idstate:7s} state={s['execution_state']:24s} formal={formal_state}")
print('\nLLM-SPECIFIC')
for m in LLM_ORDER: print(f"{m:13s} repo={'READY' if (ROOT/'vendor'/m).exists() else 'MISSING':7s} state={METHODS[m]['execution_state']}")
print('\nPARTIAL-SENSING SUBTRACK')
for m in SUBTRACK_ORDER: print(f"{m:13s} repo={'READY' if (ROOT/'vendor'/m).exists() else 'MISSING':7s} state={METHODS[m]['execution_state']}")
