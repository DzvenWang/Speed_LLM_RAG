#!/usr/bin/env python3
from pathlib import Path
import json, sys
ROOT=next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd()); sys.path.insert(0,str(ROOT))
from baseline.registry import ORDER, METHODS
from baseline.contracts import audit_repo_contract
from baseline.utils import write_json, load_yaml
cfg=load_yaml(ROOT/'configs/baseline.yaml'); out=Path(cfg['project']['exp_root'])/'_audit'; out.mkdir(parents=True,exist_ok=True)
lockp=ROOT/'vendor'/'SOURCE_LOCK.json'; lock=json.loads(lockp.read_text()) if lockp.exists() else {'repositories':{}}
res={}
for m in ORDER:
    c=audit_repo_contract(ROOT,m); spec=METHODS[m]; rec=lock.get('repositories',{}).get(m,{})
    c['expected_commit']=spec['commit']; c['resolved_commit']=rec.get('commit'); c['source_policy']=spec['source_policy']; c['execution_state']=spec['execution_state']; c['table_group']=spec['table_group']; c['research_question']=spec['question']
    c['formal_track']=cfg['methods'][m]['formal_track']
    res[m]=c
    print(m,json.dumps({'entrypoints_ok':c['all_entrypoints_present'],'group':c['table_group'],'formal_track':c['formal_track'],'execution_state':c['execution_state'],'resolved_commit':c['resolved_commit']},sort_keys=True))
write_json(out/'official_compatibility_audit.json',res); print('wrote',out/'official_compatibility_audit.json')
if not all(v['all_entrypoints_present'] for v in res.values()): raise SystemExit(2)
print('OFFICIAL COMPATIBILITY AUDIT PASS')
