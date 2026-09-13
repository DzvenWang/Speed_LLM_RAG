#!/usr/bin/env python3
from pathlib import Path
import sys, yaml
ROOT=Path(__file__).resolve().parent
for p in (ROOT/'ExperimentCommon', ROOT/'Backbone', ROOT/'Baseline'):
    sys.path.insert(0,str(p))
from ExperimentCommon.experiment_common.constants import (
    CONTRACT_VERSION, DATA_CONTRACT_REVISION, TIME_FEATURE_REVISION,
    METRICS_REVISION, PREDICTION_CONTRACT_REVISION,
)

b=yaml.safe_load((ROOT/'Backbone/configs/backbone_selection.yaml').read_text())
l=yaml.safe_load((ROOT/'Baseline/configs/baseline.yaml').read_text())
c=yaml.safe_load((ROOT/'ExperimentCommon/configs/experiment_contract.yaml').read_text())
assert str(b['version'])=='0.5.0'
assert str(l['version'])=='0.6.0'
assert str(c['version'])==CONTRACT_VERSION=='1.0.0'
for name,cfg in [('Backbone',b),('Baseline',l)]:
    sc=cfg['shared_contract']
    assert str(sc['required_version'])==CONTRACT_VERSION, (name,'contract')
    assert sc['data_contract_revision']==DATA_CONTRACT_REVISION, (name,'data')
    assert sc['time_feature_revision']==TIME_FEATURE_REVISION, (name,'time')
    assert sc['metrics_revision']==METRICS_REVISION, (name,'metrics')
    assert sc['prediction_contract_revision']==PREDICTION_CONTRACT_REVISION, (name,'prediction')
assert b['selection']['reference_model']=='PatchSTG'
assert l['frozen_backbone_contract']['required_project_version']=='0.5.0'
assert l['frozen_backbone_contract']['use_reference_checkpoint_for_all_baseline_seeds'] is True

for project in ('Backbone','Baseline'):
    proot=ROOT/project
    for base in (proot/'backbone' if project=='Backbone' else proot/'baseline', proot/'scripts', proot/'pbs'):
        if not base.exists():
            continue
        for f in base.rglob('*'):
            if not f.is_file() or f.suffix not in {'.py','.sh','.pbs'}:
                continue
            if f.name in {'smoke_direct_cache_contract.py','cleanup_legacy_cache_aliases.py'}:
                continue
            txt=f.read_text(errors='ignore')
            assert "'_cache'" not in txt and '"_cache"' not in txt and '/_cache/' not in txt, (project,f,'local cache alias')
            assert 'ensure_consumer_alias' not in txt, (project,f,'alias helper')

print('JOINT CONTRACT VALIDATION PASS')
