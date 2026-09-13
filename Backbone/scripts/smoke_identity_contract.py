#!/usr/bin/env python3
from pathlib import Path
import json,sys,tempfile,yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
COMMON=ROOT.parent/'ExperimentCommon'; sys.path.insert(0,str(COMMON))
from backbone.identity import load_cache_identity,require_identity,require_cache_version_compatibility
cfg=yaml.safe_load((ROOT/'configs/backbone_selection.yaml').read_text())
with tempfile.TemporaryDirectory() as td:
    p=Path(td)
    m={'contract_version':'1.0.0','cache_schema_version':1,'data_contract_revision':'traffident_experiment_cache_v1','time_feature_revision':'traffident_time_features_v1','metrics_revision':'traffident_metrics_v1','prediction_contract_revision':'traffident_prediction_v1','stage':'development','subgraph':'TraffiDent_Net_06','protocol_stage_hash':'P','split_protocol_view_hashes':{'train':'T','val':'V'},'cache_fingerprint':'C','contract_fingerprint':'CF','time_feature_fingerprint':'TF','time_feature_source':'core/time_index.csv','time_feature_parser':'datetime_string'}
    (p/'cache_manifest.json').write_text(json.dumps(m))
    ident=load_cache_identity(p); require_cache_version_compatibility(cfg,ident,'screening'); require_identity({'run_identity':ident},ident)
    bad=dict(ident); bad['cache_fingerprint']='X'
    try: require_identity({'run_identity':bad},ident); raise AssertionError('mismatch accepted')
    except RuntimeError: pass
print('BACKBONE IDENTITY CONTRACT SMOKE PASS')
