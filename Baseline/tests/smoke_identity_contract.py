from pathlib import Path
import json,tempfile,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'ExperimentCommon'))
from baseline.run_identity import load_cache_identity,build_run_manifest,identity_key
with tempfile.TemporaryDirectory() as td:
    p=Path(td); m={'contract_version':'1.0.0','cache_schema_version':1,'data_contract_revision':'traffident_experiment_cache_v1','time_feature_revision':'traffident_time_features_v1','metrics_revision':'traffident_metrics_v1','prediction_contract_revision':'traffident_prediction_v1','stage':'development','subgraph':'TraffiDent_Net_06','protocol_stage_hash':'P','split_protocol_view_hashes':{'train':'T','val':'V'},'cache_fingerprint':'C','contract_fingerprint':'CF','time_feature_fingerprint':'TF','time_feature_source':'core/time_index.csv','time_feature_parser':'datetime_string'}; (p/'cache_manifest.json').write_text(json.dumps(m))
    ident=load_cache_identity(p,'0.6.0'); run=build_run_manifest(p,'0.6.0','RAFT',10,'generic_retrieval'); assert identity_key(run)==identity_key(ident); assert run['code_project_version']=='0.6.0'
print('BASELINE IDENTITY CONTRACT SMOKE PASS')
