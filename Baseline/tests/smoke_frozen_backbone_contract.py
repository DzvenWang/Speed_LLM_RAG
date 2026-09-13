from pathlib import Path
import tempfile,json,hashlib,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from baseline.backbones.frozen_registry import load_frozen_patchstg_registry
with tempfile.TemporaryDirectory() as td:
    p=Path(td); ck=p/'best.pt'; ck.write_bytes(b'x'); sha=hashlib.sha256(b'x').hexdigest()
    ident={'identity_schema_version':3,'project':'Backbone','consumer_stage':'formal','shared_cache_stage':'formal','dataset':'TraffiDent_Net_06','protocol_stage_hash':'P','cache_fingerprint':'C','shared_contract_version':'1.0.0','data_contract_revision':'traffident_experiment_cache_v1','time_feature_revision':'traffident_time_features_v1','metrics_revision':'traffident_metrics_v1','prediction_contract_revision':'traffident_prediction_v1','time_feature_fingerprint':'TF','time_feature_source':'core/time_index.csv','time_feature_parser':'datetime_string','contract_fingerprint':'CF'}
    obj={'schema_version':3,'project_version':'0.5.0','selected_model':'PatchSTG','run_identity':ident,'formal_seeds':[10,20,30],'reference_seed':10,'reference_checkpoint':str(ck),'reference_checkpoint_sha256':sha}; mf=p/'FROZEN_BACKBONE.json'; mf.write_text(json.dumps(obj))
    got=load_frozen_patchstg_registry(mf,expected_protocol_hash='P',expected_time_feature_revision='traffident_time_features_v1',expected_time_feature_fingerprint='TF',expected_time_feature_source='core/time_index.csv'); assert got['reference_seed']==10
print('BASELINE FROZEN BACKBONE CONTRACT SMOKE PASS')
