#!/usr/bin/env python3
from pathlib import Path
import sys, tempfile, json
import numpy as np
R=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(R))
from presure.utils import load_yaml
from presure.qwen_reasoner import (
    decision_label, parse_reasoner_output, proposal_prompt_features,
    format_reasoning_user, target_response, QwenDecisionMetrics, resolve_hf_snapshot,
)
cfg=load_yaml(R/'configs/presure.yaml')
assert cfg['version'] in {'0.8.0','0.9.0','0.9.1','0.9.2','0.9.2','0.9.3','0.9.4'}
q=cfg['qwen_support_value_probe']; assert q['labels']['actions']==['USE','REJECT','ABSTAIN']
# Label contract.
assert decision_label([.10,-.2],[1,1])['action']=='USE'
assert decision_label([-.10,-.2],[1,1])['action']=='REJECT'
assert decision_label([.01,-.01],[1,1])['action']=='ABSTAIN'
# Stable proposal-feature decoding (E=4, 29 scalar prefix + 3x2 contexts + unique count).
x=np.zeros(36,np.float32); x[0:4]=[.8,.7,.6,.5]; x[4]=1; x[16]=.9; x[17]=.8; x[20]=.2; x[22]=.5; x[23]=.3; x[25]=.1; x[26]=.2; x[27]=.4; x[-1]=.5
f=proposal_prompt_features(x); assert abs(f['need_probability']-.9)<1e-5 and f['proposed_by']==['generic']
r={'need_probability':.9,'horizon_index':3,'unique_proposals':2,'expert_disagreement':.33,'proposals':[{'id':'P0','valid':True,'proposed_by':['generic'],'agreement_ratio':.5,'m3_confidence':.8,'m3_margin':.2,'expert_probabilities':{'generic':.8,'hard':.7,'pred10':.6,'pred20':.5},'local_residual':.3,'residual_mean':.1,'residual_std':.2,'residual_maxabs':.4,'context_cosine':.7,'context_absdiff_mean':.1,'target_utility':.1}], 'label_action':'USE','label_proposal_index':0}
assert 'current decision context' in format_reasoning_user(r).lower(); assert target_response(r)=='{"action":"USE","proposal":"P0"}'
p=parse_reasoner_output('```json\n{"action":"USE","proposal":"P0"}\n```',['P0']); assert p['parse_ok'] and p['proposal']=='P0'
p=parse_reasoner_output('nonsense',['P0']); assert p['action']=='ABSTAIN' and not p['parse_ok']
m=QwenDecisionMetrics(); m.update(r,'USE','P0',True); sm=m.summary(); assert sm['selected_beneficial_rate']==1.0 and sm['mean_selection_regret']==0.0
# Local HF cache resolver contract without network.
with tempfile.TemporaryDirectory() as d:
    root=Path(d); snap=root/'snapshots'/'abc'; snap.mkdir(parents=True); (snap/'config.json').write_text('{}')
    assert resolve_hf_snapshot(root)==snap
print('PRESURE V0.8.0 QWEN SUPPORT-VALUE REASONING PROBE SMOKE PASS')
