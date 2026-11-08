import json,importlib.util
from pathlib import Path
import pytest,torch


def read_module():
    p=Path(__file__).resolve().parents[1]/'scripts'/'summarize_v3_attribution.py'
    spec=importlib.util.spec_from_file_location('tcrr_summarizer',p)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def test_contribution_provenance_and_gain_math(tmp_path):
    mod=read_module();root=tmp_path/'results';ds='TraffiDent_Net_06'
    for i,level in enumerate(('M0','M1','M2','M3')):
        folder=root/ds/'formal'/f'run_{level}';folder.mkdir(parents=True)
        cfg={'version':'3.0.0','v3':{'contribution':level},
             'experiment':{'dataset':ds,'target':'speed','seed':10,'history':12,'horizon':12,'backbone_mode':'incomplete'},
             'retrieval':{'top_k':12},'training':{'batch_size':64,'max_epochs':80,'schedule':'frozen','lr_new':.001}}
        torch.save({'version':'3.0.0','backbone_sha256':'same_initialization',
                    'config':cfg,'best_epoch':10,'best_val_mae':2.0-.1*i},folder/'best.pt')
        (folder/'summary.json').write_text(json.dumps({'validation':{'MAE':2.0-.1*i,'RMSE':3.0}}))
    out=mod.summarize(root,ds,tmp_path/'out')
    assert abs(out.loc['validation','C2_MAE_gain']-.1)<1e-6
    assert (tmp_path/'out'/'provenance.json').exists()
    # Independently trained models must share the exact same backbone.
    ck=root/ds/'formal'/'run_M2'/'best.pt'
    data=torch.load(ck,weights_only=False);data['backbone_sha256']='different';torch.save(data,ck)
    with pytest.raises(RuntimeError,match='protocol mismatch'):
        mod.summarize(root,ds,tmp_path/'out_bad')
