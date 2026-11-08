"""End-to-end v9.3 CPU model / Teacher / Mask / Cache / checkpoint synthetic smoke.

Needs installed Base_SupportBench at sibling project root or PYTHONPATH.
Does NOT claim a real STLAformer/Gadi experiment.
"""
import json,shutil
from pathlib import Path
import numpy as np,pandas as pd,pytest,torch,yaml
from torch import nn
from tcrr_stla.v93.teacher import prepare_split,atomic_json
from tcrr_stla.v93.dataset import DirectViewDataset
from tcrr_stla.v93.model import TCRRV93
from tcrr_stla.v93.training import run_training
from tcrr_stla.v93.diagnostics.audit import diagnose_run


class FakeFrozenSTLA(nn.Module):
    def forward(self,x,m,tod,dow):
        z=(x*m).mean(1)
        f=(z[:,None]+torch.arange(12,dtype=x.dtype,device=x.device)[None,:,None]*.005)
        return f,torch.stack([z*(i+1)/8 for i in range(8)],-1)


def test_synthetic_full_pipeline(tmp_path):
    import supportbench.sensing as sensing
    project=tmp_path/'project';base=project/'Base_SupportBench';
    (base/'supportbench').mkdir(parents=True)
    # Reuse the actual installed official Base_SupportBench source, not a fake generator.
    src=Path(sensing.__file__).resolve()
    shutil.copy2(src,base/'supportbench/sensing.py')
    shutil.copy2(src.with_name('utils.py'),base/'supportbench/utils.py')
    (base/'supportbench/__init__.py').write_text('')
    (base/'configs').mkdir()
    (base/'configs/base_supportbench.yaml').write_text(yaml.safe_dump({'sensing':{'global_seed':10}}))
    sg=project/'data/processed/traffident/base_supportbench/speed/TraffiDent_Net_06'
    layouts=sg/'graph/layouts';layouts.mkdir(parents=True)
    for family in ('regional','directional'):
        cat={'severities':{f's{s}':{'id':[{'nodes':list(range(max(1,round(6*s/100)))),
                        'layout_id':f'{family}_{s}','component_count':1}]}
                        for s in (20,40,60,90)}}
        (layouts/f'{family}_layouts.json').write_text(json.dumps(cat))
    cache=tmp_path/'cache';cache.mkdir()
    rng=np.random.default_rng(10)
    for split,Q in [('train',60),('val',30)]:
        x=rng.normal(size=(Q,12,6)).astype('float32')
        artificial=np.zeros_like(x,dtype=bool)
        artificial[:,3:6,1:3]=True
        observed=~artificial
        y=(x.mean(1)[:,None,:]+.1).repeat(12,axis=1)
        arrays={'x_obs_norm':x*observed,'y_norm':y,'y_raw':y,
                'history_observed':observed,'future_valid':np.ones_like(y,dtype=bool),
                'direct_mask':np.ones_like(y,dtype=bool),
                'artificial_missing':artificial,
                'tod_hist':np.zeros((Q,12),dtype=np.int64),
                'dow_hist':np.zeros((Q,12),dtype=np.int64)}
        if split=='train':arrays['x_full_norm']=x
        for key,value in arrays.items():np.save(cache/f'{split}_{key}.npy',value)
        pd.DataFrame({'window_start':np.arange(Q,dtype=np.int64)*12+10}).to_csv(cache/f'{split}_manifest.csv',index=False)
    identity={'schema':930,'target':'speed','dataset':'TraffiDent_Net_06','test':True}
    root=tmp_path/'teacher';root.mkdir();atomic_json(root/'identity.json',identity)
    teacher=FakeFrozenSTLA();device=torch.device('cpu')
    prepare_split(cache,root/'train','train',teacher,device,project,'speed','TraffiDent_Net_06',10,
                  views=4,batch_size=5,identity=identity)
    prepare_split(cache,root/'val','val',teacher,device,project,'speed','TraffiDent_Net_06',10,
                  views=1,batch_size=5,identity=identity)
    train=DirectViewDataset(cache,root,'train',train_fraction=.8)
    assert len(train)>0
    a=train[0]
    assert 'rref' in a and 'cf_teacher' in a
    assert a['student']['x_obs'].shape==(12,6)
    settings={'model':{'mode':'incremental','history':12,'horizon':12,'hidden':16,'correction_bound':3.},
       'graph':{'mode':'none'},'preparation':{'limit':None,'val_limit':None},
       'counterfactual':{'paired_supervision':True,'paired_utility':True},
       'training':{'train_fraction':.8,'batch_size':8,'eval_batch_size':8,'workers':0,
          'lr':.001,'clip':5.,'amp_fp16':False,
          'epochs':{'residual':1,'cf_response':1,'utility':1,'cf_joint':1}},
       'loss':{'residual_forecast':1.,'response':1.,'damage':.05,'uncertainty':.01,
          'paired':.2,'forecast':1.,'utility':.05,'response_joint':.2,'paired_joint':.05,'cross_mask_utility':.02}}
    model=TCRRV93(8,6,hidden=16)
    out=tmp_path/'run'
    summary=run_training(cache,root,out,settings,model,seed=10,smoke=True)
    assert (out/'best_cf_joint.pt').exists()
    assert (out/'check_report.json').exists()
    diag=diagnose_run(out,cache,root,settings,None,None,project,'speed','TraffiDent_Net_06',model)
    assert np.isfinite(diag['cf_response_mae_norm'])
    assert np.isfinite(summary['check']['mae_norm'])
