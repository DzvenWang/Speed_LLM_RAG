"""Safety tests for a legitimate chronological producer and full failure-row retention."""
import json
import numpy as np
import pandas as pd
import pytest
from fare.oof import plan_folds,row_keys,verify_paired,sha_json
from fare.memory import from_support_cache,EpisodeMemory
from fare.config import hash_file


def create_cache(tmp_path, nrows=100, duplicate=False):
    cache=tmp_path/'cache';cache.mkdir()
    starts=np.arange(nrows,dtype=np.int64)*40+100
    if duplicate:starts[1]=starts[0]
    df=pd.DataFrame({'window_start':starts,'family':['regional_outage' if i%2 else 'entry_random' for i in range(nrows)],
                     'severity':[40]*nrows,'view_id':[f'view_{i}' for i in range(nrows)]})
    df.to_csv(cache/'train_manifest.csv',index=False)
    n=3;h=12;p=12
    arrays={'x_obs_norm':np.zeros((nrows,h,n),dtype=np.float32),'history_observed':np.ones((nrows,h,n),bool),
            'x_full_norm':np.zeros((nrows,h,n),dtype=np.float32),
            'artificial_missing':np.zeros((nrows,h,n),bool),
            'y_norm':np.ones((nrows,p,n),dtype=np.float32),
            'y_raw':np.ones((nrows,p,n),dtype=np.float32),
            'future_valid':np.ones((nrows,p,n),bool),
            'direct_mask':np.zeros((nrows,n),bool),
            'tod_hist':np.zeros((nrows,h),dtype=np.int16),
            'dow_hist':np.zeros((nrows,h),dtype=np.int16)}
    for name,arr in arrays.items():np.save(cache/f'train_{name}.npy',arr)
    np.save(cache/'adjacency.npy',np.eye(3,dtype=np.float32))
    return cache,df


def test_planner_uses_strict_target_end_and_purge(tmp_path):
    _,df=create_cache(tmp_path)
    folds=plan_folds(df,12,12,nfolds=4,warmup=.3,purge=12,minimum_train=8,minimum_validation=4)
    assert len(folds)==4
    assert all(f['fit_label_end']+12 < min(df.window_start.iloc[f['val_rows']]) for f in folds)
    assert all(f['val_label_end']+12 < f['evaluate_from'] for f in folds)
    scored=[r for f in folds for r in f['score_rows']]
    assert len(scored)==len(set(scored))
    assert len(scored)>60


def test_row_key_keeps_same_time_different_failure_and_identical_views(tmp_path):
    _,df=create_cache(tmp_path,duplicate=True)
    keys=row_keys(df)
    assert len(keys)==len(df)==len(set(keys))
    assert df.window_start.iloc[0]==df.window_start.iloc[1]
    assert keys[0]!=keys[1]


def test_legacy_oof_file_refused(tmp_path):
    cache,df=create_cache(tmp_path)
    f=tmp_path/'bad.npz'
    np.savez(f,window_start=df.window_start.to_numpy(),trained_until=np.zeros(len(df)),
             prediction=np.zeros((len(df),12,3)),prediction_complete=np.zeros((len(df),12,3)))
    class DS:
        path=cache;meta=df;h=12;p=12;n=3
    with pytest.raises(RuntimeError,match='schema-6'):
        verify_paired(f,DS())


def test_oof_identity_and_fold_cutoff_audited(tmp_path):
    cache,df=create_cache(tmp_path,duplicate=True)
    folder=tmp_path/'fold';folder.mkdir()
    ck=folder/'best.pt';ck.write_bytes(b'fixture checkpoint')
    idx=np.array([2,3,4],np.int64)
    file=tmp_path/'good.npz'
    np.savez(file,row_id=idx,episode_key=row_keys(df)[idx],window_start=df.window_start.to_numpy()[idx],
             fold_id=np.zeros(len(idx),np.int16),trained_until=np.full(len(idx),30,np.int64),
             prediction=np.zeros((len(idx),12,3),np.float32),prediction_complete=np.zeros((len(idx),12,3),np.float32))
    proof={'plan_sha256':'plan','fold_id':0,'val_label_end':30,'fit_label_end':20,'evaluate_from':180,
           'purge':12}
    doc={'schema_version':6,'oof_sha256':hash_file(file),'records':3,'normalization':'fold_prefix_valid_history',
         'source_plan_sha256':'plan','cache_manifest_sha256':hash_file(cache/'train_manifest.csv'),
         'producers':[{'fold_id':0,'checkpoint':str(ck),'checkpoint_sha256':hash_file(ck),'training_provenance':proof}]}
    file.with_suffix('.json').write_text(json.dumps(doc))
    class DS:
        path=cache;meta=df;h=12;p=12;n=3
    assert len(verify_paired(file,DS())['row_id'])==3
    tampered=np.load(file)
    np.savez(file,**{k:(np.array(['corrupted']*len(idx)) if k=='episode_key' else tampered[k]) for k in tampered.files})
    with pytest.raises(RuntimeError,match='episode identity'):
        verify_paired(file,DS())


def test_full_memory_retains_duplicate_time_episodes(tmp_path,monkeypatch):
    cache,df=create_cache(tmp_path,nrows=4,duplicate=True)
    from fare import oof
    source=tmp_path/'oof.npz';indices=np.arange(4)
    np.savez(source,row_id=indices,episode_key=row_keys(df),fold_id=np.zeros(4,dtype=np.int16),
             window_start=df.window_start.to_numpy(),trained_until=np.zeros(4,dtype=np.int64),
             prediction=np.zeros((4,12,3),np.float32),prediction_complete=np.ones((4,12,3),np.float32))
    monkeypatch.setattr(oof,'verify_paired',lambda path,ds,strict=True:np.load(path,allow_pickle=False))
    bank=from_support_cache(cache,None,10,mode='external_oos_paired',external_base_file=source)
    assert bank.m==4 and bank.starts[0]==bank.starts[1]
    assert bank.episode_info['family'][0]!=bank.episode_info['family'][1]
    bank.save(tmp_path/'memory')
    restored=EpisodeMemory.load(tmp_path/'memory')
    assert restored.m==4
    assert np.array_equal(restored.episode_info['episode_key'],row_keys(df))


def test_fold_training_and_predictor_checkpoint_interface(tmp_path,monkeypatch):
    """Train via real Backbone.adapter code using tiny official-API-compatible vendor fixture."""
    from fare import oof as producer,backbone as fb
    import torch,yaml
    cache,df=create_cache(tmp_path,nrows=100)
    monkeypatch.setattr(producer,'root',lambda:tmp_path)
    monkeypatch.setattr(fb,'root',lambda:tmp_path)
    modeldir=tmp_path/'Backbone/vendor/STLAformer/model';modeldir.mkdir(parents=True)
    (modeldir/'STLformer.py').write_text('''
import torch
from torch import nn
class Attn(nn.Module):
 def forward(self,z,ne,te):return z
class STLformer(nn.Module):
 def __init__(self,num_nodes,in_steps,out_steps,steps_per_week,input_dim,output_dim,input_embedding_dim,tow_embedding_dim,feed_forward_dim,num_heads,num_layers,dropout):
  super().__init__()
  self.num_nodes=num_nodes;self.steps_per_week=steps_per_week;self.input_dim=input_dim;self.tow_embedding_dim=tow_embedding_dim
  self.model_dim=input_embedding_dim+tow_embedding_dim
  self.input_proj=nn.Linear(input_dim,input_embedding_dim)
  self.tow_embedding=nn.Embedding(steps_per_week,tow_embedding_dim)
  self.attn_layers_st=nn.ModuleList([Attn() for _ in range(num_layers)])
  self.output_proj=nn.Linear(in_steps*self.model_dim,out_steps)
  self.out_steps=out_steps;self.output_dim=output_dim
 def node_position_encoding(self,z):return torch.zeros_like(z)
 def time_position_encoding(self,z):return torch.zeros_like(z)
 def MLP(self,z):return torch.zeros_like(z)
 def forward(self,x):
  t=x[...,1];z=torch.cat([self.input_proj(x),self.tow_embedding((t*self.steps_per_week).long().clamp(0,self.steps_per_week-1))],-1)
  for l in self.attn_layers_st:z=l(z,None,None)
  b,h,n,d=z.shape
  return self.output_proj(z.transpose(1,2).reshape(b,n,h*d)).transpose(1,2).unsqueeze(-1)
''')
    conf=tmp_path/'Backbone/configs';conf.mkdir(parents=True)
    recipes={'models':{'STLAformer':{'batch_size':4,'max_epochs':2,'patience':2,'optimizer':'adam',
       'learning_rate':.001,'scheduler':'none','train_loss':'mae',
       'input_dim':2,'input_embedding_dim':4,'tow_embedding_dim':4,'feed_forward_dim':8,
       'num_heads':2,'num_layers':2,'dropout':0.,'steps_per_week':2016}}}
    (conf/'model_recipes.yaml').write_text(yaml.safe_dump(recipes))
    ref=tmp_path/'reference.pt'
    torch.save({'model':'STLAformer','target':'speed','num_nodes':3,'mean':0.,'std':1.},ref)
    plan=producer.save_plan(cache,None,'speed','TraffiDent_Net_06',tmp_path/'plan.json',nfolds=1,warmup=.6,purge=12)
    fold=plan['folds'][0]
    checkpoint=producer.train_fold({},plan,fold,ref,tmp_path/'producer',epochs=2,batch_size=4,device=torch.device('cpu'))
    assert (tmp_path/'producer/COMPLETED.json').is_file()
    assert (tmp_path/'producer/last.pt').is_file()
    ix,p,c,ck=producer.predict_fold(plan,fold,checkpoint,batch_size=4,device=torch.device('cpu'))
    assert len(ix)==fold['score_count']
    assert p.shape==(len(ix),12,3)
    assert c.shape==p.shape
    assert np.isfinite(p).all()
    side=producer.export_paired(plan,[checkpoint],tmp_path/'paired.npz',batch_size=4,device=torch.device('cpu'))
    class DS:
        path=cache;meta=df;h=12;p=12;n=3
    assert len(producer.verify_paired(tmp_path/'paired.npz',DS())['row_id'])==len(ix)
    assert side['records']==len(ix)
