#!/usr/bin/env python3
"""Offline CPU-only end-to-end integration against synthetic STLA/cache contracts.

IMPORTANT: A lightweight fake STLAformer substitutes for the real Backbone.
This catches stage interfaces, data shapes, replay, caching, checkpointing,
resume and report, but NOT fidelity with the real Gadi official vendor model.
Never use synthetic metrics as forecasting performance claims.
"""
from __future__ import annotations
import os,sys,tempfile,subprocess,textwrap,json,importlib.util
from pathlib import Path
import numpy as np,pandas as pd,torch

PROJ=Path(__file__).resolve().parents[1]
VENDOR='''
import torch
from torch import nn
class DummyAttention(nn.Module):
 def __init__(self,d):
  super().__init__();self.linear=nn.Linear(d,d)
 def forward(self,h,node_emb,time_emb):
  return h + 0.01 * self.linear(h)
class STLformer(nn.Module):
 def __init__(self,num_nodes,in_steps,out_steps,steps_per_week=2016,input_dim=2,output_dim=1,input_embedding_dim=4,tow_embedding_dim=4,feed_forward_dim=16,num_heads=2,num_layers=1,dropout=0.0):
  super().__init__()
  self.num_nodes=num_nodes;self.in_steps=in_steps;self.out_steps=out_steps
  self.steps_per_week=steps_per_week;self.input_dim=input_dim;self.output_dim=output_dim
  self.input_embedding_dim=input_embedding_dim;self.tow_embedding_dim=tow_embedding_dim
  self.model_dim=input_embedding_dim+tow_embedding_dim
  self.input_proj=nn.Linear(input_dim,input_embedding_dim)
  self.tow_embedding=nn.Embedding(steps_per_week,tow_embedding_dim)
  self.MLP=nn.Linear(self.model_dim,self.model_dim)
  self.attn_layers_st=nn.ModuleList([DummyAttention(self.model_dim) for _ in range(num_layers)])
  self.output_proj=nn.Linear(in_steps*self.model_dim,out_steps*output_dim)
 def node_position_encoding(self,z):return torch.zeros_like(z)
 def time_position_encoding(self,z):return torch.zeros_like(z)
'''


def make_fake_target(root,target,n=9,train_count=24,val_count=10,test_count=6):
    rng=np.random.default_rng(43 if target=='speed' else 44)
    src=root/'fake_cache'/target/'formal';src.mkdir(parents=True,exist_ok=True)
    mean,std=(50.,10.) if target=='speed' else (200.,40.)
    for sev in (20,40,60,90):
        (root/'fake_cache'/target/f'evaluation_s{sev}').mkdir(parents=True,exist_ok=True)
    def save(split,count,d,offset):
        H=P=12
        true_x=rng.normal(size=(count,H,n)).astype('float32')
        valid=(rng.random((count,H,n))>.40)
        x=true_x*valid
        y=rng.normal(size=(count,P,n)).astype('float32')
        arrays=dict(x_obs_norm=x,y_norm=y,y_raw=y*std+mean,history_observed=valid,
                    future_valid=np.ones((count,P,n),np.bool_),direct_mask=(rng.random((count,n))>.5),
                    tod_hist=np.tile(np.arange(H,dtype=np.int16)[None],(count,1)),
                    dow_hist=np.zeros((count,H),dtype=np.int16), artificial_missing=~valid)
        if split=='train':arrays['x_full_norm']=true_x
        for key,a in arrays.items():np.save(d/f'{split}_{key}.npy',a)
        pd.DataFrame({'window_start':np.arange(count)*12+offset}).to_csv(d/f'{split}_manifest.csv',index=False)
    make1=lambda split,count,offset:save(split,count,src,offset)
    make1('train',train_count,0);make1('val',val_count,10000)
    for sev in (20,40,60,90):save('test',test_count,root/'fake_cache'/target/f'evaluation_s{sev}',20000+sev*1000)
    adj=np.zeros((n,n),'float32');dist=np.full((n,n),-1,'float32')
    for i in range(n-1):
        adj[i,i+1]=adj[i+1,i]=1.;dist[i,i+1]=dist[i+1,i]=100.
    np.save(src/'adjacency.npy',adj)
    bench=root/'data/processed/traffident/base_supportbench'/target/'TraffiDent_Net_06'
    (bench/'core').mkdir(parents=True,exist_ok=True);(bench/'graph').mkdir(exist_ok=True)
    pd.DataFrame({'local_idx':np.arange(n),'lat':np.full(n,-37.8),'lng':145+np.arange(n)*.001,
                  'road_direction':['TEST__E']*n}).to_csv(bench/'core/nodes.csv',index=False)
    np.save(bench/'graph/adjacency.npy',adj);np.save(bench/'graph/distance.npy',dist)
    # Use the same imported fake module class as the vendor file.
    vendor=root/'Backbone/vendor/STLAformer/model/STLformer.py'
    spec=importlib.util.spec_from_file_location('synthetic_vendor',vendor)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    mc={'input_dim':2,'input_embedding_dim':4,'tow_embedding_dim':4,'feed_forward_dim':16,
        'num_heads':2,'num_layers':6,'dropout':0.0,'steps_per_week':2016}
    torch.manual_seed(123)
    model=mod.STLformer(num_nodes=n,in_steps=12,out_steps=12,**mc)
    dest=root/'Backbone_exps'/target/'STLAformer'/'TraffiDent_Net_06'/'formal'/'fake_seed10'
    dest.mkdir(parents=True,exist_ok=True)
    checkpoint=dest/'best.pt'
    torch.save({'model':'STLAformer','num_nodes':n,'model_config':mc,'state_dict':model.state_dict(),
                'target':target,'dataset':'TraffiDent_Net_06','mean':mean,'std':std},checkpoint)
    return checkpoint


def main():
    tmp=Path(tempfile.mkdtemp(prefix='corsta_synthetic_'))
    vendor=tmp/'Backbone/vendor/STLAformer/model/STLformer.py'
    vendor.parent.mkdir(parents=True,exist_ok=True);vendor.write_text(VENDOR)
    common=tmp/'Backbone/backbone';common.mkdir(parents=True,exist_ok=True)
    (common/'__init__.py').write_text('')
    (common/'utils.py').write_text('def load_yaml(path):return {}\n')
    (common/'shared_cache.py').write_text(textwrap.dedent("""
        from pathlib import Path
        import os
        def cache_path(cfg,stage,severity=None):
         root=Path(os.environ['SPEED_LLM_RAG_ROOT'])
         target=os.environ['TRAFFIDENT_TARGET']
         return root/'fake_cache'/target/('formal' if stage=='formal' else 'evaluation_s'+str(severity))
        def validate_shared_cache(cfg,stage,severity=None):
         assert cache_path(cfg,stage,severity).exists()
    """))
    (tmp/'Backbone/configs').mkdir(parents=True,exist_ok=True)
    (tmp/'Backbone/configs/backbone.yaml').write_text('{}\n')
    (tmp/'Base_SupportBench').mkdir()
    target='speed'
    ck=make_fake_target(tmp,target,train_count=32,val_count=16,test_count=8)
    run=tmp/'CoRSTA_exps'/target/'incomplete'/'TraffiDent_Net_06'/'smoke'/'seed10'
    env=os.environ.copy()
    env.update(SPEED_LLM_RAG_ROOT=str(tmp),BACKBONE_MODE='incomplete',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
        PYTHONPATH=str(PROJ)+os.pathsep+str(tmp/'Backbone'))
    cmd=[sys.executable,'-u',str(PROJ/'scripts/run.py'),
         '--target',target,'--backbone-checkpoint',str(ck),
         '--run-dir',str(run),'--workers','0','--smoke','--cpu']
    logs=tmp/'synthetic.log'
    with logs.open('w') as f:
        result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
    print('INTEGRATION_LOG',logs,'ROOT',tmp)
    out=logs.read_text();print(out[-10000:])
    assert result.returncode==0, f'CoRSTA integration failed: log={logs}'
    assert (run/'summary.json').exists()
    for line in out.splitlines():
        assert line.startswith('['),f'Log without timestamp: {line!r}'
    print('CORSTA_SYNTHETIC_END_TO_END_PASS')

if __name__=='__main__':main()
