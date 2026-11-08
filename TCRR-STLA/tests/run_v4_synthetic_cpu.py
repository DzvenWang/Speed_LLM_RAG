import torch,numpy as np,pandas as pd,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tcrr_stla.v4.trainer as vt
from tcrr_stla.v4.trainer import fit_shared_adapter,train_v4
from tcrr_stla.config import file_sha256
from torch import nn
class TinySTLA(nn.Module):
 def __init__(self,N=4,P=3):
  super().__init__()
  self.base=nn.Module();self.base.model_dim=12;self.base.num_nodes=N;self.base.out_steps=P;self.base.w=nn.Parameter(torch.tensor(1.))
  self.mask_proj=nn.Sequential(nn.Linear(1,12),nn.GELU(),nn.Linear(12,12))
  self.model_dim=12
 def forward(self,x,mask,tod,dow):
  base=x[:,-1:,:].expand(-1,self.base.out_steps,-1)*self.base.w + self.mask_proj(mask.mean(1,keepdim=False)[...,None]).mean(-1)[:,None]
  z=self.mask_proj(mask.mean(1)[...,None])
  return base,z
with tempfile.TemporaryDirectory() as td:
 root=Path(td);c=root/'data';c.mkdir();rng=np.random.default_rng(1)
 H,P,N=3,3,4;cnt=8
 for sp in ['train','val']:
  x=rng.normal(size=(cnt,H,N)).astype('float32');obs=(rng.random((cnt,H,N))>.2);art=(rng.random((cnt,H,N))>.5)&(~obs);full=np.where(obs|art,x,0).astype('float32');xobs=np.where(obs,x,0)
  y=rng.normal(size=(cnt,P,N)).astype('float32');t=np.zeros((cnt,H),np.int64);dow=t.copy()
  for k,v in dict(x_obs_norm=xobs,x_full_norm=full,history_observed=obs,artificial_missing=art,y_norm=y,y_raw=y,future_valid=np.ones_like(y,bool),direct_mask=np.ones((cnt,N),bool),tod_hist=t,dow_hist=dow).items():np.save(c/f'{sp}_{k}.npy',v)
  pd.DataFrame({'window_start':np.arange(cnt)*100}).to_csv(c/f'{sp}_manifest.csv',index=False)
 cfg=dict(version='4.0.0',experiment=dict(target='speed',dataset='TraffiDent_Net_06',history=H,horizon=P,seed=10),
 project=dict(root=str(root),retrieval_root=str(root/'r'),exp_root=str(root/'e')),
 model=dict(latent_dim=8,fusion_hidden=16,max_normalized_correction=3.),
 runtime=dict(amp=False,num_workers=0,pin_memory=False,replay_bank_dtype='float32',replay_query_batch_size=3,normal_teacher_on_gpu=False,replay_bank_on_gpu=False,print_every=2),
 retrieval=dict(shortlist_per_period=1,replay_batch_size=5),
 traffic=dict(graph_neighbors=2),
 training=dict(batch_size=3,max_epochs=80,lr_new=.001,weight_decay=.0001,grad_clip=5.,ablation='full',rmse_weight=.05,min_delta=1e-8,patience=10),
 v4=dict(s0_epochs=1,s0_lr=.001,stages=dict(transfer=40,controller=10,joint=30)))
 ck=root/'backbone.pt';ck.write_bytes(b'fake')
 def fake_initial(cfg,checkpoint,device,adapter_path=None):
  m=TinySTLA(N,P).to(device)
  if adapter_path:
   z=torch.load(adapter_path,map_location='cpu',weights_only=False);m.mask_proj.load_state_dict(z['mask_proj'])
  return m,dict(mean=0.,std=1.,num_nodes=N,model_config={})
 vt.initial_model=fake_initial
 vt.cache_path=lambda *a,**k:c
 adapter=fit_shared_adapter(cfg,ck,c,root/'ad.pt',torch.device('cpu'),smoke=True)
 cfg['runtime']['teacher_adapter_checkpoint']=str(adapter)
 rt=root/'rtrain.npz';rv=root/'rval.npz'
 ids=np.tile(np.array([2,3,4],np.int32),(cnt,1));sc=np.random.default_rng(1).normal(size=ids.shape).astype('float32')
 for p in [rt,rv]:np.savez_compressed(p,version=np.int32(25),indices=ids,scores=sc,coverage=np.ones_like(sc))
 result=train_v4(cfg,c,ck,rt,rv,root/'run',adapter,smoke=True)
 print('INTEGRATION_PASS',result.exists())
