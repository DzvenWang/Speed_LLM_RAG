import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from tcrr_stla.config import load_config
from tcrr_stla.model import TCRRSTLA
from tcrr_stla.evidence import FutureMemory,shortlist,counterfactual_residuals
from tcrr_stla.trainer import compute_losses

class STLA(nn.Module):
    model_dim=6
    def __init__(self):
        super().__init__();self.w=nn.Linear(1,6);self.out=nn.Linear(6,12)
    def set_train_phase(self,*args):pass
    def forward(self,x,mask,tod,dow):
        z=self.w((x*mask).mean(1)[...,None]);y=self.out(z).transpose(1,2)
        return y,z

def test_full_model(tmp_path):
    torch.manual_seed(1);B,H,P,N=3,12,12,7
    cfg=load_config();cfg['version']='2.0.0';cfg['experiment']['horizon']=12
    model=TCRRSTLA(STLA(),cfg)
    x=torch.randn(B,H,N);mask=(torch.rand(B,H,N)>.4).float()
    t=torch.zeros(B,H,dtype=torch.long);d=t.clone()
    cand_y=torch.randn(B,6,P,N);res=torch.randn_like(cand_y)*.2
    scores=torch.randn(B,6);cov=torch.rand(B,6);dis=torch.rand(B,6,N)
    kw=dict(candidate_y=cand_y,residual=res,candidate_scores=scores,candidate_coverage=cov,hist_distance=dis)
    pred=model(x,mask,t,d,**kw)
    assert pred['prediction'].shape==(B,P,N)
    assert torch.allclose(pred['candidate_weights'].sum(1),torch.ones(B,P,N),atol=1e-5)
    pred['prediction'].square().mean().backward()
    assert model.candidate_scorer[0].weight.grad is not None
    noc=model(x,mask,t,d,mode='no_evidence',**kw)
    assert torch.allclose(noc['prediction'],noc['base_prediction'])
    ctrl=model(x,mask,t,d,mode='no_retrieval')
    assert ctrl['prediction'].shape==(B,P,N)
    files={'train_x_full_norm.npy':np.random.randn(18,H,N).astype('f4'),
           'train_y_norm.npy':np.random.randn(18,P,N).astype('f4'),
           'train_tod_hist.npy':np.zeros((18,H),dtype='i8'),
           'train_dow_hist.npy':np.zeros((18,H),dtype='i8')}
    for k,v in files.items():np.save(tmp_path/k,v)
    store=FutureMemory(tmp_path);ind=torch.randint(0,18,(B,6))
    cand=store.fetch(ind,torch.device('cpu'))
    r=counterfactual_residuals(model.stla,cand,mask,chunk_size=4)
    assert r.shape==(B,6,P,N)
    truth=torch.randn(B,P,N)
    batch=dict(y_norm=truth,y_raw=truth,future_valid=torch.ones(B,P,N,dtype=torch.bool))
    loss,logs=compute_losses(pred,batch,0.,1.,cfg,'full')
    assert torch.isfinite(loss) and np.isfinite(logs['ranking'])
