import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np,pandas as pd,torch
from torch import nn
from mask_rast_stla.config import load_config
from mask_rast_stla.model import MaskRASTSTLA
from mask_rast_stla.data import FullTrainRepository,SupportSplit
from mask_rast_stla.memory import build_memory_index
from mask_rast_stla.graph import graph_neighbours
from mask_rast_stla.trainer import losses

class STLA(nn.Module):
    model_dim=6
    def __init__(self):super().__init__();self.w=nn.Linear(1,6);self.out=nn.Linear(6,12)
    def set_train_phase(self,*args):pass
    def forward(self,x,mask,tod,dow,support=None,reliability=None):
        z=self.w((x*mask).mean(1)[...,None]);return self.out(z).transpose(1,2),z,None

def test_full_model(tmp_path):
    torch.manual_seed(3);B,H,P,N=2,12,12,9;M=60
    cfg=load_config();cfg['retrieval']['top_k']=4;cfg['retrieval']['graph_neighbors']=3
    np.save(tmp_path/'train_x_full_norm.npy',np.random.randn(M,H,N).astype('f4'))
    np.save(tmp_path/'train_y_norm.npy',np.random.randn(M,P,N).astype('f4'))
    np.save(tmp_path/'train_tod_hist.npy',np.zeros((M,H),dtype='i8'))
    np.save(tmp_path/'train_dow_hist.npy',np.zeros((M,H),dtype='i8'))
    pd.DataFrame({'window_start':np.arange(M)*30}).to_csv(tmp_path/'train_manifest.csv',index=False)
    neighbours=graph_neighbours(tmp_path,tmp_path/'graph.npz',neighbours=3)
    model=MaskRASTSTLA(STLA(),N,H,P,cfg);model.set_graph(neighbours)
    repo=FullTrainRepository(tmp_path);index=build_memory_index(model.memory_encoder,repo,'cpu',16)
    b=dict(x=torch.randn(B,H,N),mask=(torch.rand(B,H,N)>.4).float(),
           tod=torch.zeros(B,H,dtype=torch.long),dow=torch.zeros(B,H,dtype=torch.long),
           window_start=torch.as_tensor([3000,3300],dtype=torch.long))
    out=model(b,index,repo)
    assert out['prediction'].shape==(B,P,N)
    assert out['router'].shape==(B,P,N,4)
    assert torch.allclose(out['router'].sum(-1),torch.ones(B,P,N),atol=1e-5)
    out['prediction'].square().mean().backward()
    assert model.scorer[0].weight.grad is not None
    abst=model(b,index,repo,mode='no_future')
    assert torch.allclose(abst['prediction'],abst['base_prediction'])
    ctrl=model(b,index,repo,mode='no_retrieval')
    assert ctrl['prediction'].shape==(B,P,N)
    b.update(y_norm=torch.randn(B,P,N),future_valid=torch.ones(B,P,N,dtype=torch.bool))
    b['y_raw']=b['y_norm'].clone()
    loss,vals=losses(out,b,0.,1.,cfg,'full')
    assert torch.isfinite(loss) and np.isfinite(vals['ranking'])
    # Evaluator uses no x_full; it must not be required in the test split.
    for name,val in [('x_obs_norm',np.random.randn(B,H,N).astype('f4')),
                      ('y_norm',np.random.randn(B,P,N).astype('f4')),
                      ('y_raw',np.random.randn(B,P,N).astype('f4')),
                      ('history_observed',np.ones((B,H,N),dtype='f4')),
                      ('future_valid',np.ones((B,P,N),dtype='bool')),
                      ('direct_mask',np.ones((B,N),dtype='bool')),
                      ('tod_hist',np.zeros((B,H),dtype='i8')),
                      ('dow_hist',np.zeros((B,H),dtype='i8'))]:np.save(tmp_path/f'test_{name}.npy',val)
    pd.DataFrame({'window_start':[9999,10010]}).to_csv(tmp_path/'test_manifest.csv',index=False)
    assert 'x_full' not in SupportSplit(tmp_path,'test')[0]


def test_region_aggregation_amp_forward_backward(tmp_path):
    """Regression: regional index_add_ must support FP16/BF16 AMP input."""
    import torch
    from mask_rast_stla.encoders import StateEncoder
    torch.manual_seed(29)
    B,H,N=2,12,9
    encoder=StateEncoder(N,H,graph_k=3)
    x=torch.randn(B,H,N)
    mask=(torch.rand(B,H,N)>0.35).float()
    tod=torch.zeros(B,H,dtype=torch.long)
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        out=encoder(x,mask,tod,tod)
        loss=out['region_keys'].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert encoder.region_proj.weight.grad is not None


def test_full_model_amp_cpu_bf16(tmp_path):
    """Regional evidence path stays functional under autocast."""
    torch.manual_seed(19)
    B,H,P,N,M=2,12,12,9,32
    cfg=load_config()
    cfg['retrieval']['top_k']=2
    cfg['retrieval']['regional_top_k']=2
    cfg['retrieval']['graph_neighbors']=3
    np.save(tmp_path/'train_x_full_norm.npy',np.random.randn(M,H,N).astype('f4'))
    np.save(tmp_path/'train_y_norm.npy',np.random.randn(M,P,N).astype('f4'))
    np.save(tmp_path/'train_tod_hist.npy',np.zeros((M,H),dtype='i8'))
    np.save(tmp_path/'train_dow_hist.npy',np.zeros((M,H),dtype='i8'))
    pd.DataFrame({'window_start':np.arange(M)*30}).to_csv(tmp_path/'train_manifest.csv',index=False)
    neighbours=graph_neighbours(tmp_path,tmp_path/'graph.npz',neighbours=3)
    model=MaskRASTSTLA(STLA(),N,H,P,cfg)
    model.set_graph(neighbours)
    repo=FullTrainRepository(tmp_path)
    index=build_memory_index(model.memory_encoder,repo,'cpu',8)
    b=dict(x=torch.randn(B,H,N),mask=(torch.rand(B,H,N)>.4).float(),
        tod=torch.zeros(B,H,dtype=torch.long),dow=torch.zeros(B,H,dtype=torch.long),
        window_start=torch.tensor([1500,1600],dtype=torch.long))
    with torch.autocast(device_type='cpu',dtype=torch.bfloat16):
        out=model(b,index,repo)
        loss=out['prediction'].float().square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert model.scorer[0].weight.grad is not None
