"""Zero-vendor synthetic tests; does not claim official STLA/Gadi compatibility."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
import pytest
from torch import nn
from corsta_v2.model import CoRSTAv2
from corsta_v2.counterfactual.teacher import PairedTeacher
from corsta_v2.data.dataset import structured_mask,_rng
from corsta_v2.data.features import observable_features
from corsta_v2.training.losses import compute_loss


torch.set_num_threads(1)


class FakeSTLA(nn.Module):
    def __init__(self,n=5,h=6,p=4,d=16):
        super().__init__();self.base=nn.Module();self.base.num_nodes=n
        self.base.in_steps=h;self.base.out_steps=p;self.hidden_dim=d
        self.enc=nn.Linear(2,d);self.tail=nn.Linear(d,d)
        self.output=nn.Linear(h*d,p)
    @property
    def model_dim(self):return self.hidden_dim
    def encode_prefix(self,x,mask,tod,dow):
        z=torch.stack([x,mask],dim=-1)
        h=self.enc(z)
        return h,()
    def encode_tail(self,h,ctx):return self.tail(h)
    def forecast_head(self,h):
        B,L,N,D=h.shape
        return self.output(h.permute(0,2,1,3).reshape(B,N,L*D)).permute(0,2,1)
    def forward(self,x,mask,tod,dow):
        h,ctx=self.encode_prefix(x,mask,tod,dow)
        return self.forecast_head(self.encode_tail(h,ctx)),h.mean(1)


def graph(n=5,k=2):
    ids=np.stack([np.roll(np.arange(n),j+1) for j in range(k)],axis=-1).astype(np.int64)
    return {'ids':ids,'valid':np.ones((n,k),bool),'edge_features':np.zeros((n,k,5),np.float32)}


def make_model():
    torch.manual_seed(5)
    net=CoRSTAv2(FakeSTLA(),graph(),hidden=16,failure_dim=8,cf_dim=8,residual_dim=8,horizon=4,heads=2)
    return net


def batch(B=2):
    x=torch.randn(B,6,5)
    m=(torch.rand(B,6,5)>.4).float();x=x*m
    obs={'x':x,'mask':m,'tod':torch.zeros(B,6,dtype=torch.long),'dow':torch.zeros(B,6,dtype=torch.long)}
    full=x.clone();full[m==0]=torch.randn_like(full)[m==0]
    b=dict(obs,x_full=full,original_valid=torch.ones_like(m),
           y_norm=torch.randn(B,4,5),y_raw=torch.randn(B,4,5),future_valid=torch.ones(B,4,5,dtype=torch.bool))
    return obs,b


def test_mask_patterns_subset_and_determinism():
    source=np.ones((12,11),bool);source[1,3]=False
    for kind in ('sensor','regional','directional','temporal'):
        m=structured_mask(source,40,kind,_rng(10,1,5,40,kind))
        m2=structured_mask(source,40,kind,_rng(10,1,5,40,kind))
        assert np.array_equal(m,m2)
        assert (~m & source).sum()==round(source.sum()*.40)
        assert not np.any(m & ~source)


def test_observed_features_no_future_or_hidden_values():
    x=torch.randn(2,6,5);m=(torch.rand(2,6,5)>.4).float()
    a=observable_features(x*m,m)
    hidden=torch.randn_like(x)*900
    b=observable_features(torch.where(m.bool(),x,hidden),m)
    assert torch.allclose(a,b)


def test_cf_residual_decomposition_and_gradient():
    m=make_model();obs,b=batch()
    m.set_stage('cf',use_graph=False)
    out=m(obs,enable_residual=False,enable_controller=False)
    t=PairedTeacher(m.stla)(b,out['h0'],out['backbone'])
    assert torch.allclose(t['cf_target']+t['residual_target'],b['y_norm']-out['backbone'])
    cfg={'training':{'forecast_rmse_weight':.05,'cf_response_weight':.05,
                      'cf_latent_weight':.01,'residual_semantic_weight':.02,
                      'cf_utility_floor':.15}}
    loss,_=compute_loss(out,b,t,'cf',0.,1.,cfg)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.cf.parameters())
    assert all(p.grad is None for p in m.stla.parameters())


def test_full_and_fallback():
    model=make_model();obs,b=batch()
    model.set_stage('joint',use_graph=True)
    out=model(obs,enable_controller=False)
    assert out['prediction'].shape==(2,4,5)
    assert torch.isfinite(out['prediction']).all()
    assert torch.allclose(out['delta_cf']+out['delta_residual'],out['cf_residual_prediction']-out['backbone'],atol=1e-6)
    off=model(obs,enable_cf=False,enable_residual=False)
    assert torch.allclose(off['prediction'],off['backbone'])
    assert torch.allclose(out['prediction'],out['cf_residual_prediction'])
    out['prediction'].square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.residual.parameters())
    assert all(p.grad is None for p in model.stla.parameters())


def test_no_future_data_leakage_in_model_forward():
    m=make_model();obs,b=batch();m.eval()
    with torch.no_grad():
        before=m(obs,enable_controller=True)['prediction']
        obs2={**obs,'x_full':b['x_full']}
        with pytest.raises(ValueError):m(obs2)
        hidden=b['x_full'].clone();hidden[obs['mask']==0]*=99
        after=m(obs,enable_controller=True)['prediction']
    assert torch.equal(before,after)


def test_graph_null_option_when_no_valid_neighbors():
    g=graph();g['valid'][:]=False
    model=CoRSTAv2(FakeSTLA(),g,hidden=16,failure_dim=8,cf_dim=8,residual_dim=8,horizon=4,heads=2)
    obs,_=batch()
    result=model(obs,enable_graph=True)
    assert torch.isfinite(result['prediction']).all()
    assert (result['cf']['spatial']['null_weight']>.999).all()


def test_stage_trainability():
    model=make_model()
    for s,names in [('cf',('cf','failure','input_adapter')),
                    ('residual',('residual',)),('control',('controller',)),
                    ('joint',('cf','residual','graph_cf','graph_residual'))]:
        trainable=model.set_stage(s)
        assert len(trainable)>0
        assert all(not p.requires_grad for p in model.stla.parameters())


def test_controller_initially_exact_joint_identity():
    m=make_model();obs,_=batch()
    m.set_stage('control',use_graph=False)
    out=m(obs,enable_controller=True)
    assert torch.allclose(out['prediction'],out['cf_residual_prediction'],atol=1e-7)
    assert torch.all(out['gate_residual']==1)


def test_target_nan_outside_valid_is_harmless():
    from corsta_v2.util import masked_mean
    x=torch.tensor([[1.0,float('nan')],[3.,float('inf')]])
    mask=torch.tensor([[True,False],[True,False]])
    assert masked_mean(x,mask)==2.0


def test_road_graph_orientation_and_sparse_edge_validity(tmp_path):
    import pandas as pd
    from corsta_v2.spatial.graph import build_road
    n=4
    graphdir=tmp_path/'data/processed/traffident/base_supportbench/speed/TraffiDent_Net_06'
    (graphdir/'core').mkdir(parents=True)
    (graphdir/'graph').mkdir()
    pd.DataFrame({'local_idx':np.arange(n),'lat':np.full(n,-37.8),
                  'lng':145+np.arange(n)*.001,'road_direction':['X__E']*n}).to_csv(graphdir/'core/nodes.csv',index=False)
    adjacency=np.zeros((n,n),'float32');dist=np.full((n,n),-1,'float32')
    for i in range(n-1):adjacency[i,i+1]=adjacency[i+1,i]=1;dist[i,i+1]=dist[i+1,i]=100
    np.save(graphdir/'graph/adjacency.npy',adjacency);np.save(graphdir/'graph/distance.npy',dist)
    g=build_road(tmp_path,'speed','TraffiDent_Net_06',k_per_direction=1)
    assert g['ids'].shape==(n,2)
    assert not g['valid'][0,0] # no upstream for first eastbound
    assert g['valid'][0,1] and g['ids'][0,1]==1
    assert g['valid'][2,0] and g['ids'][2,0]==1
    assert g['valid'][2,1] and g['ids'][2,1]==3
