import torch
from pcft_stla.v3.future import SupportAnchoredFuture
from pcft_stla.v3.coherent import EpisodeCoherentPropagation
from pcft_stla.v3.action import GainHarmEvolutionPolicy
from pcft_stla.v3.losses import pcft_v3_losses
from pcft_stla.propagation_v2 import PropagationSupportV2


def test_future_anchor_and_lossless_none():
    torch.manual_seed(42);b,k,p,n=2,3,12,7
    residual=torch.randn(b,k,p,n,requires_grad=True)
    mask=(torch.rand(b,12,n)>.5).float()
    f=SupportAnchoredFuture(p)
    out=f(residual,mask,torch.rand(b,n))
    assert torch.allclose(out,residual)
    out.sum().backward()
    assert f.scale[-1].weight.grad is not None


def test_coherent_episode_transport_and_independent_ablations():
    torch.manual_seed(22)
    b,g,r,k,h,p,n=2,3,2,2,12,12,8
    neighbours=torch.stack([torch.arange(n).roll(1),torch.arange(n).roll(2)],1)
    propagation=PropagationSupportV2(n,neighbours,lags=(1,2,3))
    w=torch.ones(3,n,2,3)*.1
    propagation.install(w,w)
    q=torch.randn(b,h,n);mask=(torch.rand(b,h,n)>.3).float()
    field=propagation.support_field(mask,p,q,torch.randn(b,p,n))
    weights=torch.softmax(torch.randn(b,g+k,p,n),1)
    g_x=torch.randn(b,g,h,n);g_y=torch.randn(b,g,p,n)
    r_x=torch.randn(b,r,k,h,n);r_y=torch.randn(b,r,k,p,n)
    tr=EpisodeCoherentPropagation(p)
    args=(propagation,g_x,g_y,r_x,r_y,q,mask,weights,torch.arange(n)%r,torch.zeros(b,r,k,dtype=torch.long),field)
    full=tr(*args)
    nograph=tr(*args,mode='no_directional_graph')
    nolag=tr(*args,mode='no_lag_alignment')
    assert full['propagation_correction'].shape==(b,p,n)
    assert torch.isfinite(full['propagation_correction']).all()
    assert not torch.allclose(full['raw_propagation'],nolag['raw_propagation'])
    # All sources belong to their retrieved complete case by construction.
    assert full['episode_count']==g+r*k
    (full['propagation_correction'].square().mean()).backward()
    assert tr.scale[-1].weight.grad is not None


def test_utility_changes_actual_action_and_has_train_only_labels():
    torch.manual_seed(31)
    b,p,n=2,12,7
    x=torch.randn(b,p,n)
    field={'observed':torch.rand(b,p,n),'forecast':torch.rand(b,p,n),'uncertainty':torch.rand(b,p,n)}
    model=GainHarmEvolutionPolicy(p)
    with torch.no_grad():
        model.estimator[-1].bias.copy_(torch.tensor([1.5,0.6,-1.5,2.0,0.,0.]))
    out=model(x,x*.1,x*.25,field)
    no_util=model(x,x*.1,x*.25,field,mode='no_action_utility')
    no_risk=model(x,x*.1,x*.25,field,mode='no_risk')
    assert not torch.allclose(out['correction'],no_util['correction'])
    assert not torch.allclose(out['correction'],no_risk['correction'])
    record={'prediction':x+out['correction'],'base_prediction':x,
            'legacy_action':out['legacy_correction'],'propagation_action':out['propagation_correction'],
            'action_expected_gain':out['action_expected_gain'],'action_expected_harm':out['action_expected_harm'],
            'action_logits':out['action_logits'],'propagation_available':torch.ones_like(x),
            'v3_contribution':'M3'}
    batch={'y_norm':torch.randn_like(x),'future_valid':torch.ones_like(x,dtype=torch.bool)}
    loss,_=pcft_v3_losses(record,batch,{'v3':{}})
    assert torch.isfinite(loss)
    loss.backward()
    assert model.estimator[0].weight.grad is not None


def test_full_pcft_m0_m3_keep_identical_candidate_budget(tmp_path):
    """End-to-end forward against a *real* train-only memory repository/index."""
    import numpy as np,pandas as pd
    from torch import nn
    from pcft_stla.config import load_config
    from pcft_stla.model import PCFTSTLA
    from pcft_stla.data import FullTrainRepository
    from pcft_stla.memory import build_memory_index

    class DummySTLA(nn.Module):
        model_dim=6
        def __init__(self):
            super().__init__();self.proj=nn.Linear(1,6);self.forecast=nn.Linear(6,12)
        def set_train_phase(self,*args):pass
        def forward(self,x,mask,tod,dow,support=None,reliability=None):
            latent=self.proj((x*mask).mean(1)[...,None]);return self.forecast(latent).transpose(1,2),latent,None

    torch.manual_seed(24);b,h,p,n,m=2,12,12,9,35
    rng=np.random.default_rng(101)
    for key,arr in {'train_x_full_norm.npy':rng.normal(size=(m,h,n)).astype('f4'),
                    'train_y_norm.npy':rng.normal(size=(m,p,n)).astype('f4'),
                    'train_tod_hist.npy':np.zeros((m,h),dtype='i8'),
                    'train_dow_hist.npy':np.zeros((m,h),dtype='i8')}.items():np.save(tmp_path/key,arr)
    pd.DataFrame({'window_start':np.arange(m)*30}).to_csv(tmp_path/'train_manifest.csv',index=False)
    repo=FullTrainRepository(tmp_path)
    batch={'x':torch.randn(b,h,n),'mask':(torch.rand(b,h,n)>.2).float(),
           'tod':torch.zeros(b,h,dtype=torch.long),
           'dow':torch.zeros(b,h,dtype=torch.long),
           'window_start':torch.tensor([3000,3600])}
    ids=[]
    for level in ('M0','M1','M2','M3'):
        cfg=load_config();cfg['v3']['contribution']=level
        cfg['retrieval']['top_k']=2;cfg['retrieval']['regional_top_k']=2
        cfg['retrieval']['graph_neighbors']=2
        model=PCFTSTLA(DummySTLA(),n,h,p,cfg)
        nei=torch.stack([torch.arange(n).roll(1),torch.arange(n).roll(2)],-1)
        model.set_graph(nei)
        graph_w=torch.ones_like(model.propagation.weights)*.2
        model.propagation.install(graph_w,graph_w)
        index=build_memory_index(model.memory_encoder,repo,'cpu',8,propagation=model.propagation)
        out=model(batch,index,repo)
        assert out['action_weights'] is not None
        assert out['candidate_weights'].shape[1]==3*2+2+1
        ids.append(out['candidate_weights'].shape[1])
        if level in ('M2','M3'):
            assert out['raw_coherent_propagation'] is not None
            assert out['episode_count']>2*2
            out['prediction'].mean().backward()
            assert model.v3_episode.scale[-1].weight.grad is not None
        else:
            assert out['raw_coherent_propagation'] is None
    assert len(set(ids))==1
