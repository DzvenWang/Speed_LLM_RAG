from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from corsta.v7.support import pair_support_features,mask_temporal_features,FEATURE_NAMES
from corsta.v7.model import SupportAwareTCRR,make_neighbours
from corsta.v7.replay import signature,_candidate_select

def road(N,K=2,empty=False):
    v={}
    for direction in ('upstream','downstream'):
        ids=torch.arange(N)[:,None].expand(N,K).clone()
        valid=torch.zeros((N,K),dtype=torch.bool)
        if not empty:
            ids[:,0]=(torch.arange(N)+ (1 if direction=='upstream' else -1))%N
            valid[:,0]=True
        v[f'{direction}_ids']=ids
        v[f'{direction}_valid']=valid
        for k in ('gps_log','road_log'):v[f'{direction}_{k}']=torch.zeros(N,K)
        v[f'{direction}_road_valid']=valid.clone()
    return v

def inputs(B=2,H=4,P=3,N=9,K=2,dim=8):
    torch.manual_seed(9)
    x=torch.randn(B,H,N)
    mask=(torch.rand(B,H,N)>.45).float()
    mask[:, -1, 0]=0
    cand=torch.randn(B,K,H,N)
    cv=torch.ones_like(cand)
    raw=pair_support_features(x,mask,cand,cv)
    return dict(base=torch.randn(B,P,N),latent=torch.randn(B,N,dim),x=x,mask=mask,
        normal=torch.randn(B,K,P,N)*.1,cf=torch.randn(B,K,P,N)*.1,
        candidate_scores=torch.randn(B,K),candidate_coverage=torch.ones(B,K),
        candidate_valid=torch.ones(B,K,P,N,dtype=torch.bool),candidate_support_raw=raw)

def test_support_pairing_preserves_invariance():
    B,H,N,K=2,6,8,3
    torch.manual_seed(0)
    x=torch.randn(B,H,N);m=(torch.rand(B,H,N)>.6).float()
    sx=torch.randn(B,K,H,N);sm=(torch.rand(B,K,H,N)>.2).float()
    a=pair_support_features(x,m,sx,sm)
    x_changed=x+1000*(1-m)
    sx_changed=sx+1000*(1-sm)
    b=pair_support_features(x_changed,m,sx_changed,sm)
    assert a.shape==(B,K,N,len(FEATURE_NAMES))
    torch.testing.assert_close(a,b,atol=1e-5,rtol=0)
    assert torch.isfinite(a).all()

def test_support_empty_observations_and_temporal_features():
    x=torch.full((2,12,9),1000.)
    m=torch.zeros_like(x)
    a=pair_support_features(x,m,x[:,None].expand(-1,2,-1,-1),torch.ones(2,2,12,9))
    assert torch.isfinite(a).all()
    assert a[...,6].eq(0).all() and a[...,7].eq(0).all()
    t=mask_temporal_features(m)
    assert t.shape==(2,9,5) and t[...,2].eq(1).all()

def test_mask_aware_functional_graph():
    x=torch.randn(4,12,10)
    valid=torch.ones_like(x);valid[:,:,2]=0
    ids,v=make_neighbours(x,3,valid,min_overlap=8)
    assert ids.shape==(10,3) and v.shape==ids.shape
    assert not v[2].any()

def test_v7_c1_c2_forward_backward():
    N,K=9,2
    model=SupportAwareTCRR(8,3,4,torch.arange(N)[:,None].expand(N,2),
        road(N),hidden=16,top_k=K)
    inp=inputs(N=N,K=K)
    out=model.forward_progressive(**inp,max_stage=2)
    for key in ('normal','c1','c12','delta1','delta2','support_reliability'):
        assert torch.isfinite(out[key]).all(),key
    assert out['c12'].shape==(2,3,N)
    torch.testing.assert_close(out['c1'],out['c12'])  # zero-init C2 is safe
    model.freeze_stage('c1')
    loss=model.forward_progressive(**inp,max_stage=1)['c1'].square().mean()
    loss.backward()
    assert any(p.grad is not None for p in model.pair_projection.parameters())
    model.zero_grad(set_to_none=True)
    model.freeze_stage('c2')
    loss=model.forward_progressive(**inp,max_stage=2)['c12'].square().mean()
    loss.backward()
    assert model.c2_delta.weight.grad is not None

def test_v7_no_neighbor_and_all_missing():
    N,K=9,2
    model=SupportAwareTCRR(8,3,4,torch.arange(N)[:,None],road(N,empty=True),
        hidden=16,top_k=K)
    inp=inputs(N=N,K=K)
    inp['mask'].zero_();inp['x']=torch.randn_like(inp['x'])*1e5
    inp['candidate_support_raw']=pair_support_features(inp['x'],inp['mask'],
        torch.zeros(2,K,4,N),torch.ones(2,K,4,N))
    inp['candidate_valid'].zero_()
    out=model.forward_progressive(**inp,max_stage=2)
    assert torch.isfinite(out['c12']).all()
    assert out['candidate_weights'][:,-1].min()>0.999

def test_backbone_target_only_masked_input_invariance():
    N,K=9,2
    model=SupportAwareTCRR(8,3,4,torch.arange(N)[:,None],road(N,empty=True),
        hidden=16,top_k=K)
    inp=inputs(N=N,K=K)
    out=model.forward_progressive(**inp)['c12']
    mod={**inp,'x':inp['x']+500*(1-inp['mask'])}
    new=model.forward_progressive(**mod)['c12']
    torch.testing.assert_close(out,new,atol=1e-5,rtol=1e-6)

def test_multiscale_retrieval_no_overlap_and_no_query_future():
    device=torch.device('cpu');T=30;H=4;N=9;P=3
    trainx=torch.randn(T,H,N);valid=torch.ones_like(trainx)
    ty=torch.randn(T,P,N);fy=torch.ones_like(ty)
    ks=[signature(trainx,valid,period=p) for p in (4,2,1)]
    qx=trainx[10:12].clone();qm=valid[10:12].clone()
    st=torch.arange(T)*12;qs=st[10:12]
    pred=torch.randn(2,P,N)
    ids,scores=_candidate_select(qx,qm,qs,ks,st,4,9,(4,2,1),
                                   trainx.numpy(),valid.numpy(),ty.numpy(),fy.numpy(),pred)
    assert ids.shape==(2,4)
    assert (ids>=0).all() and torch.isfinite(scores).all()
    for b in range(2):
        assert len(ids[b].unique())==4
        for row in ids[b]:
            assert not (st[int(row)]<qs[b]+H+P and qs[b]<st[int(row)]+H+P)

def test_parameter_structure_has_no_auxiliary_target_or_c3():
    N=9
    model=SupportAwareTCRR(8,3,4,torch.arange(N)[:,None],road(N),hidden=16,top_k=2)
    assert not hasattr(model,'c3_response')
    assert 'flow' not in model.forward_progressive.__code__.co_varnames
    assert 'y' not in model.forward_progressive.__code__.co_varnames

def test_c1_initial_equals_normal_and_auxiliary_backprop():
    from corsta.v7.losses import cfrm_auxiliary
    N,K=9,2
    model=SupportAwareTCRR(8,3,4,torch.arange(N)[:,None],road(N),hidden=16,top_k=K)
    inp=inputs(N=N,K=K)
    out=model.forward_progressive(**inp,max_stage=1)
    torch.testing.assert_close(out['c1'],out['normal'])
    batch={'y_norm':torch.randn(2,3,N),'future_valid':torch.ones(2,3,N)}
    aux=cfrm_auxiliary(out,batch,0.,1.,wrong_pair=('cf_wrong_increment' in out))
    assert torch.isfinite(aux)
    model.freeze_stage('c1')
    (out['c1'].square().mean()+aux).backward()
    assert model.c1_activation.grad is not None

def test_chunk_shuffle_each_query_exactly_once():
    from corsta.v7.sampling import ChunkShuffleBatchSampler
    sampler=ChunkShuffleBatchSampler(71,chunk_size=16,batch_size=24)
    ids=[int(x) for batch in sampler for x in batch]
    assert len(ids)==71 and sorted(ids)==list(range(71))
