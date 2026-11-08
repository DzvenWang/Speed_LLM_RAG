"""Full-source local CPU tests for the new v9.3 data/model/training protocol."""
from __future__ import annotations
import sys,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader
from tcrr_stla.v93.model import TCRRV93,bounded
from tcrr_stla.v93.losses import response_loss,stage_loss,utility_target
from tcrr_stla.v93.interventions import enumerate_intervention
from tcrr_stla.v93.training import stage_parameters,normal_fingerprint
from tcrr_stla.v93.functional import make_functional_neighbours


def fake_student(B=2,L=12,N=6,H=12,C=8):
    m=torch.ones(B,L,N)
    m[:,-6:,1:4]=0
    x=torch.randn(B,L,N)*m
    return dict(x_obs=x,observed=m,tod=torch.zeros(B,L,dtype=torch.long),
                dow=torch.zeros(B,L,dtype=torch.long),base=torch.randn(B,H,N),
                context=torch.randn(B,N,C))


def test_no_retrieval_forward_invariant_and_gradient():
    model=TCRRV93(8,6,mode='incremental',hidden=16)
    stu=fake_student()
    y=torch.randn_like(stu['base'])
    out=model(stu)
    assert out['prediction'].shape==(2,12,6)
    assert torch.equal(model(stu,enable_cf=False)['prediction'],model(stu,enable_cf=False)['normal_prediction'])
    stage_parameters(model,'cf_joint')
    before=normal_fingerprint(model)
    loss=(out['prediction']-y).square().mean()
    loss.backward()
    assert model.cf_response_head.response.weight.grad is not None
    assert model.cf_controller.gate[-1].weight.grad is not None
    assert model.cf_controller.utility[-1].weight.grad is not None
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
    opt.step()
    assert normal_fingerprint(model)==before
    torch.testing.assert_close(model(stu,enable_cf=False)['prediction'],model(stu,enable_cf=False)['normal_prediction'],rtol=0,atol=0)


def test_reference_loss_not_masked_residual():
    model=TCRRV93(8,6,mode='decomposed',hidden=16)
    stu=fake_student();out=model(stu,enable_cf=False)
    b=dict(y=torch.randn_like(out['base']),rref=torch.zeros_like(out['base']),
           future_valid=torch.ones_like(out['base'],dtype=torch.bool))
    c={'residual_forecast':1.0}
    loss,parts=stage_loss('residual',out,b,'decomposed',model.bound,c)
    assert abs(float(loss.detach())-float(torch.nn.functional.smooth_l1_loss(out['normal_raw'],b['rref']).detach()))<1e-5


def test_cf_identity_and_pair_sign():
    y=torch.randn(2,12,6);bref=torch.randn_like(y);a=torch.randn_like(y);b=torch.randn_like(y)
    ra=y-bref;da=bref-a;db=bref-b
    torch.testing.assert_close(ra+da,y-a)
    torch.testing.assert_close(da-db,b-a)


def test_teacher_view_rotation_20_cells():
    cells=[enumerate_intervention(i,j,4) for i in range(60) for j in (1,2,3)]
    assert len(set(cells))==20
    assert enumerate_intervention(0,0,4) is None


def test_functional_validity():
    x=torch.randn(20,12,6);v=torch.ones_like(x,dtype=torch.bool)
    v[:,:,-1]=False
    ids,valid=make_functional_neighbours(x,v,k=3,min_overlap=20)
    assert ids.shape==(6,3)
    assert not valid[-1].any()


def test_response_and_utility_targets():
    model=TCRRV93(8,6,mode='incremental',hidden=16)
    stu=fake_student();out=model(stu)
    y=torch.randn_like(out['base']);v=torch.ones_like(y,dtype=torch.bool)
    b=dict(y=y,future_valid=v,cf_teacher=torch.randn_like(y),
           damage_teacher=torch.randn_like(y),query_id=torch.arange(len(y)))
    l=response_loss(out,b)
    assert all(torch.isfinite(x) for x in l.values())
    gain=utility_target(out,y,model.bound)
    torch.testing.assert_close(gain,(out['normal_prediction']-y).abs() -
       (out['base']+bounded(out['normal_raw']+out['cf_response'],model.bound)-y).abs())


def test_frozen_normal_graph_change():
    m=TCRRV93(8,6,hidden=16)
    stu=fake_student()
    m.eval()
    with torch.no_grad():a=m(stu)['normal_prediction'].clone()
    for p in m.cf_adapter.parameters():p.data.add_(10.)
    with torch.no_grad():b=m(stu)['normal_prediction']
    torch.testing.assert_close(a,b,rtol=0,atol=0)

@pytest.mark.parametrize('mode',['fixed','v7_edge_attention'])
def test_graph_modes_forward_and_gradients(mode):
    N=6;K=2
    road={}
    for side in ('upstream','downstream'):
        road[f'{side}_ids']=np.tile(np.array([[0,1]],dtype=np.int64),(N,1))
        road[f'{side}_valid']=np.ones((N,K),dtype=bool)
        road[f'{side}_gps_log']=np.ones((N,K),dtype=np.float32)*.25
        road[f'{side}_road_log']=np.ones((N,K),dtype=np.float32)*.75
        road[f'{side}_road_valid']=np.ones((N,K),dtype=bool)
    ids=torch.arange(N,dtype=torch.long)[:,None].repeat(1,2)
    ids[:,0]=(ids[:,0]+1)%N;ids[:,1]=(ids[:,1]+2)%N
    valid=torch.ones_like(ids,dtype=torch.bool)
    m=TCRRV93(8,N,hidden=16,graph_mode=mode,road_graph=road,functional=(ids,valid))
    out=m(fake_student())
    assert out['prediction'].shape==(2,12,N)
    out['cf_response'].mean().backward()
    assert m.cf_response_head.response.weight.grad is not None
