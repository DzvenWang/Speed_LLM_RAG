"""Executable synthetic unit tests (no Gadi/data/vendor required)."""
from __future__ import annotations
import torch
from torch import nn
from corsta.student import SplitSTLA
from corsta.stla import MaskAwareSTLAformer
from corsta.deep import CoRSTA
from corsta.v7.model import SupportAwareTCRR
from corsta.batching import CHOICES,smaller_size

class FakeAttention(nn.Module):
    def __init__(self,d):super().__init__();self.w=nn.Linear(d,d)
    def forward(self,h,node,time):return h+.01*self.w(h)

class FakeSTLA(nn.Module):
    def __init__(self,n=5,d=8):
        super().__init__()
        self.num_nodes=n;self.in_steps=self.out_steps=12;self.model_dim=d
        self.steps_per_week=2016;self.input_dim=2;self.output_dim=1
        self.tow_embedding_dim=4
        self.input_proj=nn.Linear(2,d-4)
        self.tow_embedding=nn.Embedding(2016,4)
        self.MLP=nn.Linear(d,d)
        self.attn_layers_st=nn.ModuleList([FakeAttention(d) for _ in range(6)])
        self.output_proj=nn.Linear(12*d,12)
    def node_position_encoding(self,z):return torch.zeros_like(z)
    def time_position_encoding(self,z):return torch.zeros_like(z)


def build(n=5,B=2,K=3):
    base=FakeSTLA(n)
    original=MaskAwareSTLAformer(base)
    student=SplitSTLA(base,2)
    student.mask_proj.load_state_dict(original.mask_proj.state_dict())
    graph={'functional':torch.arange(n)[:,None].repeat(1,2),
           'functional_valid':torch.zeros(n,2,dtype=torch.bool),
           'road':{}}
    for direction in ('upstream','downstream'):
        for name in ('ids','valid','gps_log','road_log','road_valid'):
            value=torch.zeros(n,2,dtype=torch.long if name=='ids' else
                torch.bool if name in ('valid','road_valid') else torch.float)
            graph['road'][f'{direction}_{name}']=value
    legacy=SupportAwareTCRR(model_dim=8,horizon=12,history=12,
                neighbours=graph['functional'],functional_valid=graph['functional_valid'],
                road_graph=graph['road'],hidden=16,top_k=K)
    model=CoRSTA(student,legacy,graph,token_dim=8,graph_dim=8,heads=2)
    mask=(torch.rand(B,12,n)>.2).float()
    b={'x':torch.randn(B,12,n)*mask,'mask':mask,
       'tod':torch.randint(0,288,(B,12)),'dow':torch.zeros(B,12,dtype=torch.long),
       'candidate_normal':torch.randn(B,K,12,n)*.2,
       'candidate_cf':torch.randn(B,K,12,n)*.1,
       'candidate_scores':torch.randn(B,K),'candidate_coverage':torch.rand(B,K),
       'candidate_valid':torch.ones(B,K,12,n,dtype=torch.bool),
       'candidate_support_raw':torch.rand(B,K,n,14)}
    return model,original,b


def test_split_stla_equivalence():
    torch.manual_seed(0)
    model,old,b=build()
    old.eval();model.student.eval()
    with torch.no_grad():
        a,_=old(b['x'],b['mask'],b['tod'],b['dow'])
        z,_=model.student(b['x'],b['mask'],b['tod'],b['dow'])
    assert torch.allclose(a,z,atol=1e-6)


def test_end_to_end_gradient_and_mask_invariance():
    torch.manual_seed(1)
    model,_,b=build()
    model.student.set_train_phase('partial',2)
    model.train()
    o=model(b)
    assert o['refined'].shape==(2,12,5)
    loss=o['refined'].square().mean()+o['c1_posthoc'].square().mean()
    loss.backward()
    for prefix in ('student.base.attn_layers_st.5','c1.transport',
                   'support.normal','c2.current','bridge1','bridge2','fusion'):
        total=sum(float(p.grad.abs().sum()) for name,p in model.named_parameters()
                  if name.startswith(prefix) and p.grad is not None)
        assert total>0, f'No gradients in {prefix}'
    model.eval()
    bb=dict(b)
    bb['x']=torch.where(b['mask'].bool(),b['x'],torch.full_like(b['x'],123456.))
    with torch.no_grad():
        left=model(b)['refined'];right=model(bb)['refined']
    assert torch.allclose(left,right,atol=1e-5)


def test_empty_neighbours_and_batch_choices():
    model,_,b=build()
    assert model(b)['refined'].isfinite().all()
    assert CHOICES==(16,32,64,128)
    assert smaller_size(128)==64
    assert smaller_size(64)==32
    assert smaller_size(32)==16
    assert smaller_size(16) is None


def test_horizon_dependent_graph_values_and_empty_nodes():
    from corsta.deep import HorizonGraphMessage
    torch.manual_seed(11)
    graph=HorizonGraphMessage(8)
    B,P,N=2,4,5
    z=torch.randn(B,P,N,8,requires_grad=True)
    ids=torch.tensor([[1,2],[0,2],[0,1],[2,4],[1,3]],dtype=torch.long)
    valid=torch.tensor([[True,True],[True,False],[True,True],[False,False],[True,True]])
    obs=torch.rand(B,P,N)
    sup=torch.rand(B,P,N)
    out=graph(z,ids,valid,obs,sup)
    assert out.shape==(B,P,N,8)
    assert torch.allclose(out[:,:,3,:],torch.zeros_like(out[:,:,3,:]))
    assert not torch.allclose(out[:,0,0],out[:,1,0])
    out.square().mean().backward()
    assert z.grad is not None and z.grad.abs().sum()>0
    assert graph.v.weight.grad is not None and graph.v.weight.grad.abs().sum()>0


def test_no_legal_c1_candidates_remains_finite():
    model,_,b=build()
    b['candidate_valid'].zero_()
    out=model(b)
    assert out['refined'].isfinite().all()
    assert out['reliability_c1'].isfinite().all()
    assert out['reliability_c1'].abs().max()<1e-5


def test_auto_batch_probe_order(monkeypatch):
    from corsta import batching
    class Device: type='cuda'
    calls=[]
    outcomes={64:False,32:True,16:True,128:True}
    def fake_probe(*a,**kw):
        size=a[2];calls.append(size)
        return {'size':size,'ok':outcomes[size],
                'peak_gib':5,'gpu_gib':32,'reason':'pass' if outcomes[size] else 'OOM'}
    monkeypatch.setattr(batching,'probe',fake_probe)
    chosen,_=batching.select_batch(object(),object(),Device(),initial=64,logger=lambda *_:None)
    assert chosen==32 and calls==[64,32]
    calls.clear();outcomes[64]=True;outcomes[128]=False
    chosen,_=batching.select_batch(object(),object(),Device(),initial=64,logger=lambda *_:None)
    assert chosen==64 and calls==[64,128]
    calls.clear();outcomes[128]=True
    chosen,_=batching.select_batch(object(),object(),Device(),initial=64,logger=lambda *_:None)
    assert chosen==128 and calls==[64,128]
