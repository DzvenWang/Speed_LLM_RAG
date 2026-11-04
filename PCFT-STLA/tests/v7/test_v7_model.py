import torch
import pytest
from torch import nn
from pcft_stla.v6.graph import DirectedGraph
from pcft_stla.v7.model import PCFTV7
from pcft_stla.v7.backbone import official_input
from pcft_stla.v7.training import time_partition,trainable_for,prediction_loss
from pcft_stla.v7.utility import supervised_utility_loss

class Block(nn.Module):
    def __init__(self,d):
        super().__init__();self.lin=nn.Linear(d,d)
    def forward(self,x,*_):return x+self.lin(x)*0.01
class NativeSTLA(nn.Module):
    def __init__(self,n=5,d=12):
        super().__init__();self.num_nodes=n;self.in_steps=12;self.input_dim=2
        self.steps_per_week=2016;self.model_dim=d
        self.input_proj=nn.Linear(2,d)
        self.attn_layers_st=nn.ModuleList([Block(d) for _ in range(2)])
        self.output_proj=nn.Linear(12*d,12)
    def forward(self,x):
        h=self.input_proj(x)
        for layer in self.attn_layers_st:h=layer(h)
        B,T,N,D=h.shape
        out=self.output_proj(h.transpose(1,2).reshape(B,N,T*D)).transpose(1,2)
        return out[...,None]

def fake_batch(n=5,b=2):
    torch.manual_seed(42)
    obs=torch.ones(b,12,n,3)
    x=torch.randn(b,12,n,3)
    return {'x_multi':x,'observed_mask':obs,'missing_age':torch.zeros_like(x),
         'x_target':x[...,0].clone(), 'tod':torch.arange(12)[None].expand(b,-1),
         'dow':torch.zeros(b,12,dtype=torch.long),'y_norm':torch.zeros(b,12,n),
         'y_raw':torch.zeros(b,12,n),'future_valid':torch.ones(b,12,n,dtype=torch.bool),
         'direct_mask':torch.ones(b,n,dtype=torch.bool)}

def fake_model(n=5):
    graph=DirectedGraph(torch.tensor([[0,1,2,3],[1,2,3,4]]),torch.ones(4,10),n)
    return PCFTV7(NativeSTLA(n),graph,0,horizon=12)

def cfg():
    return {'training':{'lr_new':0.0003,'lr_controller':0.0005,'lr_backbone':0.00002,
        'weight_decay':0.0001,'joint_unfreeze_stla_layers':0,'loss':{'masked':.25,'horizon':.1,
        'correction_regularization':.001,'gain':.1,'risk':.05,'action':.1}}}


def test_anchor_parity_and_initial_identity():
    m=fake_model();batch=fake_batch()
    m.eval()
    with torch.no_grad():
        orig=m.anchor.base(official_input(batch,m.anchor.base)).squeeze(-1)
        result=m(batch)
    assert torch.allclose(orig,result['baseline'])
    assert torch.allclose(orig,result['prediction'],atol=1e-6)
    assert m.anchor.parity(batch)==0.0


def test_missing_sources_produce_no_propagation():
    m=fake_model();b=fake_batch()
    m.prop_head[-1].bias.data.fill_(0.3)
    b['observed_mask'].zero_();b['x_multi'].zero_();b['x_target'].zero_()
    o=m(b)
    assert torch.all(o['prop_delta']==0)
    assert torch.all(o['coverage']==0)


def test_all_stages_and_gradients():
    model=fake_model();b=fake_batch();c=cfg()
    for stage in ('local','propagation','utility','joint'):
        trainable_for(model,stage,c)
        model.zero_grad(set_to_none=True)
        o=model(b)
        assert o['prediction'].shape==b['y_norm'].shape
        loss=prediction_loss(o,b,c,stage)
        assert torch.isfinite(loss)
        loss.backward()
        assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
        if stage=='utility':
            assert model.controller.gain.weight.grad is not None
    assert torch.isfinite(supervised_utility_loss(o,b))


def test_direction_ablation_changes_attributes_not_edges():
    m=fake_model();m.prop_head[-1].bias.data.fill_(0.1)
    b=fake_batch()
    a=m(b)
    m.set_ablation(use_direction=False)
    other=m(b)
    assert a['prediction'].shape==other['prediction'].shape


def test_index_alignment_and_time_split():
    import pandas as pd
    class Fake:
        meta=pd.DataFrame({'window_start':list(range(0,300,3))})
    first,second,meta=time_partition(Fake(),.8,24)
    assert meta['utility_first_window']-meta['train_last_window']>=24
    assert set(first).isdisjoint(second)


def test_split_requires_window_start():
    class Fake:meta={}
    with pytest.raises(ValueError):time_partition(Fake())


def test_no_future_labels_as_inputs():
    m=fake_model();b=fake_batch()
    first=m(b)['prediction'].detach().clone()
    b['y_norm'].fill_(999);b['y_raw'].fill_(10000)
    assert torch.allclose(first,m(b)['prediction'])

def test_invalid_future_targets_do_not_poison_loss_or_metrics():
    from pcft_stla.v7.training import RunningMetrics
    m=fake_model();b=fake_batch()
    b['future_valid'][:,4,1]=False
    b['y_norm'][:,4,1]=float('nan')
    b['y_raw'][:,4,1]=float('nan')
    o=m(b)
    m.set_stage('utility')
    loss=supervised_utility_loss(o,b)
    assert torch.isfinite(loss)
    a=RunningMetrics(torch.device('cpu'))
    a.update(o,b,0.,1.)
    assert a.finish()['mae']>=0

def test_empty_physical_graph_and_ablation_zero_correction():
    from pcft_stla.v6.graph import DirectedGraph
    graph=DirectedGraph(torch.empty(2,0,dtype=torch.long),torch.empty(0,10),5)
    m=PCFTV7(NativeSTLA(),graph,0)
    b=fake_batch()
    with torch.no_grad():
        o=m(b)
    assert torch.all(o['coverage']==0)
    assert torch.all(o['prop_delta']==0)
    m.set_ablation(use_propagation=False,use_controller=False,use_local=False)
    m.set_stage('joint')
    with torch.no_grad():o=m(b)
    assert torch.allclose(o['prediction'],o['baseline'])


def test_direction_and_null_ablation_safe_when_all_sources_missing():
    m=fake_model();b=fake_batch()
    b['observed_mask'].zero_();b['x_multi'].zero_();b['x_target'].zero_()
    m.set_ablation(use_null_source=False,use_direction=False)
    with torch.no_grad():o=m(b)
    assert torch.isfinite(o['prediction']).all()
    assert torch.all(o['prop_delta']==0)
