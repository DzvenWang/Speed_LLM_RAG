import torch
from pcft_stla.propagation_v2 import PropagationSupportV2
from pcft_stla.action_v2 import ForecastActionController


def test_temporal_coherence_protects_cross_region_edges():
    nbr=torch.tensor([[1],[0]])
    p=PropagationSupportV2(2,nbr,lags=(1,))
    w=torch.ones(3,2,1,1)
    p.install(w,w)
    B,C,H,P,N=1,1,12,12,2
    x=torch.zeros(B,C,H,N)
    future=torch.zeros(B,C,P,N)
    future[...,0]=torch.arange(P)
    future[...,1]=-torch.arange(P)
    q=torch.zeros(B,H,N);mask=torch.ones_like(q)
    non_coherent=torch.tensor([[[11,22]]],dtype=torch.long)
    transfer=p.transfer(x,future,P,q,mask,provenance=non_coherent)
    assert torch.allclose(transfer,torch.zeros_like(transfer))
    same=torch.tensor([[[11,11]]],dtype=torch.long)
    transfer2=p.transfer(x,future,P,q,mask,provenance=same)
    assert transfer2.abs().sum()>0


def test_full_horizon_support_allows_model_estimate_and_action_training():
    torch.manual_seed(33)
    neighbors=torch.tensor([[1],[0]])
    p=PropagationSupportV2(2,neighbors,lags=(1,))
    p.install(torch.ones(3,2,1,1),torch.ones(3,2,1,1))
    mask=torch.zeros(1,12,2)
    base=torch.randn(1,12,2)
    field=p.support_field(mask,12,query_x=torch.zeros_like(mask),base_forecast=base)
    assert field['observed'][:,7:].sum()==0
    assert field['forecast'][:,7:].sum()>0
    ctrl=ForecastActionController(12)
    result=ctrl(base,torch.randn_like(base)*.1,torch.randn_like(base)*.1,field)
    assert result['action_weights'].shape==(1,12,2,3)
    assert torch.allclose(result['action_weights'].sum(-1),torch.ones_like(base),atol=1e-5)
    result['correction'].mean().backward()
    assert ctrl.predictor[-1].weight.grad is not None
