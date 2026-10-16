"""Production-shaped regression: Net-06 N=388, P=12, Top64/16/8.

This does not substitute for running the real vendor on a V100.  It tests the
full FARE graph/retrieval/transport/utility/risk computation at realistic size,
with CPU FP16 autocast reproducing the masking overflow in the Gadi failure.
"""
import torch
from fare.traffic import TrafficGraph
from fare.memory import EpisodeMemory
from fare.model import FARE
from fare.training import forecasting_loss
from conftest import DummyBackbone


def test_net06_shape_fp16_autocast_gradient():
    torch.manual_seed(10)
    n=388;h=p=12;m=80;b=2
    adj=torch.zeros(n,n,dtype=torch.float32)
    idx=torch.arange(n-1)
    adj[idx,idx+1]=1
    graph=TrafficGraph(adj,region_size=24,max_neighbors=4,max_hops=3,path_sources=16)
    model=FARE(DummyBackbone(n,64),graph,{
        'model':{'hidden_dim':64,'horizon':p,'alpha_max':1.25},
        'retrieval':{'coarse_k':64,'shortlist_k':16,'final_k':8}})
    x=torch.randn(m,h,n)
    mask=torch.ones(m,h,n,dtype=torch.bool)
    future=torch.randn(m,p,n)
    memory=EpisodeMemory(x,mask,future,torch.ones_like(future,dtype=torch.bool),
                         torch.arange(m),torch.arange(m)+h-1,future*.9)
    batch=dict(x=torch.randn(b,h,n),mask=torch.ones(b,h,n,dtype=torch.bool),
               tod=torch.zeros(b,h,dtype=torch.long),dow=torch.zeros(b,h,dtype=torch.long),
               origin=torch.full((b,),400,dtype=torch.long),row_id=torch.arange(b),
               target=torch.randn(b,p,n),future_valid=torch.ones(b,p,n,dtype=torch.bool),
               direct_mask=torch.zeros(b,n,dtype=torch.bool))
    with torch.autocast('cpu',dtype=torch.float16):
        result=model(batch,memory)
        loss,_=forecasting_loss(result,batch,{'training':{}})
    assert result['candidate_pool']['ids'].shape==(b,n,16)
    assert result['selection']['selected_ids'].shape==(b,n,8)
    assert result['prediction'].shape==(b,p,n)
    assert torch.isfinite(result['prediction']).all()
    assert torch.isfinite(loss)
    loss.backward()
    for name,parameters in model.named_parameters():
        if parameters.grad is not None:
            assert torch.isfinite(parameters.grad).all(),name
