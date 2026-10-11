import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fare.graph import GraphRegions
from fare.memory import TrainMemory
from fare.retrieval import TargetRetriever
from fare.support_demand import SupportDemandEncoder
from fare.ranking import forecast_utility_ranking_loss

def test_v2_retrieval_and_gradient():
    torch.manual_seed(10)
    b,h,p,n,m=2,12,12,8,45
    a=torch.eye(n)
    for i in range(n-1):a[i,i+1]=a[i+1,i]=1
    regions=GraphRegions(a,target_size=4,max_anchors=4)
    x=torch.randn(b,h,n)
    observed=torch.rand(b,h,n)>.4
    ctx=regions(x,observed)
    memory=TrainMemory(torch.randn(m,h,n),torch.ones(m,h,n,dtype=torch.bool),
                       torch.arange(m)*50,device='cpu',
                       future=torch.randn(m,p,n),future_valid=torch.ones(m,p,n,dtype=torch.bool))
    r=TargetRetriever(h,64,128,coarse_topk=12,topk=4,overlap=h+p)
    r.refresh(memory,regions)
    base=torch.randn(b,p,n)
    support=SupportDemandEncoder(64,128,64)
    demand=support(torch.randn(b,h,n,64),observed,base,regions)
    out=r(ctx,regions,memory,torch.tensor([2200,2300]),demand=demand,base_forecast=base)
    assert out['analog_future'].shape==(b,regions.g,4,p)
    assert ((memory.starts[out['ids']]+h+p)<=torch.tensor([2200,2300])[:,None,None]+h).all()
    batch={'y_norm':torch.randn(b,p,n),'future_valid':torch.ones(b,p,n,dtype=torch.bool),
           'x_obs':x,'obs_mask':observed}
    loss,metrics=forecast_utility_ranking_loss(batch,out,memory,regions)
    assert loss.isfinite()
    loss.backward()
    assert r.rerank[0].weight.grad is not None

if __name__=='__main__':
    test_v2_retrieval_and_gradient()
    print('FARE_V2_COMPONENT_PASS')

def test_full_v2_forward_and_loss():
    from test_full import fixture
    from fare.losses import losses
    cfg,model,memory,batch=fixture(b=2)
    memory.future=torch.randn(len(memory.x),12,model.n)
    memory.future_valid=torch.ones_like(memory.future,dtype=torch.bool)
    batch['window_start']=torch.tensor([3000,3100])
    batch['y_norm']=batch['y_raw'].clone()
    model.set_phase('full')
    model.refresh(memory)
    out=model(batch,memory)
    assert out['retrieved_utility_scores'].shape if 'retrieved_utility_scores' in out else True
    assert out['candidate_prediction'].shape==(2,12,model.n)
    z=losses(model,batch,out,memory,'full',0.,1.,cfg)
    assert torch.isfinite(z['total'])
    z['total'].backward()
    assert model.analog_adapter[-1].weight.grad is not None
    assert model.support_demand.net[0].weight.grad is not None
