import torch
from tcrr_stla.config import load_config
from tcrr_stla.model import TCRRSTLA
from test_tcrr_v2 import STLA


def test_full_traffic_utility_is_node_horizon_specific():
    torch.manual_seed(77)
    B,H,P,N,K=2,12,12,7,4
    model=TCRRSTLA(STLA(),load_config())
    x=torch.randn(B,H,N);mask=(torch.rand(B,H,N)>.65).float()
    mask[0,:,1]=0
    cand=torch.randn(B,K,H,N)
    y=torch.randn(B,K,P,N)
    residual=.1*torch.randn_like(y)
    normal=.05*torch.randn_like(y)
    t=torch.zeros(B,H,dtype=torch.long)
    out=model(x,mask,t,t,candidate_y=y,residual=residual,normal_residual=normal,
       candidate_histories=cand,candidate_scores=torch.rand(B,K),
       candidate_coverage=torch.rand(B,K),hist_distance=torch.rand(B,K,N),
       candidate_valid=torch.ones(B,K,P,N,dtype=torch.bool))
    assert out['candidate_expected_gain'].shape==(B,K,P,N)
    assert out['mixture_expected_gain'].shape==(B,P,N)
    assert torch.isfinite(out['prediction']).all()
    out['prediction'].mean().backward()
    assert model.traffic.candidate[-1].weight.grad is not None
    assert torch.isfinite(model.traffic.candidate[-1].weight.grad).all()
