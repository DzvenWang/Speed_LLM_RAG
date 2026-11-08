import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pytest
from torch import nn
from tcrr_stla.v5.model import TCRRSTLAV5,MODES
from tcrr_stla.v5.support import PairSupport
from tcrr_stla.v5.counterfactual import TemporalResponseEncoder
from tcrr_stla.v5.losses import compute_losses

class FakeBackbone(nn.Module):
    model_dim=24
    def __init__(self,n=7,p=12):
        super().__init__();self.sentinel=nn.Parameter(torch.tensor(1.))
        self.n=n;self.p=p
    def forward(self,x,mask,tod,dow):
        return x.new_zeros((x.shape[0],self.p,self.n)),x.new_zeros((x.shape[0],self.n,self.model_dim))

def data(n=7,p=12,k=4,b=2,h=12):
    torch.manual_seed(40)
    x=torch.randn(b,h,n)
    m=(torch.rand(b,h,n)>.35).float()
    cand=torch.randn(b,k,h,n)
    valid=torch.rand(b,k,h,n)>.1
    normal=torch.randn(b,k,p,n)*.05
    return dict(x=x,mask=m,tod=torch.zeros(b,h,dtype=torch.long),dow=torch.zeros(b,h,dtype=torch.long),
       candidate_y=torch.randn(b,k,p,n),normal_residual=normal,
       residual=normal+torch.randn_like(normal)*.08,
       candidate_scores=torch.randn(b,k)*.1,candidate_coverage=torch.rand(b,k),
       candidate_valid=torch.ones(b,k,p,n,dtype=torch.bool),
       candidate_histories=cand,candidate_source_valid=valid.float())

def model(n=7,p=12):
    cfg={'model':{'latent_dim':16,'fusion_hidden':32,'max_normalized_correction':3.},
         'experiment':{'horizon':p},'v5':{'fixed_gate':.75}}
    m=TCRRSTLAV5(FakeBackbone(n,p),cfg)
    return m

@pytest.mark.parametrize('n,k',[(7,4),(13,2)])
def test_four_modes_backward(n,k):
    m=model(n);d=data(n=n,k=k)
    for mode in MODES:
        out=m(**d,mode=mode)
        assert out['prediction'].shape==(2,12,n)
        assert out['candidate_weights'].shape==(2,k+1,12,n)
        assert torch.allclose(out['candidate_weights'].sum(1),torch.ones_like(out['prediction']),atol=1e-6)
        assert torch.isfinite(out['prediction']).all()
        assert torch.isfinite(out['candidate_loo_proposal']).all()
        assert torch.allclose(out['gate'],torch.full_like(out['gate'],.75))
        if mode in ('M00','M01'):
            assert torch.count_nonzero(out['cf_increment'])==0
        if mode in ('M00','M10'):
            assert torch.count_nonzero(out['support_logits_delta'])==0
        target=torch.randn_like(out['prediction'])
        batch={'y_norm':target,'y_raw':target,'future_valid':torch.ones_like(target,dtype=torch.bool)}
        loss,_=compute_losses(out,batch,0,1,{'v5':{},'training':{}},'c12')
        assert torch.isfinite(loss)
        m.zero_grad(set_to_none=True);loss.backward()
        assert any(p.grad is not None for p in m.c2.parameters())


def test_cf_zero_response_is_zero_increment():
    m=model();d=data();d['residual']=d['normal_residual'].clone()
    out=m(**d,mode='M11')
    assert torch.count_nonzero(out['cf_increment'])==0


def test_all_invalid_candidates_back_to_base():
    m=model();d=data();d['candidate_valid'].fill_(False)
    for mode in MODES:
        out=m(**d,mode=mode)
        assert torch.allclose(out['prediction'],out['base_prediction'],atol=1e-6)
        assert (out['none_weight']>.999).all()


def test_masked_values_never_influence_pair_agreement():
    torch.manual_seed(1)
    B,K,H,N=2,3,12,9
    qx=torch.randn(B,H,N); qm=(torch.rand(B,H,N)>.5).float()
    cx=torch.randn(B,K,H,N); sm=(torch.rand(B,K,H,N)>.3).float()
    p=PairSupport()
    a=p(qx,qm,cx,sm)['raw']
    qx2=qx+1000*(1-qm)
    cx2=cx+500*(1-sm)
    b=p(qx2,qm,cx2,sm)['raw']
    assert torch.allclose(a,b,atol=1e-5)


def test_no_overlap_does_not_report_perfect_match():
    p=PairSupport();q=torch.randn(1,12,5)
    out=p(q,torch.ones_like(q),torch.randn(1,2,12,5),torch.zeros(1,2,12,5))
    assert not out['has_overlap'].any()
    assert not out['raw'][...,6].any() # observed agreement


def test_controller_uses_no_labels():
    m=model();d=data();out=m(**d,mode='M11',controller_mode='learned')
    assert torch.isfinite(out['prediction']).all()
    assert out['gate'].min()>=0 and out['gate'].max()<=1


def test_counterfactual_time_encoder_shape():
    net=TemporalResponseEncoder()
    inp=torch.randn(1,2,12,7)
    assert net(inp,inp).shape==inp.shape
