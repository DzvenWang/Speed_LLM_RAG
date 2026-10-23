"""v3 mechanism-level and small integrated tests; never use test-future features."""
import torch
from torch import nn
from tcrr_stla.v3.transfer import CounterfactualResidualTransfer
from tcrr_stla.v3.losses import mechanism_losses
from tcrr_stla.model import TCRRSTLA
from tcrr_stla.config import load_config


class FakeSTLA(nn.Module):
    model_dim=8
    def __init__(self):
        super().__init__();self.enc=nn.Linear(1,8);self.predict=nn.Linear(8,12)
    def set_train_phase(self,*args):pass
    def forward(self,x,mask,tod,dow):
        z=self.enc((x*mask).mean(1)[...,None]);return self.predict(z).transpose(1,2),z


def case(level):
    cfg=load_config();cfg['v3']['contribution']=level
    torch.manual_seed(7)
    model=TCRRSTLA(FakeSTLA(),cfg)
    b,k,h,p,n=2,4,12,12,6
    x=torch.randn(b,h,n);mask=(torch.rand(b,h,n)>.3).float()
    norm=torch.randn(b,k,p,n)*.1;fail=torch.randn_like(norm)*.2
    inp=dict(candidate_y=torch.randn(b,k,p,n),normal_residual=norm,
             residual=norm+fail,candidate_scores=torch.randn(b,k),
             candidate_coverage=torch.rand(b,k),candidate_valid=torch.ones(b,k,p,n,dtype=torch.bool),
             candidate_histories=torch.randn(b,k,h,n))
    t=torch.zeros(b,h,dtype=torch.long)
    return model,x,mask,t,inp


def test_contribution_forward_backward_and_no_test_future():
    for level in ('M0','M1','M2','M3'):
        model,x,mask,t,inp=case(level)
        out=model(x,mask,t,t,**inp)
        assert out['prediction'].shape==(2,12,6)
        assert out['candidate_logits'].shape==(2,5,12,6)
        assert torch.isfinite(out['prediction']).all()
        no=model(x,mask,t,t,**inp,mode='no_evidence')
        assert torch.allclose(no['prediction'],no['base_prediction'],atol=1e-6)
        out['prediction'].mean().backward()
        assert model.v3.reference_risk[0].weight.grad is not None
        if level in ('M2','M3'):
            assert model.v3.transfer_ranker[-1].weight.grad is not None
        if level=='M3':
            assert model.v3.gain_harm[-1].weight.grad is not None


def test_counterfactual_pairing_and_mode_isolation():
    model,x,mask,t,inp=case('M3')
    full=model(x,mask,t,t,**inp)
    pair=model(x,mask,t,t,**inp,mode='shuffled_cf_pairing')
    assert not torch.allclose(full['failure_residual'],pair['failure_residual'])
    nof=model(x,mask,t,t,**inp,mode='no_failure_residual')
    no_replay=model(x,mask,t,t,**inp,mode='no_mask_replay')
    assert torch.allclose(no_replay['failure_residual'], torch.zeros_like(no_replay['failure_residual']))
    assert not torch.allclose(full['cf_response'],nof['cf_response'])
    no_gain=model(x,mask,t,t,**inp,mode='no_gain_harm')
    assert not torch.allclose(full['prediction'],no_gain['prediction']) or torch.allclose(model.v3.gain_harm[-1].weight,torch.zeros_like(model.v3.gain_harm[-1].weight))


def test_v3_auxiliary_label_stays_out_of_forward():
    model,x,mask,t,inp=case('M3')
    pred=model(x,mask,t,t,**inp)
    batch={'future_valid':torch.ones_like(pred['prediction'],dtype=torch.bool),'y_norm':torch.randn_like(pred['prediction'])}
    loss,stats=mechanism_losses(pred,batch,cf_target=torch.randn_like(pred['prediction']))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.v3.component_scale[0].weight.grad is not None


def test_p0_candidate_audit_interface_is_preserved():
    from tcrr_stla.p0 import candidate_probe_batch
    model,x,mask,t,inp=case('M3')
    with torch.no_grad():
        out=model(x,mask,t,t,**inp)
        probe, bins=candidate_probe_batch(out,{
            'y_norm':torch.randn_like(out['prediction']),
            'future_valid':torch.ones_like(out['prediction'],dtype=torch.bool),
        },max_delta=model.max_delta)
    assert len(probe['risk_prob_mean'])==x.shape[0] and len(bins)==10
    assert len(probe['best_single_candidate_gain_norm'])==x.shape[0]
