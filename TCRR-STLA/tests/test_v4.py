"""v4 synthetic contract tests: this suite needs no STLA vendor checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from tcrr_stla.v4.transfer import CounterfactualResidualTransferV4
from tcrr_stla.v4.losses import compute_v4_losses,valid_cvar
from tcrr_stla.v4.model import TCRRSTLAV4
from tcrr_stla.v4.trainer import stage_plan

class FakeSTLA(torch.nn.Module):
    model_dim=12
    def __init__(self):
        super().__init__();self.base=torch.nn.Linear(1,1)
        self.mask_proj=torch.nn.Linear(1,1)
    def forward(self,x,m,t,d):
        return x[:,-1:,:].expand(-1,4,-1)+self.base.weight.mean()*0,m[:,0,:,None].expand(-1,-1,12)

def make(B=2,K=3,P=4,N=5):
    x=torch.randn(B,6,N);mask=(torch.rand(B,6,N)>.3).float();base=torch.randn(B,P,N)
    normal=torch.randn(B,K,P,N)*0.4
    failure=torch.randn(B,K,P,N)*0.2
    score=torch.randn(B,K)*0.2;coverage=torch.rand(B,K)
    valid=torch.ones(B,K,P,N,dtype=torch.bool)
    history=torch.randn(B,K,6,N)
    source=torch.ones(B,K,6,N)
    return dict(x=x,mask=mask,base=base,normal=normal,failure=failure,
                score=score,coverage=coverage,valid=valid,history=history,source=source)

def test_all_missing_candidates_returns_base_exactly():
    b=make();tr=CounterfactualResidualTransferV4(4,8)
    out=tr(b['x'],b['mask'],b['base'],torch.randn(2,5,8),b['normal'],b['normal']+b['failure'],
           b['normal'],b['score'],b['coverage'],torch.zeros_like(b['valid']),
           candidate_histories=b['history'],source_valid=b['source'])
    assert torch.equal(out['prediction'],b['base'])
    assert out['candidate_weights'][:,:3].sum()==0

def test_utility_head_changes_ranking_and_receives_gradient():
    b=make();tr=CounterfactualResidualTransferV4(4,8,utility_weight=3.0)
    ctx=torch.randn(2,5,8)
    out=tr(b['x'],b['mask'],b['base'],ctx,b['normal'],b['normal']+b['failure'],
           b['normal'],b['score'],b['coverage'],b['valid'],candidate_histories=b['history'],source_valid=b['source'])
    out['prediction'].mean().backward()
    grad=tr.c12.gain_harm.weight.grad
    assert grad is not None and float(grad.abs().sum())>0
    before=out['candidate_weights'].detach().clone()
    with torch.no_grad():tr.c12.gain_harm.bias[0]=3.0
    after=tr(b['x'],b['mask'],b['base'],ctx,b['normal'],b['normal']+b['failure'],
           b['normal'],b['score'],b['coverage'],b['valid'],candidate_histories=b['history'],source_valid=b['source'])['candidate_weights']
    assert not torch.allclose(before,after)

def test_stage_activation_and_full_loss():
    b=make();tr=CounterfactualResidualTransferV4(4,8)
    tr.set_stage('transfer');assert any(p.requires_grad for p in tr.c12.parameters());assert all(not p.requires_grad for p in tr.c3.parameters())
    tr.set_stage('controller');assert all(not p.requires_grad for p in tr.c12.parameters());assert any(p.requires_grad for p in tr.c3.parameters())
    tr.set_stage('joint');assert all(p.requires_grad for p in tr.parameters())
    out=tr(b['x'],b['mask'],b['base'],torch.randn(2,5,8),b['normal'],b['normal']+b['failure'],
           b['normal'],b['score'],b['coverage'],b['valid'],candidate_histories=b['history'],source_valid=b['source'])
    out['hard_shuffled_candidate']=None;out['hard_shuffled_gain_pred']=None
    y=b['base']+torch.randn_like(b['base'])*0.3
    batch=dict(y_norm=y,y_raw=y,future_valid=torch.ones_like(y,dtype=torch.bool))
    cfg=dict(training=dict(rmse_weight=.05),v4=dict())
    loss,stats=compute_v4_losses(out,batch,0,1,cfg,'joint')
    assert torch.isfinite(loss).item();assert 'control' in stats and 'tail' in stats

def test_cvar_top_tail():
    harm=torch.tensor([0.,1.,2.,8.,10.]);valid=torch.ones(5,dtype=torch.bool)
    assert abs(float(valid_cvar(harm,valid,.4))-9.0)<1e-6

def test_stages_cover_all_80_epochs():
    cfg={'training':{'max_epochs':80},'v4':{'stages':{'transfer':40,'controller':10,'joint':30}}}
    assert sum(n for _,n in stage_plan(cfg,False))==80
    assert [n for _,n in stage_plan(cfg,True)]==[1,1,1]

def test_source_valid_requires_intersection_and_shapes():
    from tcrr_stla.evidence import counterfactual_residuals
    class Teacher(torch.nn.Module):
        def forward(self,x,m,t,d):return (x*m).sum(1)[:,None].expand(-1,4,-1),None
    B,K,H,N,P=2,3,6,5,4
    x=torch.randn(B,K,H,N);source=torch.ones_like(x);source[:,:,:,:2]=0
    cand={'x':x,'y':torch.zeros(B,K,P,N),'tod':torch.zeros(B,K,H,dtype=torch.long),
          'dow':torch.zeros(B,K,H,dtype=torch.long),'source_valid':source}
    qmask=torch.ones(B,H,N)
    r=counterfactual_residuals(Teacher(),cand,qmask,strict_source_mask=True)
    assert torch.equal(r[...,:2],torch.zeros_like(r[...,:2]))

def test_model_frozen_base_with_precomputed_context():
    cfg={'model':{'latent_dim':8,'fusion_hidden':16,'max_normalized_correction':3.},
         'experiment':{'horizon':4},'v4':{}}
    m=TCRRSTLAV4(FakeSTLA(),cfg)
    m.set_stage('joint')
    assert all(not p.requires_grad for p in m.stla.parameters())
    b=make();out=m(b['x'],b['mask'],torch.zeros(2,6,dtype=torch.long),torch.zeros(2,6,dtype=torch.long),
       candidate_y=b['normal'],residual=b['normal']+b['failure'],normal_residual=b['normal'],
       candidate_scores=b['score'],candidate_coverage=b['coverage'],candidate_valid=b['valid'],
       candidate_histories=b['history'],candidate_source_valid=b['source'],
       base_prediction=b['base'],frozen_context=torch.randn(2,5,12))
    assert out['prediction'].shape==b['base'].shape
    assert out['hard_shuffled_gain_pred'] is not None
