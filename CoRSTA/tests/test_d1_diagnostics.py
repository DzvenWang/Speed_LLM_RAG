"""D1 unit/integration tests without Gadi or official vendor files."""
from __future__ import annotations
import copy,types
import torch
from torch.utils.data import Dataset
from test_corsta import build
from diagnostics.metrics import MaskedMetric,query_oracle,element_oracle
from diagnostics.interventions import intervene,MODES,FusionCapture
from diagnostics.candidate import CandidateAudit
from diagnostics.gradients import one_batch
from diagnostics.run_d1 import selection_pass,parameter_drift


def made():
    model,_,b=build(n=5,B=2,K=3)
    model.eval()
    with torch.no_grad():
        p,h=model.student(b['x'],b['mask'],b['tod'],b['dow'])
    b.update({'cached_base':p.detach().clone(),'cached_latent':h.detach().clone(),
              'future_valid':torch.ones_like(p,dtype=torch.bool),
              'y_norm':p.detach().clone()+torch.randn_like(p)*.1,
              'y_raw':(p.detach().clone()+torch.randn_like(p)*.1)*10+50,
              'index':torch.tensor([0,1]),'window_start':torch.tensor([101,201]),
              'direct_mask':torch.ones(2,5,dtype=torch.bool)})
    return model,b


def test_weighted_mae_and_oracles():
    y=torch.tensor([[[0.,1.]],[[5.,1.]]]);v=torch.tensor([[[1,1]],[[1,0]]],dtype=torch.bool)
    a=y+1.;b=y-1.;c=y
    m=MaskedMetric();m.add(a,y,v)
    assert m.count==3 and abs(m.to_dict()['mae']-1)<1e-8
    q,_=query_oracle([a,b,c],y,v);e,_=element_oracle([a,b,c],y)
    assert torch.equal(q,c) and torch.equal(e,c)


def test_interventions_hooks_removed_and_fusion_details():
    m,b=made()
    with torch.no_grad():
        full=m(b)['refined']
        with intervene(m,'no_feedback'):disabled=m(b)['refined']
        plain=m(b)['refined']
        assert torch.allclose(plain,full,atol=1e-5)
        assert torch.allclose(disabled,m(b)['backbone'],atol=1e-5)
        for mode in MODES:
            with intervene(m,mode):z=m(b)['refined']
            assert z.isfinite().all()
        with FusionCapture(m) as cap:
            m(b);stats=cap.get()
        assert stats['injection_c1'].shape==(2,)
        assert stats['graph_mix'].shape==(2,12,5,3)
        assert all(torch.isfinite(x).all() for x in (stats['injection_c1'],stats['injection_c2']))


def test_candidate_audit_finite():
    model,b=made();legacy=model.c1
    saved={}
    h=legacy.transport.register_forward_hook(lambda mod,ins,out:saved.setdefault('tr',out))
    try:
        with torch.no_grad():
            out=legacy.forward_progressive(b['cached_base'],b['cached_latent'],b['x'],b['mask'],
                b['candidate_normal'],b['candidate_cf'],candidate_scores=b['candidate_scores'],
                candidate_coverage=b['candidate_coverage'],candidate_valid=b['candidate_valid'],
                candidate_support_raw=b['candidate_support_raw'],max_stage=1)
    finally:h.remove()
    audit=CandidateAudit(.5);audit.add(b,out,saved['tr'],50.,10.)
    result=audit.to_dict();assert result['queries']==2
    assert result['mae_by_mode']['candidate_oracle_query']['mae'] is not None
    assert result['mae_by_mode']['transport_normal_plus_cf']['mae'] is not None


def test_gradient_diagnostic_and_parameter_drift():
    model,b=made()
    model.student.set_train_phase('partial',2);model.c1.freeze_stage('c1')
    model.eval()
    report=one_batch(model,b,50.,10.)
    assert 'stla_tail' in report and 'c1' in report
    assert report['stla_tail']['refined_grad_norm']>0
    assert report['bridge_fusion']['refined_grad_norm']>0
    joint=copy.deepcopy(model)
    with torch.no_grad():next(joint.student.base.output_proj.parameters()).add_(.01)
    delta=parameter_drift(model,joint)
    assert delta['student_output']['relative_l2_drift']>0
    assert delta['student_prefix']['relative_l2_drift']==0


class FakeDataset(Dataset):
    def __init__(self,b):self.b=b
    def __len__(self):return len(self.b['x'])
    def __getitem__(self,i):return {k:v[i] for k,v in self.b.items()}


def test_selection_full_one_shot_synthetic(tmp_path):
    model,b=made();legacy=copy.deepcopy(model.c1);joint=copy.deepcopy(model)
    a=types.SimpleNamespace(workers=0)
    selection=selection_pass(a,model,joint,legacy,{'mean':50.,'std':10.},
        FakeDataset(b),[0,1],torch.device('cpu'),2,tmp_path,.5)
    assert selection['selection_queries']==2
    assert selection['metrics']['oracle_element']['mae']<=selection['metrics']['bootstrap_c1']['mae']+1e-6
    assert selection['metrics']['oracle_query']['mae']<=selection['metrics']['bootstrap_c1']['mae']+1e-6
    assert selection['no_feedback_backbone_equivalence_max_abs_norm']<1e-5
    assert (tmp_path/'selection_query_metrics.csv').is_file()
    assert len((tmp_path/'selection_query_metrics.csv').read_text().splitlines())==3


def test_cpu_batch_policy_and_cli(monkeypatch):
    import sys
    from diagnostics.run_d1 import choose_batch,cli
    model,b=made()
    size,attempts=choose_batch(model,FakeDataset(b),torch.device('cpu'),64)
    assert size==16 and attempts[0]['reason'].startswith('CPU')
    monkeypatch.setattr(sys,'argv',['run_d1.py','--target','speed','--backbone-checkpoint','/tmp/best.pt','--device','cpu'])
    a=cli()
    assert a.device=='cpu' and a.grad_queries==128
