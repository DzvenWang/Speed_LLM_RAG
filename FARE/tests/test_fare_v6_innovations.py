"""Method-level regression and end-to-end CPU smoke for FARE v6."""
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from fare.forecast_utility import utility_loss,candidate_ranking_loss
from fare.utility_teacher import generate_utility_teacher,replay_utility_loss
from fare.innovation_audit import audit,paired_block_stats
from fare.training import fit


class TinySet(Dataset):
    def __init__(self,b,n=6):self.b=b;self.n=n
    def __len__(self):return 4
    def __getitem__(self,i):
        row={k:(v[i%2].clone() if k!='row_id' else torch.tensor(i))
             for k,v in self.b.items()}
        row['x_full']=row['x'].clone()
        row['raw_valid']=row['mask'].clone()
        row['target_raw']=row['target'].clone()
        return row


def test_strict_support_and_single_candidate(fixture_data):
    m,bank,b=fixture_data
    with torch.no_grad():
        m.support_delta[-1].bias.fill_(.2)
        full=m(b,bank)
        no=m(b,bank,intervention='no_evidence')
        support=m(b,bank,intervention='support_only')
        assert torch.equal(no['prediction'],support['prediction'])
        assert torch.equal(no['prediction'],full['support_prediction'])
        assert not torch.allclose(no['prediction'],no['base_prediction'])
        forced=m.candidate_single(b,full['prepared'],torch.zeros(2,6,dtype=torch.long))
        assert torch.isfinite(forced['prediction']).all()
        good=forced['selection']['selected_valid'].any((2,3))
        assert torch.all(forced['selection']['null'][good]==0)
        assert torch.all(forced['alpha'][good]==1)
        assert torch.allclose(forced['prediction'],forced['candidate_prediction'],atol=1e-6)
        no_history=dict(b,origin=torch.zeros_like(b['origin']))
        empty=m(no_history,bank)
        assert torch.equal(empty['prediction'],empty['support_prediction'])
        assert torch.all(empty['selection']['null']==1)


def test_oof_sensing_decomposition_and_ablation(fixture_data):
    m,bank,b=fixture_data
    bank.failure_delta.fill_(.02)
    bank.failure_valid.fill_(True)
    assert torch.allclose(bank.residual_complete+bank.failure_delta,bank.residual)
    full=m(b,bank)
    ordinary=m(b,bank,intervention='ordinary_residual')
    assert torch.isfinite(full['prediction']).all()
    assert torch.isfinite(ordinary['prediction']).all()
    assert not torch.allclose(full['proposals']['deltas'],ordinary['proposals']['deltas'])


def test_stable_gain_loss_extreme_labels(fixture_data):
    m,bank,b=fixture_data
    r=m(b,bank)['rank']
    p={k:v for k,v in r.items()}
    p['sigma']=torch.full_like(r['sigma'],1e-12)
    lab=torch.full_like(r['gain'],200.)
    valid=r['valid'][...,None].expand_as(lab)
    total,parts=utility_loss(p,lab,valid,return_parts=True)
    assert torch.isfinite(total)
    assert all(torch.isfinite(z) for z in parts.values())
    total.backward()
    assert torch.isfinite(m.ranker.gain.weight.grad).all()


def test_teacher_supervises_real_recall_and_changes_queries(fixture_data):
    m,bank,b=fixture_data;ds=TinySet(b)
    bank.failure_delta.fill_(.03)
    records=generate_utility_teacher(m,bank,ds,torch.device('cpu'),
                                     limit=4,batch_size=2,candidates=3,nodes_per_query=4)
    m.zero_grad(set_to_none=True)
    loss,stats=replay_utility_loss(m,records,torch.device('cpu'),max_replays=4,memory=bank)
    assert stats['retrieval_supervised_records']>0
    assert torch.isfinite(loss)
    loss.backward()
    for name in ('state_query.weight','cap_query.weight','matcher.0.weight'):
        grad=dict(m.retriever.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(),name


def test_candidate_ranking_prefers_real_gain():
    scores=torch.tensor([[[.0,.0,-.2]]],requires_grad=True)
    gains=torch.tensor([[[[.3,.3],[.0,.0],[-.2,-.2]]]])
    v=torch.ones_like(gains,dtype=torch.bool)
    l=candidate_ranking_loss(scores,gains,v)
    l.backward()
    assert scores.grad[0,0,0]<0
    assert scores.grad[0,0,2]>0


def test_paired_block_requires_real_blocks():
    tr={'prediction':np.zeros((3,12,2)), 'target':np.ones((3,12,2)),
        'valid':np.ones((3,12,2),dtype=bool),'row_id':np.arange(3)}
    alt={**tr,'prediction':np.ones((3,12,2))*2}
    res=paired_block_stats(tr,alt,np.array([100,200,300]),10)
    assert res['ci95'] is None and res['blocks']==2


def test_validation_audit_is_fail_closed(fixture_data,tmp_path):
    m,bank,b=fixture_data
    cfg={'training':{'batch_size':2,'workers':0},'memory':{'purge_steps':0},
         'innovation_audit':{'validation_queries':4,'candidate_queries':2,
                             'candidate_nodes':3,'candidate_slots':2,'bootstrap_samples':10,'seed':10}}
    report=audit(m,bank,TinySet(b),torch.device('cpu'),cfg,tmp_path)
    assert set(report)=={'C1','C2','C3'}
    assert all(v['status']=='INCONCLUSIVE' for v in report.values())
    assert (tmp_path/'innovation_verdict.json').is_file()
    assert json.loads((tmp_path/'evaluation_manifest.json').read_text())['test_set_used_for_tuning'] is False


def test_v6_staged_cpu_fit_and_audit(fixture_data,tmp_path):
    m,bank,b=fixture_data;ds=TinySet(b)
    cfg={'experiment':{'seed':10},'training':{'lr':1e-3,'weight_decay':0.,'grad_clip':3.,
         'batch_size':2,'workers':0,'masked_weight':.1,'tail_weight':.01,
         'utility_weight':.1,'replay_interval':1,'replay_batch':2,'log_every':100,
         'stage_epochs':{'support':1,'utility':1,'joint':1,'risk':1},
         'amp':False,'support_weight':0.,'stage_audit_enabled':False},
         'teacher':{'max_queries':4,'batch_size':2,'candidates_per_query':3,'nodes_per_query':4,'support_queries':0},
         'memory':{'purge_steps':0}}
    ck=fit(m,bank,ds,ds,torch.device('cpu'),cfg,tmp_path,{'backbone_sha256':'toy'})
    assert ck.exists()
    saved=torch.load(ck,map_location='cpu',weights_only=False)
    assert saved['version']=='6.0.0'
    assert (tmp_path/'stage_support_best.pt').exists()
    assert (tmp_path/'stage_joint_best.pt').exists()
    metrics=[json.loads(z) for z in (tmp_path/'train_metrics.jsonl').read_text().splitlines()]
    assert {r['stage'] for r in metrics}=={'support','utility','joint','risk'}
    assert all(np.isfinite(r['train_loss']) for r in metrics)
