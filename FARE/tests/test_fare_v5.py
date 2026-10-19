"""V5 behavioral contracts, tested on a real new-model forward graph (toy Backbone)."""
import torch
import numpy as np
import pytest
from fare.forecast_utility import utility_loss
from fare.utility_teacher import generate_utility_teacher,replay_utility_loss


def test_full_and_grad(fixture_data):
    m,bank,b=fixture_data
    out=m(b,bank)
    assert tuple(out['prediction'].shape)==(2,12,6)
    assert torch.isfinite(out['prediction']).all()
    w=out['selection']['weights']
    assert w.shape[3]==3
    total=w.sum((2,3))+out['selection']['null']
    assert torch.allclose(total,torch.ones_like(total),atol=1e-5)
    (out['prediction']-b['target']).square().mean().backward()
    assert m.support.embed[0].weight.grad is not None


def test_null_exact_and_no_memory(fixture_data):
    m,bank,b=fixture_data
    with torch.no_grad():
        o=m(b,bank,intervention='no_evidence')
        assert torch.equal(o['prediction'],o['base_prediction'])
        b2=dict(b,origin=torch.zeros_like(b['origin']))
        x=m(b2,bank)
        assert torch.equal(x['prediction'],x['base_prediction'])
        assert torch.allclose(x['selection']['null'],torch.ones_like(x['selection']['null']))


def test_counterfactual_and_replay_gradient(fixture_data):
    from torch.utils.data import Dataset
    m,bank,b=fixture_data
    class Small(Dataset):
        def __len__(self):return 4
        def __getitem__(self,i):return {k:(v[i%2] if k!='row_id' else torch.tensor(i)) for k,v in b.items()}
    teacher=generate_utility_teacher(m,bank,Small(),torch.device('cpu'),
                                     limit=4,batch_size=2,candidates=3,nodes_per_query=4)
    assert len(teacher)==4
    assert all('labels' in x and 'checked' in x for x in teacher.values())
    loss,stats=replay_utility_loss(m,teacher,torch.device('cpu'),max_replays=2)
    assert torch.isfinite(loss)
    assert stats['utility_labels']>0
    loss.backward()
    assert m.ranker.gain.weight.grad is not None
    assert m.ranker.gain.weight.grad.abs().sum()>0


def test_temporal_order_matters(fixture_data):
    m,bank,b=fixture_data
    out=m(b,bank)
    prop=out['prepared']['proposals']
    query=out['support']['embedding']
    e=m.ranker.evidence
    one=e(prop,query)
    reversed_=dict(prop,deltas=prop['deltas'].flip(-1),available=prop['available'].flip(-1),
                   compatibility=prop['compatibility'].flip(-1))
    two=e(reversed_,query)
    # Historical mean unchanged, but order-aware encoded token must differ.
    assert not torch.allclose(one,two,atol=1e-6)


def test_candidate_utility_supervision(fixture_data):
    m,bank,b=fixture_data
    out=m(b,bank)
    rank=out['rank'];shape=rank['gain'].shape
    labels=torch.randn(shape)*.15
    checked=rank['valid'][...,None].expand_as(labels)
    loss=utility_loss(rank,labels,checked)
    assert torch.isfinite(loss)
    loss.backward()
    assert m.ranker.gain.weight.grad is not None


def test_random_and_shuffle_recompute_candidate_features(fixture_data):
    m,bank,b=fixture_data
    with torch.no_grad():
        full=m(b,bank)
        rnd=m(b,bank,intervention='random_memory')
        shuf=m(b,bank,intervention='shuffled_memory')
    assert torch.isfinite(rnd['prediction']).all()
    assert torch.isfinite(shuf['prediction']).all()
    assert torch.all(rnd['candidate_pool']['scores']==0)
    assert torch.all(shuf['candidate_pool']['scores']==0)
    assert rnd['prepared']['proposals']['deltas'].shape[-2:]==(3,12)


def test_resume_and_checkpoint_version(fixture_data,tmp_path):
    from torch.utils.data import Dataset
    from fare.training import fit
    m,bank,b=fixture_data
    class Small(Dataset):
        def __len__(self):return 4
        def __getitem__(self,i):return {k:(v[i%2] if k!='row_id' else torch.tensor(i)) for k,v in b.items()}
    cfg={'experiment':{'seed':10},'training':{'lr':1e-3,'weight_decay':0.,'grad_clip':3.,
         'batch_size':2,'workers':0,'masked_weight':.1,'tail_weight':.1,'utility_weight':.3,
         'replay_interval':1,'replay_batch':2,'log_every':100,
         'stage_epochs':{'evidence':1,'retrieval':1,'joint':1,'risk':1}},
         'teacher':{'max_queries':4,'batch_size':2,'candidates_per_query':3,'nodes_per_query':6},
         'memory':{'purge_steps':0}}
    ck=fit(m,bank,Small(),Small(),torch.device('cpu'),cfg,tmp_path,{'unit_test':True})
    assert ck.exists()
    obj=torch.load(ck,map_location='cpu',weights_only=False)
    assert obj['version']=='5.0.0'
    assert (tmp_path/'train_metrics.jsonl').exists()


def test_cpu_half_autocast(fixture_data):
    m,bank,b=fixture_data
    with torch.autocast('cpu',dtype=torch.float16),torch.no_grad():
        out=m(b,bank)
        assert torch.isfinite(out['prediction']).all()
