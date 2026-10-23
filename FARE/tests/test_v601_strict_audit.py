"""Scientifically consequential regression checks for FARE v6.0.1 audit.

No STLAformer vendor or external data required for these method-unit tests.
"""
import csv
import json
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset
from fare.innovation_audit import (_assert_validation,_stratified_subset,
    _candidate_scores,_candidate_probe,paired_block_stats,audit)


class ValidationFixture(Dataset):
    split='val'
    def __init__(self, batch):
        self.batch=batch
    def __len__(self):return 6
    def __getitem__(self,i):
        out={k:v[i%2].clone() for k,v in self.batch.items()}
        out['row_id']=torch.tensor(100+i)
        out['origin']=torch.tensor(500+i*288)
        out['x_full']=out['x'].clone();out['raw_valid']=out['mask'].clone()
        out['target_raw']=out['target'].clone()
        return out


def _cfg():
    return {'training':{'batch_size':2,'workers':0},'memory':{'purge_steps':0},
        'innovation_audit':{'validation_queries':4,'candidate_queries':4,'candidate_nodes':3,
            'candidate_slots':2,'candidate_alphas':[0.,.25,.5,1.],
            'bootstrap_samples':50,'seed':10,'progress_every_batches':100}}


def test_static_prior_query_independent(fixture_data):
    model,memory,b=fixture_data
    out=model(b,memory)
    good,scores=_candidate_scores(out['prepared'],b,2)
    assert scores['static_prior'].shape==good.shape
    altered=dict(b,row_id=b['row_id']+53)
    _,b_scores=_candidate_scores(out['prepared'],altered,2)
    assert torch.equal(scores['static_prior'],b_scores['static_prior'])
    assert not torch.equal(scores['matched_random'],b_scores['matched_random'])
    static=out['candidate_pool']['candidate_capability'][...,:2,:]
    assert torch.allclose(scores['static_prior'],static[...,7]-.1*static[...,3].abs())


def test_shared_horizon_weights_are_actually_shared(fixture_data):
    m,memory,b=fixture_data
    m.eval()
    with torch.no_grad():
        full=m(b,memory)
        other=m.intervene_cached(b,memory,full,'audit_shared_horizon')
    w=other['selection']['weights']
    null=other['selection']['null']
    assert torch.allclose(w,w[...,:1].expand_as(w),atol=0,rtol=0)
    assert torch.allclose(null,null[...,:1].expand_as(null),atol=0,rtol=0)
    assert torch.isfinite(other['prediction']).all()


def test_same_pool_controls_do_not_replace_memory_or_backbone(fixture_data):
    m,memory,b=fixture_data;m.eval()
    with torch.no_grad():
        full=m(b,memory)
        legal=full['candidate_pool']['ids']
        for mode in ('audit_static_prior','audit_same_pool_similarity','audit_same_pool_random'):
            out=m.intervene_cached(b,memory,full,mode)
            ids=out['selection']['selected_ids']
            assert torch.all((ids[...,None]==legal[:,:,None,:]).any(-1))
            assert torch.allclose(out['support_prediction'],full['support_prediction'])
            assert torch.isfinite(out['prediction']).all()


def test_candidate_null_oracle_and_alpha_zero(fixture_data,tmp_path):
    m,memory,b=fixture_data
    result=_candidate_probe(m,memory,ValidationFixture(b),torch.device('cpu'),_cfg(),tmp_path)
    assert result['evaluated_node_queries']>0
    assert result['curves']['alpha_0']['oracle_with_null'] == 0.0
    for method in ('learned','similarity','matched_random','static_prior'):
        assert abs(result['curves']['alpha_0'][method])<1e-7
    assert result['curves']['alpha_1']['oracle_with_null']>=0
    assert len(result['fixed_alphas'])==4
    assert (tmp_path/'candidate_node_details.csv').is_file()
    assert result['paired_learned_vs_random']['learned_gate']['query_count']>0


def test_fresh_audit_does_not_mutate_checkpoint_parameters(fixture_data,tmp_path):
    m,memory,b=fixture_data
    old={k:v.detach().clone() for k,v in m.state_dict().items()}
    out=audit(m,memory,ValidationFixture(b),torch.device('cpu'),_cfg(),tmp_path)
    assert all(v['status']=='INCONCLUSIVE' for v in out.values())
    assert all(torch.equal(v,m.state_dict()[k]) for k,v in old.items())
    assert (tmp_path/'horizon_metrics.csv').exists()
    assert (tmp_path/'candidate_node_details.csv').exists()
    assert (tmp_path/'audit_source.json').exists() is False
    manifest=json.loads((tmp_path/'evaluation_manifest.json').read_text())
    assert manifest['test_set_used_for_tuning'] is False
    assert manifest['intervention_definitions']['audit_static_prior'].startswith('OOF')


def test_refuse_test_split(fixture_data,tmp_path):
    m,memory,b=fixture_data
    ds=ValidationFixture(b);ds.split='test'
    with pytest.raises(ValueError,match='Validation'):
        audit(m,memory,ds,torch.device('cpu'),_cfg(),tmp_path)


def test_stratified_sample_uses_only_metadata():
    class NoRead(Dataset):
        split='val'
        def __init__(self):
            self.ix=np.arange(1200)
            self.meta=pd.DataFrame({'window_start':np.arange(1200)*12,
                                    'severity':np.tile([20,40,60,90],300),
                                    'family':np.tile(['regional','temporal','directional'],400)})
        def __len__(self):return len(self.ix)
        def __getitem__(self,i):raise AssertionError('Sampler must not read any future labels')
    ds=NoRead()
    a=_stratified_subset(ds,192,10);b=_stratified_subset(ds,192,10)
    assert a.indices==b.indices
    assert len(a)==192
    frame=ds.meta.iloc[a.indices]
    assert set(frame.severity)=={20,40,60,90}
    assert set(frame.family)=={'regional','temporal','directional'}


def test_paired_statistics_reject_misaligned_targets():
    z=np.zeros((4,12,3));v=np.ones_like(z,dtype=bool)
    aa={'prediction':z,'target':z,'valid':v,'row_id':np.arange(4)}
    bb={**aa,'target':z+1}
    with pytest.raises(ValueError,match='targets'):
        paired_block_stats(aa,bb,np.arange(4)*300)
