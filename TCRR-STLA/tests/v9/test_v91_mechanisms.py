"""Behavior-level tests for v9.1 CF utility, supervision, fixed anchor and pairing."""
import numpy as np
import pytest
import torch
from test_v9_core import fake
from test_v9_preparation import fixture,FakeTeacher
from tcrr_stla.v9.model import (TCRRV9,utility_targets,train_loss,masked_mae,
                               cross_failure_pairing,paired_cross_failure_loss)
from tcrr_stla.v9.prepare import prepare_normal,prepare_view,PreparedView
from tcrr_stla.v9.graph import GraphBundle
from tcrr_stla.v9.training import run_training


def test_cf_utility_reads_target_node_response():
    m,kw=fake(b=2,k=3,h=12,n=6)
    first=m(**kw)
    later=kw['cf'].clone();later[:,0,:,2]+=torch.linspace(-1.,1.,12)
    changed=m(**{**kw,'cf':later})
    difference=(first['cf_utility'][:,0,:,2]-changed['cf_utility'][:,0,:,2]).abs().max()
    assert float(difference)>1e-6, 'CF utility is blind to target-node CF response'


def test_cf_utility_reads_horizon_order():
    m,kw=fake(b=2,k=3,h=12,n=6)
    first=m(**kw)['cf_utility']
    reversed_=m(**{**kw,'cf':kw['cf'].flip(2)})['cf_utility']
    assert float((first-reversed_).abs().max())>1e-6


def test_teacher_utility_ignores_invalid_future_labels_even_with_nans():
    m,kw=fake()
    out=m(**kw)
    target=torch.randn_like(kw['base']);valid_future=torch.zeros_like(target,dtype=torch.bool)
    valid_future[:,0]=True
    alt=target.clone();alt[:,1:]=torch.nan
    value1,mask1=utility_targets(out['base'],out['normal_raw'],out['candidate_cf'],
                                 target,kw['valid'],m.bound,m.blocks,valid_future,True)
    value2,mask2=utility_targets(out['base'],out['normal_raw'],out['candidate_cf'],
                                 alt,kw['valid'],m.bound,m.blocks,valid_future,True)
    assert torch.isfinite(value2).all()
    torch.testing.assert_close(value1,value2)
    torch.testing.assert_close(mask1,mask2)
    l1,_=train_loss(out,target,valid_future,kw['valid'])
    l2,_=train_loss(out,alt,valid_future,kw['valid'])
    torch.testing.assert_close(l1,l2)


def test_zero_candidate_cannot_rank_over_null():
    m,kw=fake()
    kw['valid'].zero_()
    out=m(**kw)
    torch.testing.assert_close(out['prediction'],out['normal_prediction'],rtol=0,atol=1e-6)
    assert bool((out['cf_weights'][:,-1]>.99999).all())


def test_graph_reliability_missing_neighbours_no_unearned_graph_evidence():
    m,kw=fake()
    kw['mask']=torch.zeros_like(kw['mask']);kw['x'][:]=100
    out=m(**kw)
    assert float(out['cf_graph_reliability'].abs().max())==0
    no_graph=m(**kw,cf_graph_enabled=False)
    torch.testing.assert_close(out['prediction'],no_graph['prediction'])


def test_cross_failure_matches_candidate_identity_and_never_position_only():
    ids0=torch.tensor([[2,5,7],[7,8,9]])
    ids1=torch.tensor([[5,2,100],[9,7,8]])
    ok=torch.ones_like(ids0,dtype=torch.bool)
    lookup,paired=cross_failure_pairing(ids0,ids1,ok,ok)
    assert paired.tolist()==[[True,True,False],[True,True,True]]
    assert lookup.tolist()[0][:2]==[1,0]
    with pytest.raises(ValueError):
        cross_failure_pairing(ids0,torch.tensor([[5,5,100],[9,7,8]]),ok,ok)


def test_model_parameters_normal_path_is_fixed_after_normal_stage(tmp_path):
    cache=fixture(tmp_path);root=tmp_path/'evidence';root.mkdir()
    t=FakeTeacher();dev=torch.device('cpu');k=2
    prepare_normal(cache,t,dev,root,limit=40,chunk=8)
    train_ids=np.zeros((40,2),dtype=np.int32)
    train_ids[:,0]=np.arange(40)//2
    train_ids[:,1]=np.maximum(np.arange(40)//2-1,0)
    train_npz=tmp_path/'train.npz'
    np.savez(train_npz,version=25,indices=train_ids,scores=np.zeros((40,k),'float32'),
             coverage=np.ones((40,k),'float32'))
    val_npz=tmp_path/'val.npz'
    np.savez(val_npz,version=25,indices=np.zeros((6,k),np.int32),scores=np.zeros((6,k),'float32'),
             coverage=np.ones((6,k),'float32'))
    for i in (0,1):prepare_view(cache,'train',i,t,dev,root,train_npz,k=k,limit=12,chunk=4)
    prepare_view(cache,'val',0,t,dev,root,val_npz,k=k,chunk=4)
    a=PreparedView(root/'train_view0');b=PreparedView(root/'train_view1')
    np.testing.assert_array_equal(a.arrays['candidate_ids'],b.arrays['candidate_ids'])
    np.testing.assert_array_equal(a.arrays['valid'],b.arrays['valid'])
    assert len(set(bytes(x) for x in b.arrays['mask_hash']))>1
    assert a.meta['paired_candidate_protocol']=='fixed_official_identity'
    g=GraphBundle(torch.eye(4))
    cfg=dict(batch_size=4,eval_batch_size=4,workers=0,normal_epochs=1,
             transfer_epochs=1,utility_epochs=1,cross_epochs=1,cross_weight=.01,
             lr=.001,hidden=12,bound=3.,blocks=3,amp=False,utility_weight=.02,
             ranking_weight=.01,clip=5.,patience=3,teacher_alignment_weight=.005,
             full_info_weight=.01)
    run=tmp_path/'run'
    run_training(cache,root,g,8,0.,1.,run,10,cfg,smoke=True)
    normal=torch.load(run/'best_normal.pt',map_location='cpu',weights_only=False)['model']
    trans=torch.load(run/'best_transfer.pt',map_location='cpu',weights_only=False)['model']
    utility=torch.load(run/'best_utility.pt',map_location='cpu',weights_only=False)['model']
    cross=torch.load(run/'best_cross.pt',map_location='cpu',weights_only=False)['model']
    for key in normal:
        if key.startswith(('query.','match.','normal_fusion.','normal_scale.','none_normal')):
            torch.testing.assert_close(normal[key],trans[key],atol=0,rtol=0)
            torch.testing.assert_close(normal[key],utility[key],atol=0,rtol=0)
            torch.testing.assert_close(normal[key],cross[key],atol=0,rtol=0)
    assert cross['utility.weight'].shape==utility['utility.weight'].shape


def test_full_teacher_alignment_uses_increment_beyond_normal_anchor():
    """If Normal gives +1 and Full Teacher gives +1.5, CF target is +0.5."""
    m,kw=fake(b=1,k=2,h=12,n=6)
    out=m(**kw)
    base=torch.zeros_like(out['base'])
    normal_pred=torch.ones_like(base)
    teacher_cf=torch.ones_like(base)*1.5
    desired_cf=torch.ones_like(base)*.5
    # The alignment contribution is zero for correct incremental prediction;
    # using teacher_cf directly as target would incorrectly double-correct.
    out.update(base=base,normal_prediction=normal_pred,cf_increment=desired_cf,
               prediction=normal_pred+desired_cf,normal_raw=base)
    y=normal_pred+desired_cf
    future=torch.ones_like(y,dtype=torch.bool)
    total_good,_=train_loss(out,y,future,kw['valid'],utility_weight=0.,rank_weight=0.,
                            teacher_cf=teacher_cf,full_info_weight=0.,teacher_alignment_weight=1.)
    out['cf_increment']=teacher_cf
    total_wrong,_=train_loss(out,y,future,kw['valid'],utility_weight=0.,rank_weight=0.,
                             teacher_cf=teacher_cf,full_info_weight=0.,teacher_alignment_weight=1.)
    assert total_good.item()<total_wrong.item()
