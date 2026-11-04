"""Small CPU contracts for the read-only frozen C1 add-on."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('frozen_c1_addon',ROOT/'scripts/c1_frozen_analysis.py')
addon=importlib.util.module_from_spec(spec)
spec.loader.exec_module(addon)


def test_weight_intervention_preserves_validity_and_none():
    w=torch.tensor([[[[.2]],[[.1]],[[.0]],[[.4]],[[.3]]]],dtype=torch.float32)
    valid=torch.tensor([[[[True]],[[True]],[[False]],[[True]]]])
    o={'candidate_weights':w,'candidate_available':valid}
    results=addon.candidate_weights_interventions(o)
    for name,ww in results.items():
        torch.testing.assert_close(ww.sum(1),torch.ones_like(ww.sum(1)))
        assert ww.shape==w.shape
        assert ww[:,2].abs().max()==0
        if name!='none_only':torch.testing.assert_close(ww[:,-1],w[:,-1])
    torch.testing.assert_close(results['uniform_keep_none'][0,:,0,0],torch.tensor([.7/3,.7/3,0.,.7/3,.3]))
    assert results['top1_keep_none'][0,3,0,0]>.69
    assert not torch.allclose(results['permute_keep_none'],w)


def test_permute_all_invalid_finite_and_none_is_one():
    w=torch.zeros(2,4,3,5);w[:,-1]=1.
    valid=torch.zeros(2,3,3,5,dtype=torch.bool)
    for _,arr in addon.candidate_weights_interventions(
            {'candidate_weights':w,'candidate_available':valid}).items():
        assert torch.isfinite(arr).all()
        torch.testing.assert_close(arr[:,-1],torch.ones_like(arr[:,-1]))
        assert (arr[:,:-1]==0).all()


def test_candidate_prediction_none_matches_original_formula():
    class Dummy:
        c1_activation=torch.nn.Parameter(torch.tensor(.5))
    o={'candidate_residuals':torch.ones(1,2,1,1),
       'residual_bound':3.,'base_prediction':torch.tensor([[[2.]]]),
       'normal':torch.tensor([[[4.]]])}
    w=torch.tensor([[[[0.]],[[0.]],[[1.]]]])
    res=addon.candidate_prediction(o,Dummy(),w)
    expected=4.+torch.tanh(torch.tensor(.5))*(2.-4.)
    torch.testing.assert_close(res.squeeze(),expected)


def test_block_bootstrap_paired_and_no_data():
    rows=[]
    for i in range(12):
        for method,loss in [('normal',5.),('c1',4.)]:
            rows.append({'index':i,'window_start':i*144,'method':method,
                         'absolute_error_sum':loss,'valid_count':1})
    result=addon.bootstrap_pair(rows,'normal','c1',288,100,1)
    assert result['difference_mae_first_minus_second']==1.
    np.testing.assert_allclose(result['bootstrap_95pct'],[1.,1.])
    assert addon.bootstrap_pair(rows,'normal','absent',288,100,1) is None


def test_auc_direction_and_ties():
    assert addon.auc_from_labels([0.1,0.5,0.8,0.9],[False,False,True,True])==1.
    assert addon.auc_from_labels([1.,1.,1.,1.],[False,False,True,True])==.5
    assert addon.auc_from_labels([0.,0.],[True,True]) is None
