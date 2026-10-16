"""Regression for Gadi V100 overflow when autocast produces FP16 score tensors.

Runs on login-node CPUs as well as CUDA nodes.  These tests intentionally leave
AMP enabled in all FARE retrieval/transport/utility paths, not just retrieval.
"""
import torch
import pytest
from fare.retrieval import HierarchicalRetriever
from fare.teacher import generate_teacher
from torch.utils.data import Dataset


def test_half_chunked_topk_mixed_lawfulness():
    q=torch.randn(2,3,4,dtype=torch.float16)
    key=torch.randn(20,3,4,dtype=torch.float16)
    lawful=torch.zeros(2,20,dtype=torch.bool)
    lawful[0,0:4]=True
    lawful[1,10:15]=True
    with torch.autocast('cpu',dtype=torch.float16):
        value,ids=HierarchicalRetriever._topk_chunked(q,key,lawful,6,7)
    assert value.dtype==torch.float32
    assert torch.isfinite(value).all()
    for b in range(2):
        allowed=lawful[b,ids[b]]
        assert (value[b][~allowed] < -1e6).all()
        assert allowed.any()


def test_half_chunked_topk_no_lawful_candidates():
    q=torch.randn(1,2,4,dtype=torch.float16)
    keys=torch.randn(6,2,4,dtype=torch.float16)
    with torch.autocast('cpu',dtype=torch.float16):
        values,ids=HierarchicalRetriever._topk_chunked(q,keys,torch.zeros(1,6,dtype=torch.bool),4,3)
    assert torch.isfinite(values).all()
    assert (values==-1e7).all()
    assert ids.shape==(1,2,4)


def test_full_model_fp16_autocast_forward_backward(fixture_data):
    model,mem,batch=fixture_data
    model.train()
    with torch.autocast('cpu',dtype=torch.float16):
        result=model(batch,mem)
        loss=(result['prediction'].float()-batch['target']).square().mean()
    assert torch.isfinite(result['prediction']).all()
    assert torch.isfinite(loss)
    assert torch.isfinite(result['selection']['weights']).all()
    assert torch.allclose(result['selection']['weights'].sum((2,3))+result['selection']['null'],
                          torch.ones_like(result['selection']['null']),atol=3e-3)
    loss.backward()
    grad=model.retriever.state_query.weight.grad
    assert grad is not None and torch.isfinite(grad).all()


def test_half_autocast_interventions_and_teacher(fixture_data):
    model,mem,batch=fixture_data
    model.eval()
    with torch.no_grad(), torch.autocast('cpu',dtype=torch.float16):
        for intervention in ('no_direction','no_evidence','no_set_utility',
                             'no_swap_utility','no_null','no_propagation_alignment',
                             'random_memory','no_risk_control','no_evolution_information',
                             'no_residual_information'):
            out=model(batch,mem,intervention=intervention)
            assert torch.isfinite(out['prediction']).all(),intervention

    class Mini(Dataset):
        def __len__(self):return 4
        def __getitem__(self,i):
            return {k: (v[i%2] if torch.is_tensor(v) and v.ndim else v)
                    if k!='row_id' else torch.tensor(i) for k,v in batch.items()}
    with torch.autocast('cpu',dtype=torch.float16):
        teacher=generate_teacher(model,mem,Mini(),torch.device('cpu'),limit=4,batch_size=2,
                                 nodes_per_query=3)
    assert teacher
    assert all(torch.isfinite(v['action_alpha']).all() for v in teacher.values())


def test_half_autocast_empty_evidence_model(fixture_data):
    model,mem,batch=fixture_data
    batch=dict(batch,origin=torch.full_like(batch['origin'],-200))
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.float16):
        out=model(batch,mem)
    assert not out['candidate_pool']['valid'].any()
    assert torch.isfinite(out['prediction']).all()
    assert torch.allclose(out['selection']['null'],torch.ones_like(out['selection']['null']),atol=1e-6)


def test_fp16_safe_masking_primitive():
    from fare.numerics import mask_invalid_logits
    half=torch.tensor([[1.,2.],[3.,4.]],dtype=torch.float16)
    bad=torch.tensor([[False,True],[True,True]])
    with torch.autocast('cpu',dtype=torch.float16):
        masked=mask_invalid_logits(half,bad)
    assert masked.dtype==torch.float32
    assert masked.tolist()==[[1.,-1e7],[-1e7,-1e7]]
