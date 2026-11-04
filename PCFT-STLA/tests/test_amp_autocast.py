"""Regression for Gadi job 182751710/182751720: AMP sparse graph message dtypes.

CPU bfloat16 autocast emulates the mixed-dtype behavior from CUDA fp16
without requiring an interactive V100 allocation.
"""
import pytest
import torch
from pcft_stla.v6.training import compute_loss

CFG = {'training': {'loss': {'rmse': 0.05, 'masked': 0.10, 'evolution': 0.05}}}

@pytest.mark.parametrize('autocast_dtype', [torch.bfloat16, torch.float16])
@pytest.mark.parametrize('stage', ['direct', 'observability', 'transport', 'coupled'])
def test_forward_backward_cpu_autocast_for_all_stages(small,stage,autocast_dtype):
    model,batch=small
    model.set_stage(stage)
    model.train()
    with torch.autocast(device_type='cpu',dtype=autocast_dtype):
        output=model(batch)
        loss,_=compute_loss(batch,output,CFG)
    assert torch.isfinite(loss)
    assert torch.isfinite(output['prediction']).all()
    loss.backward()
    assert model.head[-1].weight.grad is not None
    assert torch.isfinite(model.head[-1].weight.grad).all()
    if stage in ('observability','transport','coupled'):
        assert model.ocpr.graph.msg.weight.grad is not None
        assert torch.isfinite(model.ocpr.graph.msg.weight.grad).all()
    if stage in ('transport','coupled'):
        assert model.dynamics.trans.msg.weight.grad is not None
        assert torch.isfinite(model.dynamics.trans.msg.weight.grad).all()

@pytest.mark.parametrize('autocast_dtype', [torch.bfloat16, torch.float16])
def test_crossmask_auxiliary_loss_under_autocast(small,autocast_dtype):
    # Also exercise the S1/S3 auxiliary step, which batch_probe does not reach.
    from pcft_stla.v6.training import crossmask_consistency
    model,batch=small
    model.set_stage('coupled')
    model.train()
    with torch.autocast(device_type='cpu',dtype=autocast_dtype):
        output=model(batch)
        state=output['state']
        loss=crossmask_consistency(model,batch,state['state'],state['reliability'])
    assert torch.isfinite(loss)
    loss.backward()
    assert model.ocpr.graph.msg.weight.grad is not None
    assert torch.isfinite(model.ocpr.graph.msg.weight.grad).all()
