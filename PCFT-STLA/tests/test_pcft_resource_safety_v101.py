"""Regression: 181395434 CUDA OOM and 181395435 DataLoader SIGBUS.

The original v1.0 model checkpoint and predicted values must remain unchanged.
"""
from __future__ import annotations
import copy
import torch
from pcft_stla.runtime_safety import (
    safe_evaluation_config, predict_with_retry, is_cuda_oom,
)


def test_safe_runtime_overrides_saved_training_config_without_mutating():
    old = {'training': {'batch_size': 64, 'ablation': 'full'},
           'runtime': {'num_workers': 4, 'pin_memory': True,
                       'persistent_workers': True, 'prefetch_factor': 2,
                       'preload_repository_gpu': True, 'amp': True}}
    original = copy.deepcopy(old)
    safe = safe_evaluation_config(old)
    assert old == original  # no saved model/training-config mutation
    assert safe['training']['batch_size'] == 8
    assert safe['training']['ablation'] == 'full'
    assert safe['runtime']['num_workers'] == 0
    assert safe['runtime']['persistent_workers'] is False
    assert safe['runtime']['preload_repository_gpu'] is False
    assert safe['runtime']['amp'] is True
    assert safe_evaluation_config(old,batch_size=16)['training']['batch_size']==16


def test_invalid_resource_settings_fail():
    cfg={'training': {},'runtime': {}}
    for bs in (0,-1):
        try:
            safe_evaluation_config(cfg,batch_size=bs)
        except ValueError: pass
        else: raise AssertionError('Invalid batch must fail')
    try:
        safe_evaluation_config(cfg,num_workers=-1)
    except ValueError: pass
    else: raise AssertionError('Negative worker count must fail')


def test_cuda_oom_detection_does_not_swallow_other_errors():
    assert is_cuda_oom(RuntimeError('CUDA out of memory. Tried to allocate'))
    assert not is_cuda_oom(RuntimeError('model state_dict mismatch'))


def test_retry_restarts_full_dataset_with_same_order_and_complete_results():
    n=11
    class Dataset:
        def __len__(self): return n
    dataset=Dataset()
    used=[]
    def fake_loader(ds,batch,shuffle,cfg):
        used.append(batch)
        assert shuffle is False
        return list(range(len(ds)))
    def fake_predict(model,loader,index,repo,device,mean,std,mode,amp=False):
        # At bs=8 report OOM after partial iteration; entire pass must restart.
        if used[-1]>4:
            raise RuntimeError('CUDA out of memory. Tried to allocate 874.00 MiB')
        return {'prediction': torch.arange(n).numpy().reshape(n,1,1)}
    result,bs=predict_with_retry(None,dataset,None,None,torch.device('cpu'),
                 0.,1.,'full',False,{},fake_loader,fake_predict,8)
    assert used==[8,4]
    assert bs==4 and len(result['prediction'])==n

def test_non_oom_error_is_not_retried():
    n=4
    class Dataset:
        def __len__(self):return n
    used=[]
    def fake_loader(ds,batch,shuffle,cfg):
        used.append(batch)
        return []
    def fake_predict(*args,**kwargs):
        raise RuntimeError('checksum mismatch')
    try:
        predict_with_retry(None,Dataset(),None,None,torch.device('cpu'),
                           0.,1.,'full',False,{},fake_loader,fake_predict,8)
    except RuntimeError as exc: assert 'checksum mismatch' in str(exc)
    else: raise AssertionError('Missing exception')
    assert used==[8]
