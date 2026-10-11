"""Fast-path regression tests. These do not require Gadi Base_SupportBench."""
import copy
import tempfile
from pathlib import Path

import torch
from torch.nn import functional as F

from test_full import fixture
from fare.losses import losses, forecast_validation_loss
from fare.ranking import target_candidate_errors, target_recovery_ranking_loss, forecast_utility_ranking_loss
from fare.trainer import FARETrainer


def test_indexed_history_future_gather_equals_dense():
    cfg,model,memory,batch=fixture(n=19,b=3,m=52)
    memory.future=torch.randn(52,12,19)
    memory.future_valid=torch.rand(52,12,19)>.3
    b,g,k=3,model.regions.g,4
    ids=torch.randint(0,52,(b,g,k))
    node_ids=model.regions.region_node_ids
    a,v=memory.fetch_nodes(ids,node_ids)
    assert torch.equal(a,torch.gather(memory.x[ids],-1,node_ids[None,:,None,None,:].expand(b,g,k,12,-1)))
    assert torch.equal(v,torch.gather(memory.valid[ids],-1,node_ids[None,:,None,None,:].expand(b,g,k,12,-1)))
    f,fv=memory.fetch_nodes(ids,node_ids,kind='future')
    assert torch.equal(f,torch.gather(memory.future[ids],-1,node_ids[None,:,None,None,:].expand(b,g,k,12,-1)))
    assert torch.equal(fv,torch.gather(memory.future_valid[ids],-1,node_ids[None,:,None,None,:].expand(b,g,k,12,-1)))
    last,vm=memory.fetch_nodes(ids,node_ids,time_index=-1)
    assert torch.equal(last,a[:,:,:,-1]);assert torch.equal(vm,v[:,:,:,-1])


def test_analog_region_cache_equals_original_dense_average():
    cfg,model,memory,batch=fixture(n=19,b=3,m=52)
    memory.future=torch.randn(52,12,19)
    memory.future_valid=torch.rand(52,12,19)>.3
    memory.ensure_region_future(model.regions,chunk_size=7)
    ids=torch.randint(0,52,(3,model.regions.g,4))
    f=memory.future[ids]
    valid=memory.future_valid[ids]
    mask=model.regions.target_mask[None,:,None,None,:]
    dense=(f*valid.float()*mask).sum(-1)/(valid.float()*mask).sum(-1).clamp_min(1)
    cached=memory.region_future[ids,torch.arange(model.regions.g)[None,:,None]]
    assert torch.allclose(cached,dense,atol=1e-6,rtol=1e-6)


def test_vectorized_state_targets_match_loop():
    cfg,model,memory,batch=fixture(n=18,b=2,m=43)
    ids=torch.randint(0,len(memory.x),(2,model.regions.g,3))
    retrieved={'ids':ids,'valid':torch.ones_like(ids,dtype=torch.bool),'weights':torch.softmax(torch.rand_like(ids,dtype=torch.float32),-1)}
    fast,eligible=target_candidate_errors(batch,retrieved,memory,model.regions)
    target=batch['artificial_mask'].bool()&batch['raw_valid'].bool()
    for j in range(model.regions.g):
        nodes=torch.where(model.regions.owner==j)[0]
        x=memory.x[ids[:,j]][:,:,:,nodes]
        v=memory.valid[ids[:,j]][:,:,:,nodes]
        wanted=target[:,:,nodes][:,None]&v
        count=wanted.sum((-1,-2))
        err=((x-batch['x_full'][:,:,nodes][:,None]).abs()*wanted).sum((-1,-2))/count.clamp_min(1)
        assert torch.allclose(err,fast[:,j],atol=1e-6)
        assert torch.equal(eligible[:,j],count>0)
    l,d=target_recovery_ranking_loss(batch,retrieved,memory,model.regions)
    assert torch.isfinite(l)


def test_fast_retrieval_path_has_same_objective_and_gradient():
    torch.manual_seed(61)
    cfg,model,memory,batch=fixture(n=15,b=2,m=48)
    memory.future=torch.randn(48,12,15)
    memory.future_valid=torch.ones(48,12,15,dtype=torch.bool)
    batch['y_norm']=torch.randn(2,12,15)
    batch['window_start']=torch.tensor([3100,3210])
    model.set_phase('retrieval');model.refresh(memory)
    full=model(batch,memory)
    l1=losses(model,batch,full,memory,'retrieval',0,1,cfg)
    model.zero_grad(set_to_none=True)
    l1['total'].backward()
    full_grads={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
    fast=model(batch,memory,phase_fast='retrieval')
    l2=losses(model,batch,fast,memory,'retrieval',0,1,cfg)
    model.zero_grad(set_to_none=True)
    l2['total'].backward()
    assert torch.allclose(l1['total'],l2['total'],atol=1e-6)
    for n,g in full_grads.items():
        p=dict(model.named_parameters())[n]
        assert p.grad is not None and torch.allclose(g,p.grad,atol=1e-5,rtol=1e-4),n

def test_validation_forecast_only_matches_original_forecast():
    cfg,model,memory,batch=fixture(n=15,b=2,m=43)
    model.set_phase('full');model.refresh(memory)
    out=model(batch,memory)
    a=losses(model,batch,out,memory,'full',0,1,cfg)
    b=forecast_validation_loss(batch,out,0,1)
    assert torch.allclose(a['forecast'],b['forecast'],atol=1e-6)


def test_trainer_fast_stage_and_log_without_benchmark():
    cfg,model,memory,batch=fixture(n=15,b=2,m=43)
    cfg['training']['num_workers']=0
    cfg['training']['batch_size']=2
    memory.future=torch.randn(43,12,15)
    memory.future_valid=torch.ones(43,12,15,dtype=torch.bool)
    class Tiny(torch.utils.data.Dataset):
        def __len__(self):return 4
        def __getitem__(self,i):
            row={k:(v[i%2].clone() if v.ndim>0 else v.clone()) for k,v in batch.items()}
            row['y_norm']=row['y_raw'].clone()
            row['row_id']=torch.tensor(i)
            row['window_start']=torch.tensor(3000+i*52)
            return row
    ds=Tiny()
    with tempfile.TemporaryDirectory() as folder:
        meta={'dataset':'TraffiDent_Net_06','target':'speed','mean':0.,'std':1.}
        trainer=FARETrainer(model,memory,ds,ds,cfg,meta,torch.device('cpu'),folder)
        model.set_phase('retrieval')
        opt=trainer._optimizer('retrieval')
        train=trainer._epoch('retrieval',opt,max_batches=1)
        val=trainer._epoch('retrieval',None,max_batches=1)
        assert torch.isfinite(torch.tensor(train['total']))
        assert torch.isfinite(torch.tensor(val['total']))
        model.set_phase('full')
        val=trainer._epoch('full',None,max_batches=1)
        assert set(val)=={'total','forecast','retrieval_valid_groups','candidate_delta','final_gate_mean'}
