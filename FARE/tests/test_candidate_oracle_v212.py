"""Native FARE v2.1.1 integration regression tests for Oracle Audit v2.1.2."""
from __future__ import annotations
import copy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
import torch
from test_v21_core import fixture
from fare.candidate_oracle import (
    inference_only,validate_candidate_pool,null_intervention,
    selected_top1_intervention,singleton_intervention,evaluate_query_regions,
)
from fare.retrieval_diagnostics import summarize


def _single_fixture():
    cfg,model,memory,batch=fixture(b=2)
    model.eval()
    return model,memory,{k:v[:1].clone() if isinstance(v,torch.Tensor) and v.shape[0]==2 else v
                         for k,v in batch.items()}


def test_forward_parity_and_no_future_leakage():
    model,memory,batch=_single_fixture()
    inp=inference_only(batch)
    assert set(inp)=={'x_obs','obs_mask','tod','dow','window_start'}
    with torch.no_grad():
        base,pool=model.retrieve_for_diagnostics(inp,memory)
        replay=model.forward_from_retrieved(inp,base,pool)['prediction']
        normal=model(inp,memory)['prediction']
        assert torch.allclose(replay,normal,atol=1e-6)
        changed=dict(batch,y_norm=torch.randn_like(batch['y_norm'])*500,
                     y_raw=torch.randn_like(batch['y_raw'])*500)
        replay_changed=model(inference_only(changed),memory)['prediction']
    assert torch.allclose(replay,replay_changed,atol=1e-6)


def test_node_sparse_adapter_agrees_full():
    model,memory,batch=_single_fixture()
    inp=inference_only(batch)
    with torch.no_grad():
        base,pool=model.retrieve_for_diagnostics(inp,memory)
        g=0
        valid=validate_candidate_pool(model,memory,inp,pool)
        idx=int(torch.where(valid[g])[0][0])
        encoded=model.retrieval.encode_audit_candidate(inp,base,memory,model.regions,pool,g,idx)
        # Recompute same candidate with full node dimensions and compare g indices.
        cid=encoded['coarse_id'];n=model.n
        ids=torch.full((1,n,1),cid,dtype=torch.long)
        q=pool['node_query'].unsqueeze(2)
        whole=model.retrieval.adapter(memory,ids,inp,base['prediction'],q,model.retrieval.latent_dim)
        nodes=encoded['node_ids']
        for name in ('forecast_delta','token','state_token','reliability','future_valid'):
            assert torch.allclose(whole[name][:,nodes],encoded['evidence'][name],atol=1e-6),name


def test_intervention_is_region_local_and_does_not_mutate_pool():
    model,memory,batch=_single_fixture()
    inp=inference_only(batch)
    with torch.no_grad():
        base,pool=model.retrieve_for_diagnostics(inp,memory)
        snapshot={k:pool[k].clone() for k in ('ids','weights','forecast_delta','tokens')}
        g=0
        null=null_intervention(model,pool,g)
        nodes=torch.where(model.regions.owner==g)[0]
        other=torch.where(model.regions.owner!=g)[0]
        assert null['weights'][:,nodes].abs().sum()==0
        assert torch.equal(null['weights'][:,other],pool['weights'][:,other])
        top1=selected_top1_intervention(model,pool,g)
        assert torch.equal(top1['weights'][:,other],pool['weights'][:,other])
        valid=validate_candidate_pool(model,memory,inp,pool)
        rank=int(torch.where(valid[g])[0][0])
        enc=model.retrieval.encode_audit_candidate(inp,base,memory,model.regions,pool,g,rank)
        singleton=singleton_intervention(model,pool,g,enc)
        assert torch.equal(singleton['weights'][:,other],pool['weights'][:,other])
        assert torch.all(singleton['ids'][:,nodes,0]==enc['coarse_id'])
        assert torch.isfinite(model.forward_from_retrieved(inp,base,singleton)['prediction']).all()
        for k,x in snapshot.items():assert torch.equal(pool[k],x),k


def test_complete_oracle_summary_and_positive_definition(tmp_path):
    model,memory,batch=_single_fixture()
    with torch.no_grad():
        rows=evaluate_query_regions(model,memory,batch,std=2.0,max_candidates=4,region_ids=[0])
    assert len(rows)==1
    item=rows[0]
    assert item['num_evaluated_candidates']>0
    assert item['num_evaluated_candidates']<=4
    assert item['oracle_with_abstention_gain']>=0
    assert item['real_gain_vs_null']==pytest.approx(item['null_mae']-item['real_mae'])
    assert all(c['legal'] for c in item['candidates'])
    item.update(split='val',severity=None,row_id=0,window_start=5000)
    result=summarize(rows,tmp_path)
    assert len(result)==1
    assert (tmp_path/'oracle_summary.md').exists()
    assert (tmp_path/'oracle_summary.csv').exists()


def test_causal_filter_prevents_future_candidates():
    model,memory,batch=_single_fixture()
    inp=inference_only(batch)
    with torch.no_grad():
        base,pool=model.retrieve_for_diagnostics(inp,memory)
    original=memory.starts.clone()
    memory.starts[:]=inp['window_start'][0]+200
    try:
        valid=validate_candidate_pool(model,memory,inp,pool)
        assert not valid.any()
    finally:memory.starts.copy_(original)
