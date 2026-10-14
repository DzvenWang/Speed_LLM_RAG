"""FARE 2.1.3 native v2.1.1 trained-network counterfactual regression."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
import pytest
from test_candidate_oracle_v212 import _single_fixture
from fare.candidate_oracle import inference_only,validate_candidate_pool,evaluate_query_regions
from fare.weight_preserving_audit import (
    leave_one_slot_out,replace_slot_keep_weight,predict_from_pool,
    audit_region_interventions)
from fare.retrieval_diagnostics import summarize_counterfactual


def context():
    model,memory,batch=_single_fixture()
    inputs=inference_only(batch)
    with torch.no_grad():
        base,pool=model.retrieve_for_diagnostics(inputs,memory)
    return model,memory,batch,inputs,base,pool


def mass(pool,nodes):
    return pool['weights'][:,nodes].sum(-1)+pool['null_weight'][:,nodes]


def test_leave_one_slot_weight_to_null_and_no_mutation():
    model,mem,batch,inputs,base,pool=context()
    g=0
    target=torch.where(model.regions.owner==g)[0]
    other=torch.where(model.regions.owner!=g)[0]
    snapshot={x:pool[x].clone() for x in ('weights','ids','null_weight','forecast_delta')}
    dropped=leave_one_slot_out(model,pool,g,slot=0)
    assert torch.equal(dropped['weights'][:,target,0],torch.zeros_like(dropped['weights'][:,target,0]))
    assert torch.allclose(mass(pool,target),mass(dropped,target),atol=1e-6)
    assert torch.allclose(dropped['null_weight'][:,target]-pool['null_weight'][:,target],
                          pool['weights'][:,target,0],atol=1e-6)
    assert torch.equal(dropped['weights'][:,other],pool['weights'][:,other])
    with torch.no_grad():
        assert torch.isfinite(predict_from_pool(model,inputs,base,dropped)).all()
    for name,val in snapshot.items():
        assert torch.equal(pool[name],val),name


def test_node_slot_removal_is_node_specific():
    model,mem,batch,inputs,base,pool=context()
    nodes=torch.where(model.regions.owner==0)[0]
    node=int(nodes[0]);out=leave_one_slot_out(model,pool,0,0,node_id=node)
    other=torch.arange(model.n)!=node
    assert torch.equal(out['weights'][:,other],pool['weights'][:,other])
    assert torch.allclose(mass(out,nodes),mass(pool,nodes),atol=1e-6)
    with pytest.raises(ValueError):
        leave_one_slot_out(model,pool,0,0,node_id=model.n+10)


def test_same_weight_replacement_preserves_mass_and_others():
    model,mem,batch,inputs,base,pool=context()
    region=0
    legal=validate_candidate_pool(model,mem,inputs,pool)
    rank=int(torch.where(legal[region])[0][0])
    encoded=model.retrieval.encode_audit_candidate(inputs,base,mem,model.regions,pool,region,rank)
    changed=replace_slot_keep_weight(model,pool,region,0,encoded)
    nodes=torch.where(model.regions.owner==region)[0]
    other=torch.where(model.regions.owner!=region)[0]
    effective=encoded['effective'][:,:,0].bool()
    expected=torch.where(effective,pool['weights'][:,nodes,0],torch.zeros_like(pool['weights'][:,nodes,0]))
    assert torch.allclose(changed['weights'][:,nodes,0],expected,atol=1e-6)
    assert torch.allclose(mass(pool,nodes),mass(changed,nodes),atol=1e-6)
    assert torch.equal(changed['weights'][:,other],pool['weights'][:,other])
    assert torch.allclose(changed['weights'][:,nodes,1:],pool['weights'][:,nodes,1:])
    assert torch.equal(changed['ids'][:,nodes,0],torch.full_like(changed['ids'][:,nodes,0],encoded['coarse_id']))
    with torch.no_grad():
        pred=predict_from_pool(model,inputs,base,changed)
    assert torch.isfinite(pred).all()


def test_replacement_invalid_node_transfers_mass_to_null():
    model,mem,batch,inputs,base,pool=context()
    valid=validate_candidate_pool(model,mem,inputs,pool)
    r=int(torch.where(valid[0])[0][0])
    candidate=model.retrieval.encode_audit_candidate(inputs,base,mem,model.regions,pool,0,r)
    candidate['effective']=torch.zeros_like(candidate['effective'])
    changed=replace_slot_keep_weight(model,pool,0,0,candidate)
    nodes=torch.where(model.regions.owner==0)[0]
    assert not changed['valid'][:,nodes,0].any()
    assert (changed['weights'][:,nodes,0]==0).all()
    assert torch.allclose(changed['null_weight'][:,nodes],
                          pool['null_weight'][:,nodes]+pool['weights'][:,nodes,0],atol=1e-6)


def test_audit_scaling_parity_and_replacement_metrics(tmp_path):
    model,mem,batch,inputs,base,pool=context()
    with torch.no_grad():
        rows=evaluate_query_regions(model,mem,batch,std=2.,max_candidates=3,region_ids=[0],
              audit_modes=('loo','replacement','scaling','node_loo'),max_nodes=1,
              strict_pool_match=True)
    assert len(rows)==1
    row=rows[0]
    assert len(row['leave_one_out'])==pool['weights'].shape[-1]
    assert len(row['same_weight_replacements'])<=3
    assert row['scaling_alpha_0_null_parity']<=1e-5
    assert row['scaling_alpha_1_real_parity']<=1e-5
    assert row['best_scaling_improvement_over_real']>=-1e-7
    assert all(x['gain_vs_real']==pytest.approx(row['real_mae']-x['replacement_mae'])
               for x in row['same_weight_replacements'])
    assert all(x['keep_gain_vs_removing']==pytest.approx(x['removed_slot_mae']-row['real_mae'])
               for x in row['leave_one_out'])
    assert all(x['node_id'] in torch.where(model.regions.owner==0)[0].tolist()
               for x in row['node_leave_one_out'])
    row.update(split='test',severity=90,row_id=1)
    rep=summarize_counterfactual(rows,tmp_path)
    assert len(rep)==1
    assert (tmp_path/'counterfactual_summary.csv').exists()
    assert (tmp_path/'counterfactual_summary.md').exists()
    assert (tmp_path/'counterfactual_summary.json').exists()


def test_original_singleton_path_still_available():
    model,mem,batch,inputs,base,pool=context()
    with torch.no_grad():
        row=evaluate_query_regions(model,mem,batch,std=1.,max_candidates=2,region_ids=[0],
              audit_modes=('singleton',),max_nodes=0)[0]
    assert len(row['candidates'])<=2
    assert 'leave_one_out' not in row
    assert 'same_weight_replacements' not in row


def test_same_weight_identity_replay_matches_real_prediction():
    model,mem,batch,inputs,base,pool=context()
    nodes=torch.where(model.regions.owner==0)[0]
    slot=0
    identity={
        'coarse_id':int(pool['ids'][0,nodes[0],slot]),
        'effective':pool['valid'][:,nodes,slot:slot+1].clone(),
        'utility':pool['utility'][:,nodes,slot:slot+1].clone(),
        'risk':pool['risk'][:,nodes,slot:slot+1].clone(),
        'evidence':{name:t[:,nodes,slot:slot+1].clone()
                    for name,t in pool['selected_evidence'].items()},
    }
    with torch.no_grad():
        changed=replace_slot_keep_weight(model,pool,0,slot,identity)
        replay=predict_from_pool(model,inputs,base,changed)
        real=predict_from_pool(model,inputs,base,pool)
    # IDs are metadata to the integration path; every tensor used in forward
    # remains identical. This catches inadvertent attention/gate reweighting.
    assert torch.allclose(replay,real,atol=1e-6,rtol=0)
    assert torch.equal(changed['weights'],pool['weights'])
    assert torch.allclose(changed['null_weight'],pool['null_weight'],atol=1e-6)


def test_changing_labels_cannot_change_fixed_counterfactual_predictions():
    model,mem,batch,inputs,base,pool=context()
    valid=validate_candidate_pool(model,mem,inputs,pool)
    rank=int(torch.where(valid[0])[0][0])
    enc=model.retrieval.encode_audit_candidate(inputs,base,mem,model.regions,pool,0,rank)
    override=replace_slot_keep_weight(model,pool,0,0,enc)
    with torch.no_grad():
        original=predict_from_pool(model,inputs,base,override)
        altered=dict(batch,y_norm=torch.full_like(batch['y_norm'],1000.))
        new_inputs=inference_only(altered)
        new_base,new_pool=model.retrieve_for_diagnostics(new_inputs,mem)
        new_override=replace_slot_keep_weight(model,new_pool,0,0,enc)
        new_pred=predict_from_pool(model,new_inputs,new_base,new_override)
    assert torch.allclose(original,new_pred,atol=1e-6)
