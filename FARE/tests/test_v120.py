"""Strict P0 research contract + P1 target-recovery ranking regression tests."""
import copy,os,sys,tempfile
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_full import fixture,test_staged_training
from fare.ranking import target_candidate_errors,target_recovery_ranking_loss
from fare.losses import losses
from fare.trainer import FARETrainer


def inputs_only(bt):
    return {k:bt[k] for k in ('x_obs','obs_mask','tod','dow','window_start')}


def test_strict_module_ablation():
    cfg,model,mem,b=fixture(n=12,b=2,m=40)
    model.eval()
    # Early FARE adapters zero initialized; assert paths, not unrealistically
    # require large numerical differences before training them.
    called={'state':0,'innovation':0}
    state_call=model.state_recovery.forward
    innov_call=model.innovation.forward
    def track_state(*args,**kw):
        called['state']+=1;return state_call(*args,**kw)
    def track_innov(*args,**kw):
        called['innovation']+=1;return innov_call(*args,**kw)
    model.state_recovery.forward=track_state
    model.innovation.forward=track_innov
    inp=inputs_only(b)
    with torch.no_grad():
        full=model(inp,mem,'full')
        assert called=={'state':1,'innovation':1}
        state=model(inp,mem,'state_only')
        assert called=={'state':2,'innovation':1}, called
        assert torch.equal(state['latent_innovation'],torch.zeros_like(state['latent_innovation']))
        assert torch.equal(state['retrieved_token_abs'],torch.zeros_like(state['retrieved_token_abs']))
        assert state['final_gate'].max() <= model.controller.max_alpha
        innovation=model(inp,mem,'innovation_only')
        assert called=={'state':2,'innovation':2},called
        assert torch.count_nonzero(innovation['state_gate'])==0
        assert torch.equal(innovation['supported_history'],b['x_obs'])
        no_mem=model(inp,mem,'no_memory')
        assert called=={'state':2,'innovation':2},called
        assert torch.equal(no_mem['final_gate'],torch.zeros_like(no_mem['final_gate']))
        assert 'retrieved_ids' not in no_mem
        assert no_mem['prediction'].shape==full['prediction'].shape
        off=model(inp,mem,'retrieval_off')
        assert called=={'state':2,'innovation':2},called
        for mode in ('no_risk','no_utility','constant_alpha','no_mask_alignment',
                     'no_learned_reranker','no_observation_distance','random_memory'):
            res=model(inp,mem,mode)
            assert torch.isfinite(res['prediction']).all(),mode
            assert res['final_gate'].max()<=model.controller.max_alpha+1e-6
    return True


def test_ranking_loss_gradient_and_inference_safety():
    cfg,m,mem,b=fixture(n=14,b=3,m=55)
    # guarantee multiple missing targets in each region
    b['artificial_mask']=~b['obs_mask']
    b['raw_valid']=torch.ones_like(b['obs_mask'])
    m.set_phase('retrieval');m.train();m.refresh(mem)
    out=m(inputs_only(b),mem)
    ret=dict(ids=out['retrieved_ids'],valid=out['retrieved_valid_candidates'],weights=out['retrieved_weights'])
    err,legal=target_candidate_errors(b,ret,mem,m.regions)
    assert err.shape==out['retrieved_weights'].shape
    assert legal.any()
    ranking,diag=target_recovery_ranking_loss(b,ret,mem,m.regions)
    assert torch.isfinite(ranking)
    assert diag['ranking_eligible_fraction']>0
    ranking.backward()
    assert m.retrieval.rerank[-1].weight.grad is not None
    assert torch.isfinite(m.retrieval.rerank[-1].weight.grad).all()
    # never permit full labels to be used as an inference retrieval feature
    tampered=copy.deepcopy(b)
    tampered['x_full']=torch.randn_like(b['x_full'])*100
    tampered['y_raw']=torch.randn_like(b['y_raw'])*100
    m.eval()
    with torch.no_grad():
        a=m(inputs_only(b),mem)
        z=m(inputs_only(tampered),mem)
    assert torch.allclose(a['prediction'],z['prediction'])


def test_missing_targets_rank_no_nan():
    cfg,m,mem,b=fixture(n=8,b=2,m=15)
    m.eval()
    with torch.no_grad():o=m(inputs_only(b),mem)
    b['artificial_mask']=torch.zeros_like(b['obs_mask'])
    ret=dict(ids=o['retrieved_ids'],valid=o['retrieved_valid_candidates'],weights=o['retrieved_weights'])
    loss,diag=target_recovery_ranking_loss(b,ret,mem,m.regions)
    assert loss.item()==0
    assert torch.isfinite(loss)
    assert diag['ranking_eligible_fraction']==0


def test_no_memory_trainable_without_retrieval():
    cfg,m,mem,b=fixture(n=10,b=2,m=25)
    cfg['experiment']['training_mode']='no_memory'
    m.set_phase('injection');m.train()
    out=m(inputs_only(b),mem,mode='no_memory')
    ls=losses(m,b,out,mem,'injection',0.,1.,cfg)
    ls['total'].backward()
    assert m.stla.failure[-1].weight.grad is not None
    assert torch.isfinite(ls['total'])
    assert ls['retrieval'].item()==0


if __name__=='__main__':
    for fn in [test_strict_module_ablation,test_ranking_loss_gradient_and_inference_safety,
               test_missing_targets_rank_no_nan,test_no_memory_trainable_without_retrieval]:
        print('RUN',fn.__name__,flush=True);fn();print('PASS',fn.__name__,flush=True)


def test_no_memory_staged_training():
    import pandas as pd
    cfg,m,mem,b=fixture(n=9,b=2,m=10)
    cfg['experiment']['training_mode']='no_memory'
    cfg['training']['phase_epochs']={p:1 for p in cfg['training']['phase_epochs']}
    cfg['training']['batch_size']=2
    cfg['training']['num_workers']=0
    cfg['training']['min_phase_epochs']=1
    class LocalDataset(torch.utils.data.Dataset):
        family_column='family'
        def __init__(self):
            self.meta=pd.DataFrame({'family':['sensor_outage']*4,'view_id':['a','b','c','d']})
        def __len__(self):return 4
        def __getitem__(self,i):
            v={k:t[i%2].clone() for k,t in b.items()}
            v['row_id']=torch.tensor(i)
            v['window_start']=torch.tensor(i*64+1500)
            return v
    with tempfile.TemporaryDirectory() as td:
        ds=LocalDataset()
        tr=FARETrainer(m,mem,ds,ds,cfg,{'dataset':'TraffiDent_Net_01','target':'flow','mean':0.,'std':1.},torch.device('cpu'),td)
        ck=tr.train(['injection','partial','full'],smoke=True)
        assert ck.exists()
        import json
        summary=json.loads((Path(td)/'phase_summary.json').read_text())
        assert [x['phase'] for x in summary]==['injection','partial','full']
        for phase in ('injection','partial','full'):
            vals=json.loads((Path(td)/'stages'/phase/'val_summary.json').read_text())
            assert vals['final']['MAE']==vals['candidate']['MAE']
        print('FARE_NO_MEMORY_3_PHASE_PASS',flush=True)

if __name__=='__main__':
    print('RUN test_no_memory_staged_training',flush=True)
    test_no_memory_staged_training()
    print('PASS test_no_memory_staged_training',flush=True)
