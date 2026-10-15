import torch

def test_full_forward_and_backward(fixture_data):
    model,mem,b=fixture_data
    out=model(b,mem)
    assert out['prediction'].shape==(2,12,6)
    assert torch.isfinite(out['prediction']).all()
    assert torch.allclose(out['selection']['weights'].sum((2,3))+out['selection']['null'],torch.ones(2,6,12),atol=1e-5)
    loss=(out['prediction']-b['target']).square().mean()
    loss.backward()
    assert model.support.embed[0].weight.grad is not None

def test_null_evidence_is_exact_base(fixture_data):
    model,mem,b=fixture_data
    with torch.no_grad():
        o=model(b,mem,intervention='no_evidence')
    assert torch.allclose(o['prediction'],o['base_prediction'],atol=1e-6)

def test_weight_preserving_loo_and_replacement(fixture_data):
    model,mem,b=fixture_data
    with torch.no_grad():
        full=model(b,mem)
        drop=model.edit_selection(full['selection'],full['proposals'],full['candidate_pool'],drop_slot=0)
        original=full['selection']['weights'].sum((2,3))+full['selection']['null']
        modified=drop['weights'].sum((2,3))+drop['null']
        assert torch.allclose(original,modified,atol=1e-5)
        rep=model.edit_selection(full['selection'],full['proposals'],full['candidate_pool'],replace_slot=0,replace_position=1)
        assert torch.allclose(rep['weights'][:,:,1],full['selection']['weights'][:,:,1])
        assert torch.allclose(rep['weights'].sum((2,3))+rep['null'],torch.ones_like(rep['null']),atol=1e-5)

def test_interventions(fixture_data):
    model,mem,b=fixture_data
    for mode in ['no_support','no_direction','no_regime','no_propagation_alignment','no_residual','no_evolution',
                 'no_set_utility','no_counterfactual_utility','no_null','no_risk_control','no_hidden_injection',
                 'no_forecast_correction','random_memory','shuffled_memory']:
        with torch.no_grad():out=model(b,mem,intervention=mode)
        assert torch.isfinite(out['prediction']).all(),mode

def test_single_train_epoch_and_teacher(fixture_data,tmp_path):
    from fare.training import fit
    from torch.utils.data import Dataset
    model,mem,b=fixture_data
    class Tiny(Dataset):
        def __len__(self):return 4
        def __getitem__(self,idx):
            return {k:(v[idx%2] if v.ndim>0 else v) if k!='row_id' else torch.tensor(idx)
                    for k,v in b.items()}
    cfg={'experiment':{'seed':10},'training':{'lr':1e-3,'weight_decay':1e-4,'grad_clip':3.0,
       'batch_size':2,'workers':0,'masked_weight':.1,'tail_weight':.1,'utility_weight':.2,'log_every':2,
       'stage_epochs':{'evidence':1,'utility':1,'integration':1,'risk':1,'joint':1}},
       'teacher':{'max_queries':4,'batch_size':2},'memory':{'purge_steps':0}}
    best=fit(model,mem,Tiny(),Tiny(),torch.device('cpu'),cfg,tmp_path,{'unit_test':True})
    assert best.is_file()
    assert (tmp_path/'training_complete.json').is_file()
