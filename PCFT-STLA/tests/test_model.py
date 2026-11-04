import torch
from pcft_stla.v6.training import compute_loss,make_trainable,optimizer_for,scaler_for,epoch_loop,MetricAccumulator

CFG={'training':{'unfreeze_last_stla_layers':1,'lr_encoder':1e-4,'lr_new':5e-4,'weight_decay':1e-4,
       'grad_clip':1.,'amp':False,'initial_amp_scale':1024,
       'loss':{'rmse':0.05,'masked':0.1,'evolution':0.05}},'runtime':{'print_every':100}}

def test_stage_forward_and_gradients(small):
    model,batch=small
    for st in model.STAGES:
        make_trainable(model,st,CFG)
        out=model(batch)
        assert out['prediction'].shape==(2,12,5)
        assert torch.isfinite(out['prediction']).all()
        loss,_=compute_loss(batch,out,CFG)
        loss.backward()
        assert model.head[-1].weight.grad is not None
        if st in ('transport','coupled'):
            assert model.dynamics.trans.msg.weight.grad is not None
            assert model.dynamics.trans.msg.weight.grad.abs().sum()>0
            assert 0<=out['future_support'].min() and out['future_support'].max()<=1
        model.zero_grad(set_to_none=True)

def test_causal_future_label_invariance(small):
    model,batch=small;model.eval();model.set_stage('coupled')
    with torch.no_grad():
        a=model(batch)['prediction']
        changed={**batch,'y_norm':torch.randn_like(batch['y_norm'])*10000,
                     'y_raw':torch.randn_like(batch['y_raw'])*10000}
        b=model(changed)['prediction']
    assert torch.equal(a,b)

def test_interventions_operate(small):
    model,batch=small;model.set_stage('coupled');model.eval()
    with torch.no_grad():
        full=model(batch)['prediction']
        disabled=model(batch,intervention='no_transport')['prediction']
        disabled_c1=model(batch,intervention='disable_c1')['prediction']
        reversed_lag=model(batch,intervention='reverse_lag')['prediction']
    assert not torch.allclose(full,disabled)
    assert not torch.allclose(full,disabled_c1)
    assert not torch.allclose(full,reversed_lag)

def test_metric_direct_mask_broadcast(small):
    _,batch=small
    acc=MetricAccumulator();acc.update(batch['y_norm'],batch,0.,1.)
    res=acc.finish()
    assert abs(res['mae'])<1e-7
    assert res['masked_count']==2*12*5
    assert res['h12_mae']<1e-7

def test_single_training_epoch(small):
    model,batch=small
    make_trainable(model,'coupled',CFG)
    opt=optimizer_for(model,CFG)
    from torch.utils.data import Dataset,DataLoader
    class One(Dataset):
        def __len__(self):return 2
        def __getitem__(self,i):return {k:v[i] for k,v in batch.items()}
    loader=DataLoader(One(),batch_size=2)
    result=epoch_loop(model,loader,torch.device('cpu'),CFG,opt=opt,scaler=scaler_for(torch.device('cpu'),CFG))
    assert result['rows']==2 and result['loss']>0

def test_no_source_no_propagation(small):
    model,batch=small;model.set_stage('coupled')
    with torch.no_grad():
        H=12;B=2;N=5;D=model.encoder.base.model_dim
        bank=torch.randn((B,H,N,D))
        support=torch.zeros((B,H,N))
        p,nxt,_=model.dynamics.trans(bank,support,model.edge_index,model.edge_attr,bank[:,-1],0)
    assert torch.equal(p,torch.zeros_like(p))
    assert torch.equal(nxt,torch.zeros_like(nxt))

def test_functional_context_is_not_physical_support(small):
    model,batch=small
    model.set_stage('coupled')
    model.eval()
    with torch.no_grad():
        a=model(batch)['prediction']
        b=model(batch,intervention='remove_functional')['prediction']
        assert not torch.allclose(a,b)
    # Zero observation sources always prohibit physical transport, regardless
    # of the existence of many nonphysical functional edges.
    H=12;B=2;N=5;D=model.encoder.base.model_dim
    with torch.no_grad():
        p,s,_=model.dynamics.trans(torch.randn(B,H,N,D),torch.zeros(B,H,N),
                                    model.edge_index,model.edge_attr,torch.randn(B,N,D),0)
    assert torch.count_nonzero(p)==0 and torch.count_nonzero(s)==0

def test_graph_intervention_direction_and_lag(small):
    model,batch=small
    model.set_stage('coupled');model.eval()
    with torch.no_grad():
        a=model(batch)['prediction']
        b=model(batch,intervention='reverse_direction')['prediction']
        c=model(batch,intervention='uniform_lag')['prediction']
    assert a.shape==b.shape==c.shape
    assert torch.isfinite(b).all() and torch.isfinite(c).all()

def test_both_graph_branches_train_in_transport_and_coupled(small):
    from pcft_stla.v6.training import make_trainable
    model,batch=small
    for stage in ('transport','coupled'):
        make_trainable(model,stage,CFG)
        assert model.ocpr.functional.msg.weight.requires_grad
        assert model.dynamics.functional.msg.weight.requires_grad
        assert model.dynamics.functional_scale.requires_grad
        assert model.evidence_reach.edge_lag.weight.requires_grad
        model.zero_grad(set_to_none=True)
        model(batch)['prediction'].mean().backward()
        assert model.dynamics.functional.msg.weight.grad is not None
        assert model.ocpr.functional.msg.weight.grad is not None
