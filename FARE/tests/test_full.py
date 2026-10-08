import copy,os,sys,tempfile
from pathlib import Path
import numpy as np, torch
from torch import nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fare.config import load_config
from fare.memory import TrainMemory,populate_latents
from fare.model import FARE
from fare.losses import losses
from fare.trainer import save_ckpt,load_ckpt

class FakePos(nn.Module):
    def forward(self,x):return x*0
class FakeBlock(nn.Module):
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(64,64),nn.GELU())
    def forward(self,x,ne,te):return self.net(x)+x

class FakeSTLA(nn.Module):
    def __init__(self,n=10):
        super().__init__();self.num_nodes=n;self.in_steps=12;self.out_steps=12;self.output_dim=1;self.input_dim=2
        self.steps_per_week=2016;self.tow_embedding_dim=32;self.model_dim=64
        self.input_proj=nn.Linear(2,32);self.tow_embedding=nn.Embedding(2016,32)
        self.node_position_encoding=FakePos();self.time_position_encoding=FakePos()
        self.MLP=nn.Linear(64,64)
        self.attn_layers_st=nn.ModuleList([FakeBlock() for _ in range(6)])
        self.output_proj=nn.Linear(12*64,12)

def fixture(n=10,b=3,m=40):
    cfg=load_config();cfg['graph']['target_region_size']=4;cfg['graph']['max_anchors']=6
    cfg['retrieval']['coarse_topk']=8;cfg['retrieval']['topk']=3
    cfg['training']['batch_size']=b
    adj=torch.eye(n)
    for i in range(n-1):adj[i,i+1]=adj[i+1,i]=1
    stla=FakeSTLA(n)
    model=FARE(stla,adj,cfg)
    memory=TrainMemory(torch.randn(m,12,n),torch.ones(m,12,n,dtype=torch.bool),torch.arange(m)*40,tod=torch.randint(0,288,(m,12)),dow=torch.randint(0,7,(m,12)))
    batch={'x_obs':torch.randn(b,12,n),'obs_mask':torch.rand(b,12,n)>.6,'tod':torch.randint(0,288,(b,12)),'dow':torch.randint(0,7,(b,12)),'window_start':torch.arange(b)*40+3,'x_full':torch.randn(b,12,n),'artificial_mask':torch.rand(b,12,n)>.5,'raw_valid':torch.ones(b,12,n,dtype=torch.bool),'y_raw':torch.randn(b,12,n),'future_valid':torch.ones(b,12,n,dtype=torch.bool),'direct_mask':torch.ones(b,n,dtype=torch.bool)}
    model.refresh(memory)
    return cfg,model,memory,batch

def test_forward_and_grad():
    cfg,model,memory,batch=fixture()
    populate_latents(memory,model,batch_size=5)
    model.set_phase('full');model.train();model.refresh(memory)
    o=model(batch,memory)
    assert o['prediction'].shape==(3,12,10)
    assert o['recovered_state'].shape==(3,12,10)
    assert o['latent_innovation'].shape==(3,12,10,64)
    assert torch.isfinite(o['prediction']).all()
    assert torch.all(o['state_gate'][batch['obs_mask']]==0)
    assert torch.allclose(o['supported_history'][batch['obs_mask']],batch['x_obs'][batch['obs_mask']])
    err=losses(model,batch,o,memory,'full',0.,1.,cfg)
    err['total'].backward()
    assert model.state_recovery.adapter[-1].weight.grad is not None
    assert model.innovation.local[-1].weight.grad is not None
    assert model.retrieval.rerank[-1].weight.grad is not None
    assert model.stla.base.output_proj.weight.grad is not None
    return float(err['total'].detach())

def test_no_candidate():
    cfg,model,memory,batch=fixture()
    batch['window_start']=torch.arange(3)*0
    memory.starts.fill_(0)
    model.refresh(memory);model.eval()
    with torch.no_grad():o=model(batch,memory)
    assert torch.isfinite(o['prediction']).all()
    assert o['final_gate'].abs().max().item()==0
    assert torch.allclose(o['prediction'],o['base_prediction'])

def test_no_future_leakage():
    cfg,model,memory,batch=fixture()
    stripped={key:batch[key] for key in ['x_obs','obs_mask','tod','dow','window_start']}
    model.eval()
    with torch.no_grad():o=model(stripped,memory)
    assert torch.isfinite(o['prediction']).all()
    # Changing unobserved full history and future labels cannot change inference.
    with torch.no_grad():o2=model(dict(stripped),memory)
    assert torch.allclose(o2['prediction'],o['prediction'])

def test_checkpoint_roundtrip():
    cfg,model,memory,batch=fixture()
    with tempfile.TemporaryDirectory() as t:
        file=Path(t)/'ck.pt'
        meta={'dataset':'TraffiDent_Net_06','target':'speed','mean':0.,'std':1.}
        save_ckpt(file,model,None,cfg,meta,1,'full',{})
        other=copy.deepcopy(model)
        for p in other.parameters():
            if p.requires_grad:p.data.add_(.01)
        load_ckpt(file,other)
        a=model.state_dict();b=other.state_dict()
        assert all(torch.equal(a[k],b[k]) for k in a)

if __name__=='__main__':
    for fn in (test_forward_and_grad,test_no_candidate,test_no_future_leakage,test_checkpoint_roundtrip):
        print('RUN',fn.__name__,flush=True);value=fn();print('PASS',fn.__name__,value,flush=True)

def test_staged_training():
    import pandas as pd
    from fare.trainer import FARETrainer
    class TinyDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.rows=[]
            for k in range(6):
                cfg,_,_,b=fixture(b=1)
                row={key:value[0].clone() if value.ndim>0 else value.clone() for key,value in b.items()}
                row['window_start']=torch.tensor(1600+k*45)
                row['row_id']=torch.tensor(k)
                self.rows.append(row)
            self.meta=pd.DataFrame({'family':['entry_random']*6,'view_id':[str(j) for j in range(6)]})
            self.family_column='family'
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):return self.rows[i]
    cfg,model,memory,batch=fixture()
    ds=TinyDataset();cfg['training']['phase_epochs']={k:1 for k in cfg['training']['phase_epochs']}
    cfg['training']['min_phase_epochs']=1;cfg['training']['patience']=2
    cfg['evaluation']['save_stage_predictions']=True
    meta={'dataset':'TraffiDent_Net_06','target':'speed','mean':0.,'std':1.}
    with tempfile.TemporaryDirectory() as folder:
        tr=FARETrainer(model,memory,ds,ds,cfg,meta,torch.device('cpu'),folder)
        phases=['retrieval','recovery','injection','partial','full']
        path=tr.train(phases,smoke=True)
        assert path.is_file()
        for phase in phases:
            assert (Path(folder)/'stages'/phase/'val_summary.json').exists()

if __name__=='__main__':
    print('RUN test_staged_training',flush=True)
    test_staged_training()
    print('PASS test_staged_training',flush=True)

def test_network_agnostic_shapes():
    for n in (7,15,28):
        cfg,model,memory,batch=fixture(n=n,b=2,m=35)
        model.eval()
        with torch.no_grad():
            y=model({k:batch[k] for k in ['x_obs','obs_mask','tod','dow','window_start']},memory)
        assert y['prediction'].shape==(2,12,n),n
        assert model.regions.target_mask.sum(-2).eq(1).all(),n

def test_sparse_observability_gradient_regression():
    """A single legal retrieval candidate used to create sqrt(0) NaN grads."""
    cfg,model,memory,batch=fixture(n=12,b=2,m=8)
    memory.starts.fill_(0)
    memory.starts[-1]=30000
    batch['window_start'].fill_(0)
    batch['obs_mask'].fill_(False)
    batch['obs_mask'][:,0,0]=True
    model.set_phase('recovery');model.train();model.refresh(memory)
    optimizer=torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],lr=1e-4
    )
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        out=model(batch,memory)
        assert torch.isfinite(out['state_uncertainty']).all()
        assert torch.isfinite(out['final_gate']).all()
        loss=losses(model,batch,out,memory,'recovery',0.,1.,cfg)['total']
        assert torch.isfinite(loss)
        loss.backward()
        for name,p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), name
        optimizer.step()
    for name,p in model.named_parameters():
        assert torch.isfinite(p).all(),name

def test_nonfinite_loss_is_not_hidden():
    from fare.losses import masked_mean
    bad=torch.tensor([float('nan'),1.])
    try:
        masked_mean(bad,torch.tensor([True,True]))
    except FloatingPointError:
        pass
    else:
        raise AssertionError('An active NaN was silently converted to zero')
    assert masked_mean(bad,torch.tensor([False,True])).item()==1.

if __name__=='__main__':
    print('RUN test_network_agnostic_shapes',flush=True)
    test_network_agnostic_shapes()
    print('PASS test_network_agnostic_shapes',flush=True)
    for fn in (test_sparse_observability_gradient_regression,test_nonfinite_loss_is_not_hidden):
        print('RUN',fn.__name__,flush=True)
        fn()
        print('PASS',fn.__name__,flush=True)
