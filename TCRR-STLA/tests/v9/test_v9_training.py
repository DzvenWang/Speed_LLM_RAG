import numpy as np
import torch
from test_v9_preparation import fixture,FakeTeacher
from tcrr_stla.v9.prepare import prepare_normal,prepare_view
from tcrr_stla.v9.graph import GraphBundle
from tcrr_stla.v9.training import run_training


def test_training_smoke_all_stages_resume(tmp_path):
    cache=fixture(tmp_path);out=tmp_path/'v9_evidence';out.mkdir()
    teacher=FakeTeacher();dev=torch.device('cpu');k=2
    prepare_normal(cache,teacher,dev,out,limit=40,chunk=8)
    ids=np.zeros((40,k),dtype=np.int32)
    ids[:,0]=np.arange(40)//2
    ids[:,1]=np.maximum(0,np.arange(40)//2-1)
    train_rc=tmp_path/'train.npz'
    np.savez(train_rc,version=25,indices=ids,scores=np.zeros((40,k),'float32'),
             coverage=np.ones((40,k),'float32'))
    val_rc=tmp_path/'val.npz'
    np.savez(val_rc,version=25,indices=np.zeros((6,k),'int32'),scores=np.zeros((6,k),'float32'),
             coverage=np.ones((6,k),'float32'))
    prepare_view(cache,'train',0,teacher,dev,out,train_rc,k=k,chunk=4,limit=12)
    prepare_view(cache,'train',1,teacher,dev,out,train_rc,k=k,chunk=4,limit=12)
    prepare_view(cache,'val',0,teacher,dev,out,val_rc,k=k,chunk=4)
    g=GraphBundle(torch.eye(4))
    cfg=dict(batch_size=4,eval_batch_size=4,workers=0,normal_epochs=1,
             transfer_epochs=1,utility_epochs=1,cross_epochs=1,cross_weight=.01,
             lr=.001,hidden=12,bound=3.,blocks=3,amp=False,utility_weight=.02,
             ranking_weight=.01,clip=5.,patience=3)
    run=tmp_path/'run'
    best=run_training(cache,out,g,8,0.,1.,run,10,cfg,smoke=True)
    assert best.is_file()
    assert (run/'best_normal.pt').is_file()
    assert (run/'best_transfer.pt').is_file()
    assert (run/'best_utility.pt').is_file()
    assert (run/'last.pt').is_file()
    # Resume a completed run should not rewrite/duplicate epochs.
    resumed=run_training(cache,out,g,8,0.,1.,run,10,cfg,smoke=True,resume=run/'last.pt')
    assert resumed==best


def test_one_epoch_resets_gradients_between_optimizer_steps():
    """Each optimizer step must use only the current minibatch gradient."""
    from tcrr_stla.v9.training import one_epoch
    from torch import nn

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.0))

        def forward(self, **b):
            pred = self.weight.expand_as(b['base'])
            return {'prediction': pred, 'normal_prediction': pred}

    class RecordingSGD(torch.optim.SGD):
        def __init__(self, parameters):
            super().__init__(parameters, lr=0.1)
            self.seen = []
            self.weight_ref = self.param_groups[0]['params'][0]

        def step(self, *args, **kwargs):
            self.seen.append(self.weight_ref.grad.detach().clone().item())
            return super().step(*args, **kwargs)

    toy=ToyModel()
    opt=RecordingSGD(toy.parameters())
    one=dict(base=torch.zeros(1,1,1),y_norm=torch.ones(1,1,1),
             future_valid=torch.ones(1,1,1),x=torch.zeros(1,1,1),
             mask=torch.ones(1,1,1),normal=torch.zeros(1,1,1,1),
             cf=torch.zeros(1,1,1,1),context=torch.zeros(1,1,1),
             score=torch.zeros(1,1),valid=torch.ones(1,1,dtype=torch.bool),
             match=torch.zeros(1,1,1,2))
    scaler=torch.amp.GradScaler('cuda',enabled=False)
    one_epoch(toy,[one,one],opt,scaler,torch.device('cpu'),0,1,'normal',
              dict(amp=False,clip=10.0))
    assert len(opt.seen)==2
    assert abs(opt.seen[0]+1.)<1e-5
    assert abs(opt.seen[1]+1.)<1e-5, 'Second gradient incorrectly includes the first batch'
