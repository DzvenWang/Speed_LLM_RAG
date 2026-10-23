"""Experiment 1 tests are structural AND train/evaluate-path checks.

They deliberately use small synthetic tensors ONLY for CPU correctness testing.
No synthetic fallback exists in formal Gadi execution.
"""
from pathlib import Path
import json
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset
from fare.transfer_v62 import source_relative_evolution,TransferEvidenceCache,Experiment1Transfer
from fare.candidate_bank_v62 import FrozenCandidateBank,build_candidate_bank
from fare import experiment1_v62 as exp


def test_source_relative_evolution_uses_source_own_last_observation():
    x=torch.tensor([[[3.,40.],[5.,70.],[0.,0.]]])
    m=torch.tensor([[[True,True],[True,False],[False,False]]])
    y=torch.tensor([[[7.,75.],[9.,80.]]]);valid=torch.ones_like(y,dtype=torch.bool)
    ev,good,age=source_relative_evolution(x,m,y,valid)
    torch.testing.assert_close(ev,torch.tensor([[[2.,35.],[4.,40.]]]))
    assert age.tolist()==[[1,2]] and good.all()
    m[:,:,-1]=False
    ev,good,_=source_relative_evolution(x,m,y,valid)
    assert (~good[:,:,-1]).all() and (ev[:,:,-1]==0).all()


def test_t0_to_t3_forward_backward_and_null(fixture_data):
    model,mem,batch=fixture_data
    mem.attach_graph(model.graph)
    with torch.no_grad():
        base=model.backbone(batch['x'],batch['mask'],batch['tod'],batch['dow'])
        sup=model.support(batch['x'],batch['mask'],model.base_proj(base['hidden']),
                          base['prediction'],model.graph)
    b,n=batch['x'].shape[0],batch['x'].shape[-1]
    ids=torch.arange(3)[None,None,:].expand(b,n,-1)
    short={'ids':ids,'valid':torch.ones_like(ids,dtype=torch.bool),
           'scores':torch.tensor([2.,1.,.5])[None,None,:].expand(b,n,-1).clone()}
    for name in exp.VARIANTS:
        head=Experiment1Transfer(name,d=8,horizon=12,max_lag=2)
        out=head(short,mem,model.graph,sup,base['prediction'])
        assert out['prediction'].shape==(b,12,n)
        assert torch.isfinite(out['prediction']).all()
        loss=out['prediction'].square().mean()
        loss.backward()
        assert head.logit_strength.grad is not None and torch.isfinite(head.logit_strength.grad).all()
        if name=='candidate_aligned':
            assert head.transport.candidate_space[-1].weight.grad is not None
            assert torch.isfinite(head.transport.candidate_space[-1].weight.grad).all()
            assert head.transport.candidate_time[-1].weight.grad is not None
            assert torch.isfinite(head.transport.candidate_time[-1].weight.grad).all()
        no=dict(short,valid=torch.zeros_like(short['valid']))
        support_ref=base['prediction']+3.0
        with torch.no_grad():
            out_empty=head(no,mem,model.graph,sup,base['prediction'],
                           reference_prediction=support_ref)
        torch.testing.assert_close(out_empty['prediction'],support_ref,atol=1e-7,rtol=0)
    assert hasattr(mem,'_v62_source_evolution')

class TinyDataset(Dataset):
    def __init__(self,batch):
        self.items=[];self.ix=np.arange(3,dtype=np.int64)
        self.n=batch['x'].shape[-1]
        for i in range(3):
            d={k:(v[i%len(v)].clone() if len(v.shape)>0 else v.clone()) for k,v in batch.items()}
            d['row_id']=torch.tensor(i,dtype=torch.long)
            d['origin']=torch.tensor(100+i,dtype=torch.long)
            self.items.append(d)
    def __len__(self):return len(self.items)
    def __getitem__(self,i):return self.items[i]


def test_fixed_bank_is_causal_and_reused(fixture_data,tmp_path):
    model,mem,batch=fixture_data
    mem.attach_graph(model.graph)
    ds=TinyDataset(batch)
    cfg={'retrieval':{'shortlist_k':3},'memory':{'purge_steps':12}}
    meta={'target':'speed','dataset':'TraffiDent_Net_06','seed':10,
          'backbone_sha256':'a','memory_sha256':'b'}
    build_candidate_bank(model,mem,model.graph,ds,ds,'cpu',tmp_path/'bank',cfg,'hash',meta,
                         batch_size=2,workers=0)
    bank=FrozenCandidateBank(tmp_path/'bank','val',ds,
          {'support_checkpoint_sha256':'hash'})
    batch2=next(iter(torch.utils.data.DataLoader(ds,batch_size=2)))
    short=bank.fetch(batch2['row_id'],'cpu')
    assert short['ids'].shape==(2,ds.n,3)
    legal=mem.valid_for(batch2['origin'],'cpu',12)
    assert (short['valid'] & ~legal.gather(1,short['ids'].flatten(1)).reshape_as(short['valid'])).sum()==0
    with pytest.raises(ValueError):
        FrozenCandidateBank(tmp_path/'bank','train',ds,{'support_checkpoint_sha256':'wrong'})


def test_full_synthetic_4_variants_training_pipeline(fixture_data,monkeypatch,tmp_path):
    original_model,mem,batch=fixture_data
    ds=TinyDataset(batch)
    cfg={'experiment':{'seed':10,'target':'speed','dataset':'TraffiDent_Net_06'},
         'model':{'hidden_dim':8,'horizon':12},
         'traffic':{'max_temporal_lag':2},
         'memory':{'purge_steps':12},
         'retrieval':{'coarse_k':4,'shortlist_k':3,'final_k':2},
         'training':{'cache_gpu_memory':False,'grad_clip':5.},
         'experiment1':{'support_lr':1e-4,'support_epochs':1,'support_batch_size':2,
                        'candidate_batch_size':2,'transfer_batch_size':2,'transfer_lr':1e-4,
                        'candidate_temperature':.25,'transfer_epochs':1,'workers':0}}
    meta={'dataset':'TraffiDent_Net_06','target':'speed','seed':10,
          'backbone_sha256':'backbonefake','memory_sha256':'memoryfake'}
    def prepared(*args,**kwargs):
        # Fresh separate model instance for every stage; strict Support checkpoint restore.
        from fare.model import FARE
        from conftest import DummyBackbone
        torch.manual_seed(10)
        model=FARE(DummyBackbone(ds.n),original_model.graph,cfg)
        mem.attach_graph(model.graph)
        return model,mem,ds,ds,meta
    monkeypatch.setattr(exp,'prepare_experiment',prepared)
    monkeypatch.setattr(exp,'_pilot_datasets',lambda _cfg:(ds,ds))
    root=tmp_path/'e1'
    exp.train_support(cfg,'cpu','unused','unused',root)
    exp.candidate_bank(cfg,'cpu','unused','unused',root)
    feasibility=exp.direct_feasibility(cfg,'cpu','unused','unused',root)
    assert not feasibility['trained_transfer']
    for v in exp.VARIANTS:exp.train_transfer(cfg,'cpu','unused','unused',root,v)
    result=exp.summarize_experiment(cfg,root)
    assert result['innovation_status']=={'C1':'INCONCLUSIVE','C2':'INCONCLUSIVE','C3':'INCONCLUSIVE'}
    assert (root/'experiment1_table.csv').exists()
    assert all((root/'transfer'/v/'best.pt').exists() for v in exp.VARIANTS)
    with pytest.raises(FileExistsError):
        exp.train_support(cfg,'cpu','unused','unused',root)
