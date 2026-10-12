"""Executable synthetic regressions without external benchmark or GPU."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch import nn
from fare.model import FARE
from fare.memory import TrainMemory
from fare.evaluation import _metrics, common_metrics
from fare.losses import losses

class Block(nn.Module):
    def forward(self,z,ne,te):
        return z

class MockSTLA(nn.Module):
    def __init__(self,n=8,h=12,p=12):
        super().__init__()
        self.num_nodes=n;self.in_steps=h;self.out_steps=p
        self.input_dim=2;self.model_dim=16;self.tow_embedding_dim=8
        self.steps_per_week=2016;self.output_dim=1
        self.input_proj=nn.Linear(2,8)
        self.tow_embedding=nn.Embedding(2016,8)
        self.node_position_encoding=nn.Identity()
        self.time_position_encoding=nn.Identity()
        self.MLP=nn.Linear(16,16)
        self.attn_layers_st=nn.ModuleList([Block() for _ in range(6)])
        self.output_proj=nn.Linear(h*16,p)


def fixture(b=2,n=8,m=40,h=12,p=12):
    torch.manual_seed(10)
    from fare.config import load_config
    cfg=load_config(str(Path(__file__).resolve().parents[1]/'configs'/'pilot.yaml'))
    cfg['model']['hidden']=16
    cfg['innovation']['memory_dim']=8
    cfg['retrieval']['key_dim']=16
    cfg['retrieval']['coarse_topk']=12
    cfg['retrieval']['shortlist_topk']=6
    cfg['retrieval']['topk']=3
    cfg['graph']['target_region_size']=4
    cfg['graph']['max_anchors']=4
    adj=torch.zeros(n,n)
    for j in range(n-1):adj[j,j+1]=adj[j+1,j]=1
    model=FARE(MockSTLA(n,h,p),adj,cfg)
    mem=TrainMemory(torch.randn(m,h,n).numpy(),np.ones((m,h,n),bool),
                    np.arange(m,dtype=np.int64)*26-1000,
                    latent=torch.randn(m,n,8).numpy(),
                    future=torch.randn(m,p,n).numpy(),
                    future_valid=np.ones((m,p,n),bool))
    inputs={'x_obs':torch.randn(b,h,n),'obs_mask':torch.rand(b,h,n)>.5,
      'tod':torch.zeros(b,h,dtype=torch.long),'dow':torch.zeros(b,h,dtype=torch.long),
      'window_start':torch.full((b,),5000,dtype=torch.long),
      'y_norm':torch.randn(b,p,n),'y_raw':torch.randn(b,p,n),
      'future_valid':torch.ones(b,p,n,dtype=torch.bool),
      'direct_mask':torch.rand(b,n)>.5,
      'x_full':torch.randn(b,h,n),'raw_valid':torch.ones(b,h,n,dtype=torch.bool)}
    model.refresh(mem)
    return cfg,model,mem,inputs


def test_complete_forward_and_no_memory():
    cfg,model,mem,data=fixture()
    out=model(data,mem)
    assert out['prediction'].shape==(2,12,8)
    assert out['retrieved_ids'].shape==(2,8,3)
    assert out['retrieval_evidence']['shortlist_ids'].shape==(2,8,6)
    assert out['retrieval_evidence']['probe_predictions'].shape==(2,8,6,12)
    assert torch.isfinite(out['prediction']).all()
    assert (out['retrieved_valid_candidates']).any()
    no=model(data,mem,mode='null_evidence')
    assert torch.allclose(no['prediction'],no['base_prediction'],atol=1e-6)
    assert torch.allclose(no['prediction'],model(data,mem,mode='retrieval_off')['prediction'],atol=1e-6)


def test_training_phases_have_real_gradients():
    cfg,model,mem,data=fixture()
    for phase in ('evidence','retrieval','integration','control'):
        model.set_phase(phase)
        model.zero_grad(set_to_none=True)
        out=model(data,mem)
        obj=losses(model,data,out,mem,phase,0.,1.,cfg)
        assert torch.isfinite(obj['total']),phase
        obj['total'].backward()
        gradients=[(n,p.grad.abs().sum().item()) for n,p in model.named_parameters()
                   if p.requires_grad and p.grad is not None]
        assert gradients,phase
        assert max(v for n,v in gradients)>0,phase


def test_memory_causality_and_future_label_isolation():
    cfg,model,mem,data=fixture()
    baseline=model(data,mem)['prediction'].detach()
    corrupt={**data,'y_norm':torch.randn_like(data['y_norm'])*1000,
                    'y_raw':torch.randn_like(data['y_raw'])*1000}
    assert torch.allclose(model(corrupt,mem)['prediction'],baseline,atol=1e-6)
    mem.starts[:]=data['window_start'][0]+100
    model.refresh(mem)
    out=model(data,mem)
    assert not out['retrieved_valid_candidates'].any()
    assert torch.allclose(out['prediction'],out['base_prediction'],atol=1e-6)


def test_mape_excludes_zero_and_near_zero():
    pred=np.array([[[0.,0.5,11.,18.]]],dtype=float)
    true=np.array([[[0.,0.5,10.,20.]]],dtype=float)
    valid=np.ones_like(true,dtype=bool)
    result=_metrics(pred,true,valid,np.array([[True,True,True,True]]))
    assert result['MAPE_count']==2 and result['count']==4
    assert abs(result['MAPE']-10)<1e-7
    assert result['mape_protocol']=='raw_target_percent_exclude_abs_le_1_v1'
    assert result['MAPE_coverage']==.5
    true[:]=0.
    no_eligible=_metrics(pred,true,valid,np.ones((1,4),bool))
    assert np.isnan(no_eligible['MAPE']) and no_eligible['MAPE_count']==0


def test_trainer_full_four_phase_end_to_end(tmp_path):
    """End-to-end CPU training with REAL common metric_bundle and checkpoints."""
    from torch.utils.data import Dataset
    from fare.trainer import FARETrainer
    cfg,model,mem,batch=fixture(b=4)
    cfg['training']['batch_size']=2
    cfg['training']['phase_epochs']={k:1 for k in ('evidence','retrieval','integration','control')}
    cfg['training']['patience']=2
    cfg['training']['num_workers']=0
    cfg['diagnostics']['enabled']=False
    cfg['evaluation']['save_stage_predictions']=True
    class Data(Dataset):
        family_column=None
        import pandas as pd
        meta=pd.DataFrame({"view_id":[f"view_{i}" for i in range(4)]})
        def __len__(self):return 4
        def __getitem__(self,i):
            r={k:(v[i].clone() if isinstance(v,torch.Tensor) and v.shape[0]==4 else v)
                 for k,v in batch.items()}
            r['row_id']=torch.tensor(i)
            r['artificial_mask']=~r['obs_mask']
            return r
    ds=Data()
    model.refresh(mem)
    runner=FARETrainer(model,mem,ds,ds,cfg,
            {'mean':0.,'std':1.,'dataset':'TraffiDent_Net_06','target':'speed'},
            torch.device('cpu'),tmp_path)
    final=runner.train()
    assert final.exists()
    assert len(runner.phase_history)==4
    assert runner.phase_history[-1]['val_MAPE'] is not None
    assert (tmp_path/'stages'/'control'/'val_per_query_metrics.csv').exists()
    result=json.loads((tmp_path/'stages'/'control'/'val_summary.json').read_text())
    assert result['final']['mape_protocol']=='raw_target_percent_exclude_abs_le_1_v1'
    assert 'H12_masked_MAPE' in result['final']


def test_temporal_holdout_is_chronological_and_purged():
    from fare.trainer import temporal_teacher_partition
    class WindowDataset:
        window_starts=np.arange(1000,dtype=np.int64)
        def __len__(self):return 1000
        def __getitem__(self,i):return i
    a,b,meta=temporal_teacher_partition(WindowDataset(),0.25,24)
    assert meta['enabled']
    assert len(a)>600 and len(b)>150
    assert a.indices[-1]+24<b.indices[0]
    assert meta['teacher_label_protocol'].startswith('time-held-out')


def test_report_has_mape_in_all_outputs(tmp_path):
    import json
    from scripts.report_pilot_val_test import create_report
    root=tmp_path/'evaluation_candidate'
    root.mkdir()
    for rootpath,tag in [(root,'val_full'),(root/'s40','s40_full')]:
        rootpath.mkdir(exist_ok=True)
        metrics={'MAE':1.2,'RMSE':2.3,'MAPE':12.3,'MAPE_count':75,'MAPE_coverage':.9,
            'metrics_revision':'traffident_metrics_v2',
            'mape_protocol':'raw_target_percent_exclude_abs_le_1_v1',
            'mape_min_target_abs':1.0}
        obj={'final':metrics,'base':{**metrics,'MAE':1.5,'RMSE':2.7,'MAPE':15.5},
             'candidate':metrics,'rows':10}
        (rootpath/f'{tag}_summary.json').write_text(json.dumps(obj))
    path,rows=create_report(tmp_path)
    txt=path.read_text()
    assert 'FARE MAPE' in txt and '12.30%' in txt
    assert rows[0]['final_MAPE']==12.3
    assert (tmp_path/'pilot_val_test_metrics.csv').exists()
