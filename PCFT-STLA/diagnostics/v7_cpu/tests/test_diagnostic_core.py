import json
import numpy as np
import pytest
import torch
from torch import nn
from pcft_stla.v6.graph import DirectedGraph
from pcft_stla.v7.model import PCFTV7
from diagnostics.v7_cpu.diagnostic.metrics import Accumulator,group_masks
from diagnostics.v7_cpu.diagnostic.sampling import choose_indices
from diagnostics.v7_cpu.diagnostic.collector import DiagnosticCollector
from diagnostics.v7_cpu.diagnostic.interventions import run_intervention,source_keep_mask


class TinySTLA(nn.Module):
    def __init__(self,n=4,d=8):
        super().__init__();self.num_nodes=n;self.in_steps=12;self.input_dim=2;self.steps_per_week=2016;self.model_dim=d
        self.input_proj=nn.Linear(2,d);self.attn_layers_st=nn.ModuleList([nn.Linear(d,d)])
        self.output_proj=nn.Linear(12*d,12)
    def forward(self,x):
        h=self.attn_layers_st[0](self.input_proj(x))
        b,t,n,d=h.shape
        return self.output_proj(h.transpose(1,2).reshape(b,n,t*d)).transpose(1,2)[...,None]

def create_model(n=4):
    graph=DirectedGraph(torch.tensor([[0,1,2,3],[1,2,3,0]]),torch.ones(4,10),n)
    model=PCFTV7(TinySTLA(n),graph,0,horizon=12)
    model.set_stage('joint');model.eval()
    return model

def batch(n=4,b=2):
    torch.manual_seed(123)
    x=torch.randn(b,12,n,3)
    obs=torch.ones_like(x)
    return dict(x_multi=x,observed_mask=obs,missing_age=torch.zeros_like(x),x_target=x[...,0],
       y_norm=torch.randn(b,12,n),y_raw=torch.randn(b,12,n),future_valid=torch.ones(b,12,n,dtype=torch.bool),
       direct_mask=torch.ones(b,n,dtype=torch.bool),tod=torch.zeros(b,12,dtype=torch.long),dow=torch.zeros(b,12,dtype=torch.long))

def test_diag_forward_is_identical_and_trace_shape():
    m=create_model();x=batch()
    with torch.inference_mode():
        a=m(x);b=m(x,diagnostic=True,trace_edges=True)
    for name in ('baseline','prediction','prop_delta','coverage'):
        assert torch.equal(a[name],b[name]),name
    tr=b['propagation_trace']
    assert tr['incoming_available_count'].shape==(2,12,4)
    assert tr['source_age_steps'].shape==(2,4)
    assert tr['edge_weights'].shape==(2,12,4)
    assert torch.all(tr['weighted_source_age']==0)


def test_edge_interventions_do_not_retrain_or_change_anchor():
    m=create_model();x=batch()
    with torch.inference_mode():
        original=m(x,diagnostic=True)
        keep=torch.tensor([True,False,True,False])
        altered=m(x,edge_keep_mask=keep,diagnostic=True)
    assert torch.equal(original['baseline'],altered['baseline'])
    assert torch.equal(original['local_delta'],altered['local_delta'])
    assert torch.isfinite(altered['prediction']).all()
    assert torch.all(altered['propagation_trace']['incoming_available_count']<=original['propagation_trace']['incoming_available_count'])
    assert source_keep_mask(m,x,'no_functional').shape==(4,)
    assert source_keep_mask(m,x,'fresh_only').shape==(2,4)
    with pytest.raises(ValueError):m(x,edge_keep_mask=torch.ones(5))
    for name in ('no_functional','no_physical','forward_only','reverse_only','fresh_only','skip_coverage_multiply','no_c1','no_c2','no_local','no_direction','no_null'):
        with torch.inference_mode():
            a,b,c=run_intervention(m,x,original,name)
        assert a.shape==b.shape==c.shape==x['y_norm'].shape


def test_all_missing_has_no_fake_propagation():
    m=create_model();x=batch()
    x['observed_mask'].zero_();x['x_multi'].zero_();x['x_target'].zero_()
    m.prop_head[-1].bias.data.fill_(3.)
    with torch.inference_mode():out=m(x,diagnostic=True,trace_edges=True)
    assert not out['prop_delta'].abs().any()
    assert not out['coverage'].abs().any()
    assert not out['propagation_trace']['source_available'].abs().any()


def test_prediction_does_not_use_future_target():
    m=create_model();x=batch()
    with torch.inference_mode():first=m(x)['prediction'].clone()
    x['y_norm'].fill_(9999.);x['y_raw'].fill_(9999.)
    with torch.inference_mode():second=m(x)['prediction']
    assert torch.equal(first,second)


def test_accumulator_exact_masked_counts_and_harm():
    a=Accumulator();y=np.array([[[1.,3.,np.nan]]]);v=np.array([[[True,True,True]]])
    p={'base':np.array([[[0.,2.,999.]]]),'local':np.array([[[1.,4.,999.]]]),
       'propagation':np.array([[[1.,2.,999.]]]),'controlled':np.array([[[1.,3.,999.]]])}
    a.add(y,p,v)
    r=a.finish()
    assert r['valid_positions']==2
    assert r['base_mae']==1.0 and r['controlled_mae']==0.0
    assert r['prop_harm_frequency']==0.0


def test_group_and_online_collector(tmp_path):
    m=create_model();x=batch();
    with torch.inference_mode():out=m(x,diagnostic=True,trace_edges=True)
    group=group_masks(x,dict(target_idx=0,incoming_available_count=np.zeros((2,4))))
    assert group['available_incoming_edges'].shape==(2,4)
    c=DiagnosticCollector(0.,1.,target_idx=0,mode='deep')
    c.update(x,out,np.array([4,9]),[100,200]);c.update_interventions(m,x,out,np.array([4,9]),['no_functional'])
    assert c.global_stats.finish()['valid_positions']==2*12*4
    assert c.action_rows()[0]['valid_positions']==96
    assert c.calibration.rows()
    assert c.grouped.rows()
    assert len(c.edge_rows())==4
    assert c.intervention_rows()


def test_sampling_is_deterministic():
    class DS:
        def __len__(self):return 100
    a=choose_indices(DS(),12,10)
    assert len(a)==12 and np.array_equal(a,choose_indices(DS(),12,10))
    assert len(np.unique(a))==len(a)


def test_synthetic_end_to_end_cpu_report(tmp_path,monkeypatch):
    from diagnostics.v7_cpu import run as cli
    import pandas as pd
    from pathlib import Path
    m=create_model()
    x=batch(b=2)
    class TinyDataset:
        meta=pd.DataFrame({'window_start':list(range(32)),'view_id':[f's{n}' for n in range(32)]})
        def __len__(self):return 32
        def __getitem__(self,i):return {k:v[i%2].clone() for k,v in x.items()}
    ds=TinyDataset()
    ckpt=tmp_path/'joint_best.pt';ckpt.write_bytes(b'fake cp, loader monkeypatched')
    base=tmp_path/'backbone_best.pt';base.write_bytes(b'fake backbone')
    ident={'target':'speed','dataset':'TraffiDent_Net_06','mean':0.,'std':1.,
           'backbone_sha256':'fake','graph_identity':'fake'}
    monkeypatch.setattr(cli,'load_model',lambda *a,**kw:(m,{'val':ds},ident,{},None,{}))
    # Avoid repeated setting global PyTorch inter-op threads from other test suites.
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda x:None)
    out=tmp_path/'diagnostic_run'
    cli.main(['--target','speed','--checkpoint',str(ckpt),'--backbone-checkpoint',str(base),
              '--output',str(out),'--mode','deep','--max-windows','4','--batch-size','2',
              '--interventions','no_functional','fresh_only'])
    wanted=['summary.json','run_manifest.json','sample_manifest.csv','c1_support_groups.csv',
            'c2_source_groups.csv','c2_message_statistics.csv','c3_action_statistics.csv','c3_gain_calibration.csv',
            'horizon_results.csv','intervention_results.csv','selected_cases.csv','edge_weight_summary.csv','window_metrics.csv','bootstrap_intervals.json','report.md']
    assert all((out/name).is_file() for name in wanted)
    summary=json.loads((out/'summary.json').read_text())
    assert summary['windows']==4 and summary['global']['valid_positions']==4*12*4
    assert summary['global']['base_mae']>=0
    with pytest.raises(FileExistsError):
        cli.main(['--target','speed','--checkpoint',str(ckpt),'--backbone-checkpoint',str(base),
                  '--output',str(out),'--max-windows','4'])


def test_chronological_block_bootstrap_does_not_assume_independent_windows():
    from diagnostics.v7_cpu.diagnostic.bootstrap import block_bootstrap
    rows=[dict(window_start=i*288,valid_positions=100,base_mae=2.,controlled_mae=1.9) for i in range(10)]
    result=block_bootstrap(rows,repeats=100,seed=10)
    assert result['status']=='ok'
    assert 4.99<result['improvement_pct_95ci_low']<5.01
    small=block_bootstrap(rows[:2],repeats=100)
    assert small['status']=='insufficient_independent_blocks'
