"""Version 2.0.2 telemetry regression tests (no Gadi hardware required)."""
import importlib.util
import json
import tempfile
import time
from pathlib import Path

import torch

from fare.online_diagnostics import OnlineDiagnostics, append_diagnostics

SCRIPT = Path(__file__).resolve().parents[1]/'scripts'

def load_script(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPT/f'{name}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sample_batch():
    batch={'x_obs':torch.zeros(2,2,3),
           'obs_mask':torch.tensor([[[True,False,True],[True,True,False]],
                                    [[True,True,False],[False,True,True]]]),
           'y_raw':torch.zeros(2,2,3),
           'future_valid':torch.ones(2,2,3,dtype=torch.bool),
           'direct_mask':torch.tensor([[True,False,False],[False,True,False]])}
    base=torch.ones(2,2,3)
    candidate=base.clone();candidate[:,:,0]=0
    final=(base+candidate)/2
    output={'base_prediction':base,'candidate_prediction':candidate,'prediction':final,
            'final_gate':torch.full_like(base,.5),
            'retrieval_valid':torch.tensor([[True,False],[True,True]]),
            'retrieved_valid_candidates':torch.ones(2,2,3,dtype=torch.bool),
            'retrieved_weights':torch.tensor([[[.6,.3,.1],[0.,0.,0.]],
                                              [[.3,.4,.3],[1.,0.,0.]]]),
            'state_uncertainty':torch.full_like(batch['x_obs'],.2)}
    return batch,output


def test_online_diagnostics_mae_gain_and_mask_groups():
    batch,out=sample_batch()
    meter=OnlineDiagnostics(every_batches=2,max_batches=2)
    assert meter.should_sample(0)
    assert not meter.should_sample(1)
    meter.update(batch,out,0,1,torch.tensor(2.5))
    meter.update(batch,out,0,1)
    assert not meter.should_sample(4)
    row=meter.summary()
    assert row['sampled_batches']==2
    assert abs(row['base_MAE']-1)<1e-6
    assert abs(row['candidate_MAE']-(2/3))<1e-6
    assert abs(row['final_MAE']-(5/6))<1e-6
    assert abs(row['candidate_MAE_gain']-(1/3))<1e-6
    assert abs(row['final_MAE_gain']-(1/6))<1e-6
    assert abs(row['candidate_helpful_fraction']-(1/3))<1e-6
    assert abs(row['gate_mean']-.5)<1e-6
    assert row['retrieval_valid_group_fraction']==.75
    assert row['retrieval_effective_topk']>1


def test_diagnostics_jsonl_csv_and_report_partial_data():
    monitor=load_script('resource_monitor')
    # report_telemetry imports resource_monitor like its normal script path.
    import sys
    sys.path.insert(0,str(SCRIPT))
    try: reporter=load_script('report_telemetry')
    finally:sys.path.remove(str(SCRIPT))
    with tempfile.TemporaryDirectory() as dirname:
        run=Path(dirname)/'run'
        stamp=time.time()
        resource={'utc_start':__import__('datetime').datetime.fromtimestamp(stamp,__import__('datetime').timezone.utc).isoformat(),
                  'utc_end':__import__('datetime').datetime.fromtimestamp(stamp+60,__import__('datetime').timezone.utc).isoformat(),
                  'train_samples_per_second':7.5}
        append_diagnostics(run,'full',1,{'sampled_batches':1},{'candidate_MAE_gain':-1.0,'final_MAE_gain':.2,'final_harmful_fraction':.3},resource)
        append_diagnostics(run,'recovery',1,{'retrieval_valid_group_fraction':1.0},{'sampled_batches':1},resource)
        prefix=Path(dirname)/'resources'
        samples=[{'unix_time':stamp+10,'elapsed_seconds':10.,'gpu_matched':True,
                  'gpu_util_pct':45.,'gpu_memory_used_mb':2048.,'cpu_cores':3.0,'process_rss_gb':2.0},
                 {'unix_time':stamp+30,'elapsed_seconds':30.,'gpu_matched':True,
                  'gpu_util_pct':55.,'gpu_memory_used_mb':4096.,'cpu_cores':4.0,'process_rss_gb':3.0}]
        Path(f'{prefix}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in samples)+'{incomplete',encoding='utf-8')
        result=reporter.report(run,str(prefix))
        assert result['epoch_count']==2
        assert result['per_epoch'][0]['gpu_util_mean_pct']==50
        assert result['per_epoch'][0]['gpu_memory_peak_mb']==4096
        assert (run/'diagnostics'/'telemetry_report.md').exists()
        assert 'candidate_MAE_gain' in (run/'diagnostics'/'epoch_diagnostics.csv').read_text()
        log=Path(dirname)/'live.log'
        log.write_text(f'[timestamp] FARE_TRAIN_PASS run_dir={run} candidate_checkpoint={run}/best.pt\n')
        assert reporter.resolve_run_dir(log)==run


def test_job_scoped_gpu_attribution():
    m=load_script('resource_monitor')
    text='0, GPU-a, 20, 5, 1024, 16384, 70, 45\n1, GPU-b, 85, 75, 8192, 16384, 200, 71'
    devices=m._parse_gpu_table(text)
    assert len(devices)==2
    selected=m._parse_gpu_apps('100, GPU-b, 5120\n101, GPU-a, 1024',{100})
    assert selected=={'GPU-b':5120.}
    assert [dev['index'] for dev in devices if dev['uuid'] in selected]==['1']
    assert m.summarize([{'gpu_matched':True,'gpu_util_pct':20.,'elapsed_seconds':2.},
                        {'gpu_matched':False,'gpu_util_pct':None,'elapsed_seconds':3.}])['gpu_util_pct']['mean']==20.


def test_pbs_background_sampler_survives_auto_evaluation():
    pbs=(SCRIPT.parent/'pbs'/'train_formal.pbs').read_text()
    assert 'resource_monitor.py' in pbs
    assert 'trap cleanup_monitor EXIT' in pbs
    assert 'FARE_RESOURCE_INTERVAL' in pbs
    assert 'report_telemetry.py' in pbs
    assert 'FARE_DIAG_INTERVAL' in pbs
