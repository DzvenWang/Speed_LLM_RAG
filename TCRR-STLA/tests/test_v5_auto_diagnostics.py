"""Automatic validation diagnosis must run after training with reproducible subsets."""
import csv
import importlib.util
import json
import sys
from pathlib import Path
import pytest

PROJECT=Path(__file__).resolve().parents[1]
SPEC=importlib.util.spec_from_file_location('auto_diagnose_v5_test',PROJECT/'scripts'/'auto_diagnose_v5.py')
AUTO=importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name]=AUTO
SPEC.loader.exec_module(AUTO)


def _run(tmp_path, controller=True):
    (tmp_path/'best_c12.pt').write_bytes(b'fake checkpoint')
    if controller:
        (tmp_path/'best_controller.pt').write_bytes(b'fake checkpoint')
        (tmp_path/'best_joint.pt').write_bytes(b'fake checkpoint')
        (tmp_path/'summary.json').write_text(json.dumps({'stage_selection':[
            {'stage':'controller','validation_mae':2.1},
            {'stage':'joint','validation_mae':2.0}
        ]}))
    return tmp_path


def test_plan_c3_and_c12_and_negative_controls(tmp_path):
    run=_run(tmp_path)
    jobs=AUTO.build_jobs(run,0,64)
    assert len(jobs)==12
    assert jobs[0].name=='C12_four_switches'
    assert jobs[0].modes==AUTO.MODES_C12
    assert jobs[1].name=='C3_eight_switches'
    assert jobs[1].checkpoint.name=='best_joint.pt'
    assert jobs[1].modes==AUTO.MODES_C3
    assert all(j.checkpoint.name=='best_c12.pt' and j.max_batches==64 for j in jobs if j.name.startswith('negative_'))
    assert ('wrong_mask','correct') in [(j.cf_control,j.support_control) for j in jobs if j.name.startswith('negative_')]


def test_without_c3_checkpoint_still_runs_c12_negatives(tmp_path):
    jobs=AUTO.build_jobs(_run(tmp_path,False),12,10)
    assert all(j.controller_mode=='fixed' for j in jobs)
    assert len(jobs)==6
    assert jobs[0].max_batches==12


def test_command_never_saves_predictions_or_touches_test(tmp_path):
    j=next(j for j in AUTO.build_jobs(_run(tmp_path),0,64) if j.name.startswith('negative_'))
    c=AUTO.command(j,8)
    assert c[c.index('--split')+1]=='validation'
    assert '--save-predictions' not in c
    assert c[c.index('--max-batches')+1]=='64'
    assert c[c.index('--output-tag')+1]=='negative_controls'


def test_diagnostic_summary_uses_same_budget(tmp_path):
    run=_run(tmp_path)
    jobs=AUTO.build_jobs(run,0,8)
    for j in jobs:
        path=AUTO.result_dir(run,j)
        path.mkdir(parents=True,exist_ok=True)
        with (path/'mechanism_v5.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['mode','MAE','RMSE','evaluated_queries','full_split_queries','sample_index_sha256','wrong_mask_effective_rate'])
            writer.writeheader()
            for m in j.modes:
                mae=5 if j.cf_control=='correct' and j.support_control=='correct' else 6
                writer.writerow({'mode':m,'MAE':mae,'RMSE':10+mae,'evaluated_queries':512 if j.max_batches else 5000,'full_split_queries':5000,'sample_index_sha256':'same-sample-hash','wrong_mask_effective_rate':1.0})
    summary=AUTO.summarize(run,jobs)
    assert len(summary['negative_controls'])==4
    assert all(x['MAE_gap_control_minus_correct']==1 for x in summary['negative_controls'])
    assert (run/'evaluation/validation/automated/negative_control_comparison.csv').is_file()


def test_fail_on_missing_c12(tmp_path):
    with pytest.raises(FileNotFoundError):
        AUTO.build_jobs(tmp_path,0,64)


def test_wrong_sample_identity_is_rejected(tmp_path):
    run=_run(tmp_path)
    jobs=AUTO.build_jobs(run,0,2)
    for j in jobs:
        folder=AUTO.result_dir(run,j)
        folder.mkdir(parents=True,exist_ok=True)
        with (folder/'mechanism_v5.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=['mode','MAE','RMSE','evaluated_queries','full_split_queries','sample_index_sha256','wrong_mask_effective_rate'])
            writer.writeheader()
            for m in j.modes:
                hash_id=('wrong-sample-hash' if j.cf_control=='wrong_mask' else 'correct-sample-hash')
                writer.writerow({'mode':m,'MAE':1.0,'RMSE':2.0,'evaluated_queries':16,
                                 'full_split_queries':5000,'sample_index_sha256':hash_id})
    with pytest.raises(RuntimeError,match='exactly the same sample'):
        AUTO.summarize(run,jobs)


def test_diagnostic_failure_propagates_and_records_status(monkeypatch,tmp_path):
    import io
    run=_run(tmp_path,False)
    jobs=AUTO.build_jobs(run,1,0)

    class BrokenProcess:
        stdout=io.StringIO('simulated diagnostic error\n')
        def wait(self):
            return 9
    monkeypatch.setattr(AUTO.subprocess,'Popen',lambda *a,**k:BrokenProcess())
    with pytest.raises(__import__('subprocess').CalledProcessError):
        AUTO.run_jobs(run,jobs,batch_size=2)
    status=json.loads((run/'evaluation/validation/automated/job_status.json').read_text())
    assert len(status)==1 and status[0]['success'] is False
    assert status[0]['returncode']==9
