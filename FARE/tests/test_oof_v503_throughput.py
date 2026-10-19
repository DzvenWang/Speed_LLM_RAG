"""Defaults and argument propagation for four independent V100 producer jobs."""
import importlib.util
import sys
from pathlib import Path

from fare.config import read_config

ROOT = Path(__file__).resolve().parents[1]


def _submit_module():
    path=ROOT/'scripts'/'submit_oof.py'
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('v503_submit',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_version_and_preserved_formal_batch():
    cfg=read_config(str(ROOT/'configs'/'formal.yaml'))
    assert cfg['version']=='5.0.3'
    assert cfg['training']['batch_size']==32
    assert cfg['memory']['require_oos_for_formal']


def test_pbs_four_independent_gpu_jobs_and_worker_arguments():
    f=(ROOT/'pbs'/'oof_fold.pbs').read_text()
    assert '#PBS -l ncpus=12,ngpus=1,mem=32GB' in f
    assert '--fold-id "$OOF_FOLD_ID"' in f
    assert '--workers "${OOF_WORKERS:-4}"' in f
    assert '--batch-size "${OOF_BATCH_SIZE:-64}"' in f
    assert 'FARE_cache/oof_v503' in f
    assert 'set -euo pipefail' in f
    final=(ROOT/'pbs'/'oof_finalize.pbs').read_text()
    assert '--finalize-only' in final
    assert '--workers "${OOF_WORKERS:-4}"' in final
    assert (ROOT/'pbs'/'full_pipeline.pbs').read_text().count('ngpus=1')==1


def test_submit_passes_workers_in_plan_env_and_receipt(tmp_path,monkeypatch):
    sub=_submit_module()
    fake_cfg=tmp_path/'formal.yaml';fake_cfg.write_text('version: 5.0.3')
    monkeypatch.setattr(sub,'root',lambda:tmp_path)
    monkeypatch.setattr(sub,'project_path',lambda p,must_exist=False:Path(p) if Path(p).is_absolute() else tmp_path/p)
    monkeypatch.setattr(sub,'read_config',lambda path,dataset,target,seed:{'experiment':{'run_tag':'formal'},'evaluation':{'include_val':False}})
    monkeypatch.setattr(sub,'backbone_checkpoint',lambda cfg,override:tmp_path/'base.pt')
    events=[]
    monkeypatch.setattr(sub.subprocess,'check_call',lambda cmd:events.append(('plan',cmd)))
    def qsub(cmd,text):
        events.append(('pbs',cmd))
        return f'{1_815_000+sum(x[0]=="pbs" for x in events)}.gadi-pbs\n'
    monkeypatch.setattr(sub.subprocess,'check_output',qsub)
    monkeypatch.setattr(sys,'argv',['submit_oof.py','--target','both','--nfolds','2','--parallel-folds','--auto-fare'])
    sub.main()
    assert events[0][0]=='plan'
    assert events[0][1][events[0][1].index('--workers')+1]=='4'
    assert events[0][1][events[0][1].index('--batch-size')+1]=='64'
    jobs=[cmd for tag,cmd in events if tag=='pbs']
    assert len(jobs)==7
    assert all('OOF_BATCH_SIZE=64' in ' '.join(c) and 'OOF_WORKERS=4' in ' '.join(c) for c in jobs[:5])
    assert all('oof_fold.pbs' in c[-1] for c in jobs[:4])
    assert jobs[4][jobs[4].index('-W')+1].count('gadi-pbs')==4
    assert all('full_pipeline.pbs' in jobs[i][-1] for i in [5,6])
    assert all('depend=afterok:' in jobs[i][jobs[i].index('-W')+1] for i in [4,5,6])
    receipts=list((tmp_path/'FARE_exps'/'oof_submissions').glob('*.json'))
    assert len(receipts)==1
    import json
    doc=json.loads(receipts[0].read_text())
    assert doc['plan']['workers']==4
    assert doc['plan']['batch_size']==64
    assert doc['version']=='5.0.3'


def test_dataloader_is_persistent_and_prefetched():
    s=(ROOT/'fare'/'oof.py').read_text()
    assert 'persistent_workers=workers>0' in s
    assert "'prefetch_factor':2" in s
    assert 'peak_allocated_gb' in s
