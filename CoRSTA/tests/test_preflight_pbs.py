"""CPU preflight integration using a synthetic STLA checkpoint; no GPU needed."""
from __future__ import annotations
import importlib.util,os,subprocess,sys
from pathlib import Path

PROJ=Path(__file__).resolve().parents[1]


def _synthetic_setup(tmp_path):
    spec=importlib.util.spec_from_file_location('corsta_fake_integration',PROJ/'tests/integration_synthetic_corsta.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vendor=tmp_path/'Backbone/vendor/STLAformer/model/STLformer.py'
    vendor.parent.mkdir(parents=True,exist_ok=True)
    vendor.write_text(module.VENDOR)
    checkpoint=module.make_fake_target(tmp_path,'speed',n=5,train_count=8,val_count=4,test_count=4)
    (tmp_path/'Base_SupportBench').mkdir()
    env=os.environ.copy()
    env.update(SPEED_LLM_RAG_ROOT=str(tmp_path),BACKBONE_MODE='incomplete',
               OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
               PYTHONPATH=str(PROJ)+os.pathsep+str(tmp_path/'Backbone'))
    args=[sys.executable,'-u',str(PROJ/'scripts/run.py'),'--stage','preflight',
          '--target','speed','--dataset','TraffiDent_Net_06','--seed','10',
          '--backbone-checkpoint',str(checkpoint)]
    return env,args


def test_cpu_preflight_without_cuda_or_formal_identity(tmp_path):
    env,args=_synthetic_setup(tmp_path)
    result=subprocess.run(args+['--cpu'],env=env,capture_output=True,text=True,timeout=90)
    assert result.returncode==0, result.stdout+result.stderr
    assert 'CORSTA_PREFLIGHT_PASS' in result.stdout
    assert 'CORSTA_PREFLIGHT_COMPLETE result=pass' in result.stdout
    assert 'device=cpu' in result.stdout
    assert not list((tmp_path/'CoRSTA_exps').rglob('run_identity.json'))
    assert not list((tmp_path/'CoRSTA_preflight').rglob('run_identity.json'))
    assert all(line.startswith('[') for line in result.stdout.splitlines())


def test_preflight_cpu_gpu_flags_conflict(tmp_path):
    env,args=_synthetic_setup(tmp_path)
    result=subprocess.run(args+['--cpu','--require-cuda'],env=env,capture_output=True,text=True,timeout=60)
    assert result.returncode!=0
    assert 'mutually exclusive' in result.stdout+result.stderr


def test_cpu_and_gpu_pbs_are_distinct():
    cpu=(PROJ/'pbs/preflight_cpu.pbs').read_text()
    gpu=(PROJ/'pbs/preflight_gpu.pbs').read_text()
    assert '#PBS -q normal' in cpu and '#PBS -l ngpus=' not in cpu
    assert '#PBS -q gpuvolta' in gpu and '#PBS -l ngpus=1' in gpu
    job=(PROJ/'pbs/preflight_job.sh').read_text()
    assert '--cpu' in job and '--require-cuda' in job
    assert 'timestamp_filter.py' in job
