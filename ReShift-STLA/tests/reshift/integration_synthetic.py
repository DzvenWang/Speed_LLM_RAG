#!/usr/bin/env python3
"""Isolated CPU synthetic end-to-end run of the complete ReShift pipeline.

Fake STLA/Backbone and fake caches (NO forecasting claims).  Per target:
  1. reshift_full:  all  (s0 -> bank -> normal -> anchor_cf -> anchor_ret -> anchor_control
                          -> lock -> test -> audit -> report), then m0 and m1
  2. reshift_no_retrieval with the SAME shared root: must REUSE s0/normal/anchor_cf
Usage: python tests/reshift/integration_synthetic.py --target both
"""
from __future__ import annotations
import argparse,json,os,subprocess,sys,tempfile,textwrap
from pathlib import Path
SRC=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SRC/'tests'));sys.path.insert(0,str(SRC))
import numpy as np


def synthetic_project(target):
    from integration_synthetic_corsta import VENDOR,make_fake_target
    root=Path(tempfile.mkdtemp(prefix='reshift_mock_'))
    vendor=root/'Backbone/vendor/STLAformer/model/STLformer.py'
    vendor.parent.mkdir(parents=True,exist_ok=True);vendor.write_text(VENDOR)
    pkg=root/'Backbone/backbone';pkg.mkdir(parents=True)
    (pkg/'__init__.py').write_text('')
    (pkg/'utils.py').write_text('def load_yaml(path):return {}\n')
    (pkg/'shared_cache.py').write_text(textwrap.dedent('''
        from pathlib import Path
        import os
        def cache_path(cfg,stage,severity=None):
            root=Path(os.environ['SPEED_LLM_RAG_ROOT'])
            target=os.environ['TRAFFIDENT_TARGET']
            return root/'fake_cache'/target/('formal' if stage=='formal' else 'evaluation_s'+str(severity))
        def validate_shared_cache(cfg,stage,severity=None):
            assert cache_path(cfg,stage,severity).exists()
    '''))
    (root/'Backbone/configs').mkdir(parents=True)
    (root/'Backbone/configs/backbone.yaml').write_text('{}\n')
    (root/'Base_SupportBench').mkdir()
    ck=make_fake_target(root,target,n=9,train_count=40,val_count=64,test_count=8)
    formal=root/'fake_cache'/target/'formal'
    np.save(formal/'val_x_full_norm.npy',np.random.default_rng(56).normal(size=(64,12,9)).astype('float32'))
    return root,ck


def run(root,ck,target,config,stage,shared,extra=()):
    env=os.environ.copy()
    env.update(SPEED_LLM_RAG_ROOT=str(root),BACKBONE_MODE='incomplete',OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(SRC)+os.pathsep+str(root/'Backbone'))
    cmd=[sys.executable,'-u',str(SRC/'scripts/reshift/run.py'),'--target',target,'--dataset','TraffiDent_Net_06',
         '--seed','10','--backbone-checkpoint',str(ck),'--config',str(SRC/'configs/reshift'/config),
         '--shared-root',str(shared),'--cpu','--smoke','--limit-train','32','--limit-val','48',
         '--limit-test','4','--workers','0','--stage',stage,*extra]
    r=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=900)
    print(f'RESHIFT_SYNTH target={target} config={config} stage={stage} status={r.returncode}',flush=True)
    if r.returncode:
        print(r.stdout[-8000:],r.stderr[-4000:],flush=True)
        raise RuntimeError(f'ReShift synthetic run failed (root={root})')
    return r.stdout


def per_target(target):
    root,ck=synthetic_project(target)
    shared=root/'ReShift_exps/shared_test'
    out=run(root,ck,target,'reshift_full.yaml','all',shared)
    assert 'RESHIFT_S0_PASS' in out and 'RESHIFT_BANK_PASS' in out and 'RESHIFT_P_FULL_PASS' in out
    runp=root/f'ReShift_exps/{target}/incomplete/TraffiDent_Net_06/smoke/seed10/reshift_full'
    for st in ('anchor_ret','anchor_control'):
        assert (runp/st/'COMPLETE.json').exists() and (runp/st/'best.pt').exists(),st
    assert any(p.name.startswith('normal-') for p in shared.iterdir())
    assert (runp/'model_lock.json').exists() and (runp/'summary.json').exists() and (runp/'RESULTS.md').exists()
    for sev in (20,40,60,90):
        d=json.loads((runp/'test'/f's{sev}.json').read_text())['metric']['per_severity'][str(sev)]['cf_diagnostics']
        assert 'arms' in d and 'retrieval_harm_bins' in d,f'missing ReShift diagnostics at s{sev}'
        a=json.loads((runp/'audit'/f's{sev}.json').read_text())
        assert set(a['relative_mae_change_vs_full'])=={'full','shuffled','cross_history','no_replay','no_retrieval'}
    for stage,name in (('m0','m0.json'),('m1','m1.json')):
        run(root,ck,target,'reshift_full.yaml',stage,shared)
        res=json.loads((runp/'diagnostics'/name).read_text())
        assert set(res)=={'20','40','60','90'},name
    out2=run(root,ck,target,'reshift_no_retrieval.yaml','all',shared)
    for st in ('normal','anchor_cf'):
        assert f'RESHIFT_STAGE_REUSE stage={st}' in out2,f'{st} was not shared'
    assert 'RESHIFT_S0_PASS' not in out2 and 'RESHIFT_AUDIT_SKIP' in out2
    print('RESHIFT_SYNTHETIC_E2E_PASS',target,root,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=('speed','flow','both'),default='both')
    a=p.parse_args()
    if a.target=='both':
        for t in ('speed','flow'):
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--target',t],check=True,timeout=3600)
    else:per_target(a.target)


if __name__=='__main__':main()
