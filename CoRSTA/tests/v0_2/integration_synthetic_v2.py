#!/usr/bin/env python3
"""End-to-end CPU-only mock STLA / mock Backbone cache; NO research claims."""
from __future__ import annotations
import os,sys,tempfile,subprocess,hashlib,textwrap
from pathlib import Path
import numpy as np, torch
SRC=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(SRC/'tests'))
from integration_synthetic_corsta import VENDOR,make_fake_target
sys.path.insert(0,str(SRC))
from corsta.stla import load_pretrained_stla
from corsta.student import SplitSTLA
from corsta_v2.util import sha256


def synthetic_project(target):
    root=Path(tempfile.mkdtemp(prefix='corsta_v2_mock_'))
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
    rng=np.random.default_rng(56)
    np.save(formal/'val_x_full_norm.npy',rng.normal(size=(64,12,9)).astype('float32'))
    base,_=load_pretrained_stla(root,ck,torch.device('cpu'))
    student=SplitSTLA(base,tail_layers=2)
    s0=root/'CoRSTA_exps'/target/'incomplete'/'TraffiDent_Net_06'/'formal'/'seed10'/'bootstrap_v7'/'s0.pt'
    s0.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'backbone_sha256':sha256(ck),'mask_proj':student.mask_proj.state_dict()},s0)
    return root,ck,s0


def main():
    for target in ('speed','flow'):
        root,ck,s0=synthetic_project(target)
        os.environ['SPEED_LLM_RAG_ROOT']=str(root)
        os.environ['BACKBONE_MODE']='incomplete'
        env=os.environ.copy()
        env.update(SPEED_LLM_RAG_ROOT=str(root),BACKBONE_MODE='incomplete',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
            PYTHONPATH=str(SRC)+os.pathsep+str(root/'Backbone'))
        base=[sys.executable,'-u',str(SRC/'scripts/v0_2/run.py'),
              '--target',target,'--dataset','TraffiDent_Net_06',
              '--seed','10','--backbone-checkpoint',str(ck),'--mask-proj-checkpoint',str(s0),
              '--config',str(SRC/'configs/v0_2/corsta_v2.yaml'),
              '--cpu','--smoke','--limit-train','24','--limit-val','48','--limit-test','8','--workers','0']
        for stage in ('preflight','all','all'):
            cmd=base+['--stage',stage]
            process=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=240)
            print('TARGET',target,'STAGE',stage,'STATUS',process.returncode)
            if process.returncode:
                print(process.stdout[-5000:]);print(process.stderr[-5000:]);raise RuntimeError('Synthetic v2 stage failed; root='+str(root))
            assert all(line.startswith('[') for line in process.stdout.splitlines())
            assert all(line.startswith('[') for line in process.stderr.splitlines())
        run=root/f'CoRSTA_v021_exps/{target}/incomplete/TraffiDent_Net_06/smoke/seed10'
        assert (run/'summary.json').exists()
        for s in ('cf','residual','joint','control'):
            assert (run/s/'COMPLETE.json').exists()
            assert (run/s/'best.pt').exists()
        assert (run/'model_lock.json').exists()
        print('CORSTA_V2_SYNTHETIC_END_TO_END_PASS target='+target+' root='+str(root))

if __name__=='__main__':main()
