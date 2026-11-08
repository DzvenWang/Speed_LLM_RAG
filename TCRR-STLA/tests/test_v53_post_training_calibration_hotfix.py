"""Regression checks for failed calibration after finishing all 60 epochs."""
from pathlib import Path
import subprocess
import torch

from tcrr_stla.v5 import trainer as mod

ROOT=Path(__file__).resolve().parents[1]


def test_calibrate_controller_uses_only_calibration_loader(monkeypatch):
    class DummyC3:
        def __init__(self): self.calls=[]
        def set_calibration(self,*args,**kwargs):self.calls.append((args,kwargs))
    class DummyModel:
        def __init__(self): self.c3=DummyC3();self.seen=[]
        def eval(self): return self
        def __call__(self,x,mask,tod,dow,**kwargs):
            self.seen.append(int(x[0,0,0]))
            shape=(1,12,20)
            return dict(base_prediction=torch.zeros(shape),correction=torch.full(shape,.1),
                gate=torch.full(shape,.6),estimated_net_utility=torch.full(shape,-.1),
                harm_probability=torch.full(shape,.9))
    def batch(tag):
        return dict(x=torch.full((1,1,1),tag),mask=torch.ones(1,1,1),
            tod=torch.zeros(1,dtype=torch.long),dow=torch.zeros(1,dtype=torch.long),
            index=torch.tensor([0]),y_raw=torch.zeros(1,12,20),
            future_valid=torch.ones(1,12,20,dtype=torch.bool))
    class DummyShared:
        def fetch(self,index,device):return torch.zeros(1,12,20),torch.zeros(1,20,8)
    monkeypatch.setattr(mod,'_batch_to_device',lambda raw,device:raw)
    monkeypatch.setattr(mod,'evidence_for_batch',lambda *a,**kw:{})
    model=DummyModel()
    cfg={'v5':{'calibration_stride':1,'min_calibration_coverage':.55}}
    fitted=mod.calibrate_controller(model,[batch(17),batch(42)],None,None,
        torch.device('cpu'),0.,1.,cfg,DummyShared(),None,None,None)
    assert model.seen==[17,42]
    assert fitted['calibration_positions']==480
    assert model.c3.calls[0][1]=={'enabled':False}
    assert len(model.c3.calls)==2


def test_finalize_entrypoint_has_resume_and_afterok_chain():
    script=ROOT/'pbs/submit_finalize_existing_run_v5.sh'
    subprocess.run(['bash','-n',str(script)],check=True)
    s=script.read_text()
    assert 'RESUME_CHECKPOINT=$RUN_DIR/last.pt' in s
    assert 'depend=afterok:$FINALIZE_ID' in s
    assert 'RETRAIN_EPOCHS=0' in s
    assert "int(p.get('epoch',-1))!=60" in s
    assert 'postprocess_v5.pbs' in s


def test_finalize_launcher_submits_only_two_jobs(tmp_path):
    """Fake PBS verifies resume path and dependent POST without Gadi access."""
    import os, sys
    root=tmp_path/'repo'
    (root/'TCRR-STLA/pbs').mkdir(parents=True)
    for name in ('formal_v5.pbs','postprocess_v5.pbs'):
        (root/'TCRR-STLA/pbs'/name).write_text('#!/bin/bash\n')
    run=root/'TCRR-STLA_exps/speed/incomplete/TraffiDent_Net_06/v5/formal/20261028_155629_seed10_full'
    run.mkdir(parents=True)
    for name in ('best_c12.pt','best_controller.pt','s0_shared_adapter.pt'):
        (run/name).write_bytes(b'checkpoint-test-placeholder')
    torch.save(dict(version='5.3.0',stage='controller',epoch=60),run/'last.pt')
    ck=root/'Backbone_exps/speed/STLAformer/TraffiDent_Net_06/formal/old_seed10/best.pt'
    ck.parent.mkdir(parents=True)
    ck.write_bytes(b'dummy-backbone')
    bin_dir=tmp_path/'bin';bin_dir.mkdir()
    calls=tmp_path/'qsub.txt'
    qsub=bin_dir/'qsub'
    qsub.write_text('#!/bin/bash\nset -eu\necho "$*" >> "${FAKE_QSUB_CALLS}"\n'
                    'if grep -q T53_speed_fin <<< "$*"; then echo 200.gadi-pbs; '
                    'else echo 201.gadi-pbs; fi\n')
    qsub.chmod(0o755)
    env={**os.environ,'SPEED_LLM_RAG_ROOT':str(root),
         'TFP_ENV_ROOT':sys.prefix,'PATH':str(bin_dir)+':'+os.environ['PATH'],
         'FAKE_QSUB_CALLS':str(calls)}
    proc=subprocess.run(['bash',str(ROOT/'pbs/submit_finalize_existing_run_v5.sh'),str(run),str(ck)],
                        env=env,capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert 'FINALIZE_ID=200.gadi-pbs' in proc.stdout
    assert 'POSTPROCESS_ID=201.gadi-pbs' in proc.stdout
    txt=calls.read_text().splitlines()
    assert len(txt)==2
    assert 'RESUME_CHECKPOINT='+str(run/'last.pt') in txt[0]
    assert 'SKIP_EVALUATE=1,SKIP_DIAGNOSTICS=1' in txt[0]
    assert 'depend=afterok:200.gadi-pbs' in txt[1]
    assert 'mem=32GB' in txt[1]
