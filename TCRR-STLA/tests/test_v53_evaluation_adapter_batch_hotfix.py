"""Real S0 identity must be independent of the evaluation loader batch size.

Regression for Gadi POSTPROCESS jobs 182502729 / 182503074: formal S0
was trained with batch=64 but validation diagnostics use batch=8. The
saved training config may not be mutated to the diagnostic batch size.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

from scripts import evaluate_v5
from tcrr_stla.v5 import shared_adapter


def _config():
    return {
        'version': '5.3.0',
        'project': {'root': '/tmp/mock-research-root'},
        'experiment': {'target': 'speed', 'dataset': 'TraffiDent_Net_06'},
        'training': {'batch_size': 64},
        'v5': {'s0_lr': 3e-4, 's0_epochs': 8},
        'runtime': {'num_workers': 4},
    }


class _DummyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_proj = torch.nn.Linear(2, 2)


def _adapter_payload(cfg):
    return {
        'version': '5.3.0',
        'target': 'speed',
        'dataset': 'TraffiDent_Net_06',
        'backbone_sha256': 'synthetic-backbone-sha',
        'data_identity': {'data': 'synthetic-data-sha'},
        's0_identity': shared_adapter.s0_identity(cfg),
        'smoke': False,
        'mask_proj': _DummyTeacher().mask_proj.state_dict(),
    }


@pytest.fixture
def synthetic_s0(monkeypatch, tmp_path):
    config = _config()
    ckpt = tmp_path / 'best_c12.pt'
    adapter = tmp_path / 's0_shared_adapter.pt'
    backbone = tmp_path / 'backbone-best.pt'
    backbone.write_bytes(b'backbone-synthetic')
    adapter_payload = _adapter_payload(config)
    torch.save(adapter_payload, adapter)
    checkpoint = {
        'version': '5.3.0', 'config': config,
        'backbone_checkpoint': str(backbone),
        'backbone_sha256': 'synthetic-backbone-sha',
        'adapter_sha256': 'synthetic-adapter-sha',
    }
    torch.save(checkpoint, ckpt)
    monkeypatch.setattr(shared_adapter, 'load_pretrained_stla', lambda *args: (object(), {}))
    monkeypatch.setattr(shared_adapter, 'MaskAwareSTLAformer', lambda base: _DummyTeacher())
    monkeypatch.setattr(shared_adapter, 'file_sha256', lambda path: 'synthetic-backbone-sha')
    monkeypatch.setattr(shared_adapter, 'adapter_data_identity', lambda cfg: {'data': 'synthetic-data-sha'})
    return config, ckpt, adapter, backbone


@pytest.mark.parametrize('evaluation_batch_size', [1, 8, 32, 64, 128])
def test_evaluation_preserves_formal_training_batch_and_loads_real_s0(
        synthetic_s0, monkeypatch, capsys, evaluation_batch_size):
    original_cfg, ckpt, adapter, backbone = synthetic_s0

    class _StopAfterS0Validation(Exception):
        pass

    original_torch_load = torch.load
    # Let the checkpoint and adapter be truly loaded by torch, then execute
    # the real S0 fingerprint comparison in initial_model.
    monkeypatch.setattr(evaluate_v5, 'file_sha256',
                        lambda path: 'synthetic-backbone-sha' if str(path) == str(backbone)
                        else 'synthetic-adapter-sha')

    def verify_s0_then_stop(cfg, ck_path, device, adapter_path):
        assert cfg['training']['batch_size'] == 64
        assert cfg['runtime']['num_workers'] == 0
        shared_adapter.initial_model(cfg, ck_path, device, adapter_path)
        raise _StopAfterS0Validation()

    monkeypatch.setattr(evaluate_v5, 'build_model', verify_s0_then_stop)
    monkeypatch.setattr(sys, 'argv', [
        'evaluate_v5.py', '--checkpoint', str(ckpt),
        '--batch-size', str(evaluation_batch_size), '--split', 'validation',
        '--validation-section', 'check', '--diagnostics',
    ])
    with pytest.raises(_StopAfterS0Validation):
        evaluate_v5.main()
    stdout = capsys.readouterr().out
    assert 'V5_EVAL_BATCH_CONFIG' in stdout
    assert f'"evaluation_batch_size": {evaluation_batch_size}' in stdout
    assert '"training_batch_size": 64' in stdout
    assert original_cfg['training']['batch_size'] == 64


def test_s0_training_batch_fingerprint_still_rejects_real_mismatch(synthetic_s0):
    cfg, _, adapter, backbone = synthetic_s0
    verified, _ = shared_adapter.initial_model(cfg, backbone, torch.device('cpu'), adapter)
    assert isinstance(verified, _DummyTeacher)
    wrong = copy.deepcopy(cfg)
    wrong['training']['batch_size'] = 8
    with pytest.raises(ValueError, match='S0 adapter training settings mismatch'):
        shared_adapter.initial_model(wrong, backbone, torch.device('cpu'), adapter)


def test_s0_data_identity_protection_not_removed(synthetic_s0, monkeypatch):
    cfg, _, adapter, backbone = synthetic_s0
    monkeypatch.setattr(shared_adapter, 'adapter_data_identity', lambda cfg: {'data': 'changed-data-sha'})
    with pytest.raises(ValueError, match='data/mask protocol fingerprint mismatch'):
        shared_adapter.initial_model(cfg, backbone, torch.device('cpu'), adapter)


def test_evaluation_rejects_zero_batch_size(synthetic_s0, monkeypatch):
    _, ckpt, _, _ = synthetic_s0
    monkeypatch.setattr(sys, 'argv', [
        'evaluate_v5.py', '--checkpoint', str(ckpt), '--batch-size', '0',
    ])
    with pytest.raises(SystemExit) as exc:
        evaluate_v5.main()
    assert exc.value.code == 2

@pytest.mark.parametrize('target', ['speed','flow'])
def test_post_only_launcher_preflights_and_queues_one_job(monkeypatch, tmp_path, target):
    import hashlib
    import json
    import os
    import shutil
    import subprocess

    source = Path(__file__).resolve().parents[1]
    root = tmp_path / 'research'
    pbs = root / 'TCRR-STLA/pbs'
    pbs.mkdir(parents=True)
    shutil.copy2(source/'pbs/submit_postprocess_existing_v5.sh',
                 pbs/'submit_postprocess_existing_v5.sh')
    (pbs/'postprocess_v5.pbs').write_text('#!/bin/bash\n')
    run = root/'TCRR-STLA_exps'/target/'incomplete/TraffiDent_Net_06/v5/formal/run'
    run.mkdir(parents=True)
    cfg = _config()
    cfg['experiment']['target'] = target
    a = _adapter_payload(cfg)
    a['target'] = target
    adapter = run/'s0_shared_adapter.pt'
    torch.save(a,adapter)
    a_sha = hashlib.sha256(adapter.read_bytes()).hexdigest()
    for name,stage in [('best_c12.pt','c12'),('best_controller.pt','controller')]:
        torch.save({'version':'5.3.0', 'stage':stage,
                    'adapter_sha256':a_sha, 'config':copy.deepcopy(cfg)}, run/name)
    (run/'summary.json').write_text('{}')
    (run/'validation_partitions.json').write_text('{}')
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    calls=tmp_path/'qsub.txt'
    qsub=bin_dir/'qsub'
    qsub.write_text('#!/bin/bash\necho "$*" >> "$TRACE"\necho 98765.gadi-pbs\n')
    qsub.chmod(0o755)
    env={**os.environ,'PATH':str(bin_dir)+os.pathsep+os.environ['PATH'],
         'TRACE':str(calls),'TFP_ENV_ROOT':sys.prefix,
         'TCRR_POST_PYTHON':sys.executable,'PYTHONPATH':str(source)}
    cmd=['bash',str(pbs/'submit_postprocess_existing_v5.sh'),str(run)]
    good=subprocess.run(cmd,capture_output=True,text=True,env=env)
    assert good.returncode==0,good.stderr
    assert good.stdout.count('POSTPROCESS_ID=98765.gadi-pbs')==1
    assert 'RETRAIN_EPOCHS=0' in good.stdout
    submitted=calls.read_text()
    assert f'T53_{target}_pp' in submitted
    assert 'mem=32GB' in submitted
    assert 'ngpus=1' in submitted
    assert f'RUN_DIR={run}' in submitted
    assert 'postprocess_v5.pbs' in submitted
    # A genuine change of the original S0 training identity must fail closed.
    bad_cfg=copy.deepcopy(cfg)
    bad_cfg['training']['batch_size']=8
    torch.save({'version':'5.3.0','stage':'c12',
                'adapter_sha256':a_sha,'config':bad_cfg},run/'best_c12.pt')
    calls.unlink()
    rejected=subprocess.run(cmd,capture_output=True,text=True,env=env)
    assert rejected.returncode!=0
    assert 'S0 formal training identity mismatch' in rejected.stderr
    assert not calls.exists()
