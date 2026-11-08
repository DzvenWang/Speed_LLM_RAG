"""Strict inference equivalence and actual C1/C2 recomputation regression tests."""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pytest
import torch
from test_v5 import model,data
from tcrr_stla.v5.model import THREE_SWITCH_MODES
from tcrr_stla.replay_cache import prepare_bank


@pytest.mark.parametrize('modes,controller',[
    (('M00','M10','M01','M11'),'fixed'),
    (tuple(THREE_SWITCH_MODES),'auto'),
    (('M110','M111'),'auto'),
])
def test_exact_predictions_from_shared_inference(modes,controller):
    torch.manual_seed(232)
    net=model(n=13);net.eval();net.set_stage('controller')
    sample=data(n=13,k=5)
    with torch.no_grad():
        separate={key:net(**sample,mode=key,controller_mode=controller)
                  for key in modes}
        together=net.forward_modes(**sample,modes=modes,controller_mode=controller)
    for key in modes:
        for field in ('prediction','proposed_correction','candidate_weights',
                      'normal_gate','cf_gate','candidate_marginal_utility_pred',
                      'proposed_normal','proposed_cf','estimated_net_utility',
                      'harm_probability','gate'):
            assert torch.allclose(separate[key][field],together[key][field],
                                  atol=1e-6,rtol=1e-6),(key,field)


def test_shared_inference_really_reuses_c1_and_c2():
    net=model();net.eval();net.set_stage('controller')
    sample=data()
    with patch.object(net.c1,'forward',wraps=net.c1.forward) as c1,\
         patch.object(net.c2,'forward',wraps=net.c2.forward) as c2,\
         patch.object(net.support_encoder,'forward',wraps=net.support_encoder.forward) as support:
        with torch.no_grad():
            net.forward_modes(**sample,modes=tuple(THREE_SWITCH_MODES))
        assert c1.call_count==2
        assert c2.call_count==4
        assert support.call_count==1


def test_no_shared_mode_forward_with_gradients_or_during_training():
    net=model();sample=data()
    with pytest.raises(RuntimeError,match='inference-only'):
        net.forward_modes(**sample,modes=['M111'])
    net.eval()
    with pytest.raises(RuntimeError,match='inference-only'):
        net.forward_modes(**sample,modes=['M111'])
    with torch.no_grad():
        assert 'M111' in net.forward_modes(**sample,modes=['M111'])


def test_shared_inference_handles_no_candidate_without_fake_correction():
    net=model();net.eval();d=data();d['candidate_valid'].fill_(False)
    with torch.no_grad():
        results=net.forward_modes(**d,modes=tuple(THREE_SWITCH_MODES))
    for output in results.values():
        assert torch.allclose(output['prediction'],output['base_prediction'],atol=1e-6)


def test_replay_cache_isolation_by_severity_and_manifest():
    source=Path(__file__).resolve().parents[1]
    code=(source/'tcrr_stla/replay_cache.py').read_text()
    evaluation=(source/'scripts/evaluate_v5.py').read_text()
    assert 'cache_tag=None' in code
    assert "'cache_tag':cache_tag" in code
    assert "f'test_s{severity}'" in evaluation
    assert 'bank=replay' in evaluation
    assert 'TCRR_EVAL_WORKERS' in evaluation


def test_resource_logging_is_actual_epoch_output_not_placeholder():
    code=(Path(__file__).resolve().parents[1]/'tcrr_stla/v5/trainer.py').read_text()
    for token in ('torch.cuda.reset_peak_memory_stats', 'torch.cuda.max_memory_allocated',
                  'torch.cuda.max_memory_reserved','resource.getrusage',
                  'queries_per_second','peak_cpu_rss_gib'):
        assert token in code
    assert 'teacher=model.stla.eval()' in code
