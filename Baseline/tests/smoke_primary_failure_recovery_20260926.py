#!/usr/bin/env python3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ginar = (ROOT/'baseline/runners/ginar_runner.py').read_text()
saraf = (ROOT/'baseline/runners/saraf_runner.py').read_text()
common = (ROOT/'baseline/runners/common.py').read_text()
ev = (ROOT/'scripts/run_evaluation.py').read_text()

assert 'GINAR DIVERGENCE GUARD EARLY-STOP' in ginar
assert 'hard_reject_on_divergence' not in ginar
assert "'divergence_guard_triggered': termination_reason == 'validation_divergence'" in ginar
assert "'selection_valid': True" in ginar

assert "'saraf_time_mark_source':tsrc" in saraf
assert "'time_feature_source':tsrc" not in saraf
assert 'extra_summary must not overwrite canonical run-manifest fields' in common

assert "mp=d/'summary.json',d/'best.pt',d/'RUN_MANIFEST.json'" in ev or "sp,cp,mp=d/'summary.json',d/'best.pt',d/'RUN_MANIFEST.json'" in ev
assert 'require_same_identity(manifest,expected' in ev
assert 'raw, impact = rast_predict' in ev
assert 'return raw' in ev
assert 'remft_predict(m,dl,device,mean,std,use_amp=use_amp)' in ev
assert "formal_eval_batch_size" in ev

print('SMOKE primary failure recovery hotfix PASS')
