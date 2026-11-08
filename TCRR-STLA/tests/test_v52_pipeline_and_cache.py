"""Release-blocking tests for staged PBS and mask/caching dependencies."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]


def test_two_stage_pbs_dependencies_are_real_and_automatic():
    script=(ROOT/'pbs/submit_pipeline_v5.sh').read_text()
    assert 'TCRR-STLA/pbs/formal_v5.pbs' in script
    assert 'TCRR-STLA/pbs/postprocess_v5.pbs' in script
    assert 'depend=afterok:$TRAIN_ID' in script
    assert 'depend=afterok:$DIAG_ID' not in script
    assert 'SKIP_DIAGNOSTICS=1,SKIP_EVALUATE=1' in script
    assert 'RUN_STAMP=$RUN_STAMP' in script
    for filename in ('formal_v5.pbs','postprocess_v5.pbs','submit_pipeline_v5.sh'):
        subprocess.run(['bash','-n',str(ROOT/'pbs'/filename)],check=True)


def test_auto_diagnosis_uses_chronological_check_not_calibration():
    source=(ROOT/'scripts/auto_diagnose_v5.py').read_text()
    assert "'--validation-section','check'" in source
    assert 'wrong_mask_effective_rate' in source
    assert 'Wrong-mask comparison has insufficient distinct-mask coverage' in source


def test_train_selection_calibration_check_are_separate():
    source=(ROOT/'tcrr_stla/v5/trainer.py').read_text()
    assert "Subset(val_ds,partitions['selection'])" in source
    assert "Subset(val_ds,partitions['calibration'])" in source
    assert "Subset(val_ds,partitions['check'])" in source
    assert 'calibrate_controller(model,vl_cal,' in source
    assert 'check_predictions=predict(model,vl_check,' in source
    assert "best_mae,best_phase,best_path,cal=min(stage_candidates" in source


def test_new_replay_requires_original_source_masks_online_and_offline():
    cache=(ROOT/'tcrr_stla/replay_cache.py').read_text()
    direct=(ROOT/'tcrr_stla/trainer.py').read_text()
    for text in (cache,direct):
        assert "strict_source_mask=str(cfg.get('version','')).startswith(('4.','5.'))" in text
    assert "'replay_protocol':'source_mask_intersection_v2'" in cache
    assert "'schema':27" in cache


def test_joint_harm_branch_and_local_regularizers_have_real_objectives():
    loss=(ROOT/'tcrr_stla/v5/losses.py').read_text()
    controller=(ROOT/'tcrr_stla/v5/controller.py').read_text()
    assert "o['joint_harm_logit']" in loss
    assert "o['pre_rejection_correction']" in loss
    assert "gate_spatial_smooth_weight" in loss
    assert "c3_local_offset_weight" in loss
    assert 'region_mean(self._temporal_blocks' in controller
    assert 'self.joint_harm(joint_input)' in controller


def test_v52_uses_new_version_in_every_active_entrypoint():
    # Root VERSION follows the latest release while the v5 entrypoints remain pinned.
    assert (ROOT/'VERSION').read_text().strip() in ('5.3.0', '9.3.0')
    cfg=(ROOT/'configs/tcrr_stla_v5.yaml').read_text()
    assert 'version: "5.3.0"' in cfg
    for path in ('scripts/train_v5.py','scripts/evaluate_v5.py'):
        assert '5.3.0' in (ROOT/path).read_text()
