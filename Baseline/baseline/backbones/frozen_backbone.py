from __future__ import annotations
import json
from pathlib import Path
import torch
from .frozen_registry import load_frozen_backbone_registry


def inspect_frozen_backbone(cfg: dict, *, require_files: bool = True) -> dict:
    """Return the validated generic frozen Backbone manifest."""
    from baseline.shared_cache import cache_path
    formal = cache_path(cfg, 'formal') / 'cache_manifest.json'
    if not formal.exists():
        raise FileNotFoundError(f'Baseline formal cache required before loading frozen Backbone: {formal}')
    cm=json.loads(formal.read_text())
    contract=cfg.get('frozen_backbone_contract',{})
    return load_frozen_backbone_registry(
        cfg['project']['frozen_backbone_json'], require_files=require_files,
        expected_protocol_hash=cm.get('protocol_stage_hash'),
        expected_time_feature_revision=cm.get('time_feature_revision'),
        expected_time_feature_fingerprint=cm.get('time_feature_fingerprint'),
        expected_time_feature_source=cm.get('time_feature_source'),
        expected_dataset=cfg['experiment']['subgraph'], expected_target=cfg['experiment']['target'],
        expected_backbone_project_version=str(contract.get('required_project_version','1.1.7')),
        expected_backbone_data_contract_revision=contract.get('required_data_contract_revision','traffident_experiment_cache_v1'),
        expected_shared_contract_version=contract.get('required_shared_contract_version','1.2.1'),
        expected_cache_schema_version=int(contract.get('required_cache_schema_version',3)),
        expected_protocol_version=contract.get('required_protocol_version','2.1.0'),
        expected_paired_group_id=contract.get('required_paired_group_id'),
        expected_metrics_revision=cfg['shared_contract']['metrics_revision'],
        expected_prediction_contract_revision=cfg['shared_contract']['prediction_contract_revision'],
    )


def build_frozen_backbone(cfg: dict, device: torch.device):
    """Dispatch to a model-specific bridge while preserving validated implementations.

    PatchSTG deliberately uses the pre-v1.3.2 FrozenPatchSTG implementation unchanged.
    Other frozen Backbone models require an explicitly implemented/validated bridge rather
    than silently reusing PatchSTG assumptions.
    """
    obj=inspect_frozen_backbone(cfg)
    selected=str(obj['selected_model'])
    if selected == 'PatchSTG':
        from .patchstg_bridge import FrozenPatchSTG
        return FrozenPatchSTG(cfg, device)
    raise NotImplementedError(
        f'Frozen Backbone selected_model={selected!r} is valid at the registry level, but '
        'Baseline has no validated downstream consumer bridge for that model yet. '
        'Add a model-specific bridge before using it with ImputeFormer/LRTC-TMCP.'
    )
