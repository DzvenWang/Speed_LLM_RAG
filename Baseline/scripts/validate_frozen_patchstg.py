#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

GADI_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.utils import load_yaml
from baseline.shared_cache import cache_path
from baseline.backbones.frozen_registry import load_frozen_patchstg_registry


def main():
    cfg = load_yaml(ROOT / 'configs/baseline.yaml')
    exp = Path(cfg['project']['exp_root'])
    formal = cache_path(cfg, 'formal') / 'cache_manifest.json'
    if not formal.exists():
        raise SystemExit(f'Missing Baseline formal cache: {formal}. Build and validate it first.')
    cm = json.loads(formal.read_text())
    contract = cfg.get('frozen_backbone_contract', {})
    obj = load_frozen_patchstg_registry(
        cfg['project']['frozen_backbone_json'],
        expected_protocol_hash=cm['protocol_stage_hash'],
        expected_time_feature_revision=cm['time_feature_revision'],
        expected_time_feature_fingerprint=cm['time_feature_fingerprint'],
        expected_time_feature_source=cm['time_feature_source'],
        expected_dataset=cfg['experiment']['subgraph'],
        expected_backbone_project_version=str(contract.get('required_project_version', '1.0.10')),
    )
    print('PASS Backbone-owned FROZEN_BACKBONE.json schema', obj['schema_version'])
    print('PASS selected_model', obj['selected_model'])
    print('PASS Backbone project_version', obj['project_version'])
    print('PASS reference_seed', obj['reference_seed'])
    print('PASS reference_checkpoint', obj['reference_checkpoint'])
    print('PASS reference_checkpoint_sha256', obj['reference_checkpoint_sha256'])
    print('PASS protocol_stage_hash', obj['run_identity']['protocol_stage_hash'])
    print('PASS time_feature_revision', obj['run_identity']['time_feature_revision'])
    print('PASS time_feature_fingerprint', obj['run_identity']['time_feature_fingerprint'])
    print('FROZEN PATCHSTG VALIDATION PASS')


if __name__ == '__main__':
    main()
