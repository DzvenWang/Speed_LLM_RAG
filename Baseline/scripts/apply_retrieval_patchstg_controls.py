#!/usr/bin/env python3
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

import yaml

GADI_ROOT = next((p for p in Path(__file__).resolve().parents if p.name == 'Baseline'), Path.cwd())
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
CFG = ROOT / 'configs' / 'baseline.yaml'


def add_unique(seq, value):
    if value not in seq:
        seq.append(value)


def main():
    if not CFG.exists():
        raise SystemExit(f'Missing config: {CFG}')
    required = [
        ROOT / 'baseline/runners/raft_runner.py',
        ROOT / 'baseline/runners/saraf_runner.py',
        ROOT / 'baseline/runners/rast_runner.py',
        ROOT / 'baseline/backbones/patchstg_bridge.py',
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise SystemExit('Matched-control patch requires current Baseline project; missing: ' + ', '.join(missing))
    rast_text = (ROOT / 'baseline/runners/rast_runner.py').read_text()
    if 'PatchSTGRASTBridge' not in rast_text or "choices=['native', 'patchstg']" not in rast_text:
        raise SystemExit('Current rast_runner.py does not contain the existing RAST-PatchSTG controlled adaptation.')
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = CFG.with_name(f'{CFG.name}.pre_retrieval_patchstg_controls_{stamp}.bak')
    shutil.copy2(CFG, backup)

    cfg = yaml.safe_load(CFG.read_text())
    controls = {
        'RAFT-PatchSTG': {
            'enabled': True,
            'table_group': 'secondary',
            'priority': 12,
            'formal_track': 'raft_retrieval_matched_frozen_patchstg_control',
            'stage_now': 'formal_seed10_ready',
            'requires_frozen_backbone_for_final_forecast': True,
            'primary_experiment': False,
            'formal_seeds': [10],
        },
        'SARAF-PatchSTG': {
            'enabled': True,
            'table_group': 'secondary',
            'priority': 13,
            'formal_track': 'saraf_retrieval_matched_frozen_patchstg_control',
            'stage_now': 'formal_seed10_ready',
            'requires_frozen_backbone_for_final_forecast': True,
            'primary_experiment': False,
            'formal_seeds': [10],
        },
        'RAST-PatchSTG': {
            'enabled': True,
            'table_group': 'secondary',
            'priority': 11,
            'formal_track': 'traffic_rag_matched_frozen_patchstg_controlled_adaptation',
            'stage_now': 'formal_seed10_ready',
            'requires_frozen_backbone_for_final_forecast': True,
            'primary_experiment': False,
            'formal_seeds': [10],
        },
    }
    methods = cfg.setdefault('methods', {})
    for name, spec in controls.items():
        current = methods.setdefault(name, {})
        current.update(spec)

    protocol = cfg.setdefault('protocol', {})
    frozen_for = protocol.setdefault('frozen_backbone_required_for', [])
    for name in controls:
        add_unique(frozen_for, name)

    story = cfg.setdefault('baseline_story', {})
    secondary_story = story.setdefault('secondary_experiments', [])
    for name in controls:
        add_unique(secondary_story, name)

    secondary = cfg.setdefault('secondary_methods', [])
    for name in controls:
        add_unique(secondary, name)

    revisions = cfg.setdefault('method_implementation_revisions', {})
    revisions['RAFT-PatchSTG'] = 'raft_native_retrieval_fusion_frozen_patchstg_directpath_r1'
    revisions['SARAF-PatchSTG'] = 'saraf_native_retrieval_fusion_frozen_patchstg_directpath_r1'
    revisions['RAST-PatchSTG'] = revisions.get(
        'RAST-PatchSTG', 'rast_native_retrieval_fusion_frozen_patchstg_backbone_r1'
    )

    CFG.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f'PATCHED {CFG}')
    print(f'BACKUP  {backup}')
    for name in controls:
        print(f'PASS method {name} formal_seeds={methods[name]["formal_seeds"]}')


if __name__ == '__main__':
    main()
