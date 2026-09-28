#!/usr/bin/env python3
from __future__ import annotations
import json, sys
from pathlib import Path

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.utils import load_yaml, write_json
from baseline.backbones.frozen_registry import sha256_file, load_frozen_patchstg_registry
from baseline.fingerprint import canonical_sha256
from baseline.run_identity import load_cache_identity, require_same_identity
from baseline.shared_cache import cache_path

cfg = load_yaml(ROOT / 'configs/baseline.yaml')
exp = Path(cfg['project']['exp_root'])
sg = cfg['experiment']['subgraph']
methods = list(cfg['formal_selection_methods'])
formal_cache = cache_path(cfg, 'formal')
expected = load_cache_identity(formal_cache, cfg['version'])

cm = json.loads((formal_cache / 'cache_manifest.json').read_text())
contract = cfg.get('frozen_backbone_contract', {})
fb = load_frozen_patchstg_registry(
    cfg['project']['frozen_backbone_json'],
    expected_protocol_hash=cm['protocol_stage_hash'],
    expected_time_feature_revision=cm['time_feature_revision'],
    expected_time_feature_fingerprint=cm['time_feature_fingerprint'],
    expected_time_feature_source=cm['time_feature_source'],
    expected_dataset=sg,
    expected_backbone_project_version=str(contract.get('required_project_version', '1.0.10')),
)

obj = {
    'schema_version': 3,
    'baseline_version': cfg['version'],
    'dataset': sg,
    'formal_run_identity': expected,
    'frozen_backbone': {
        'schema_version': fb['schema_version'],
        'project_version': fb['project_version'],
        'selected_model': fb['selected_model'],
        'reference_seed': fb['reference_seed'],
        'reference_checkpoint': fb['reference_checkpoint'],
        'reference_checkpoint_sha256': fb['reference_checkpoint_sha256'],
        'run_identity': fb['run_identity'],
    },
    'methods': {},
    'method_reporting_tracks': {
        m: cfg['methods'][m].get('table_group', 'unknown') for m in methods
    },
}

for m in methods:
    obj['methods'][m] = {}
    seeds = [int(x) for x in cfg['methods'][m].get('formal_seeds', cfg['experiment']['formal_seeds'])]
    for seed in seeds:
        rr = exp / m / sg / 'formal'
        candidates = []
        if rr.exists():
            for d in rr.glob(f'*_seed{seed}'):
                sp, cp = d / 'summary.json', d / 'best.pt'
                if not (sp.exists() and cp.exists()):
                    continue
                try:
                    sm = json.loads(sp.read_text())
                    require_same_identity(sm, expected, label=str(sp))
                    if str(sm.get('code_project_version', sm.get('project_version'))) != str(cfg['version']):
                        raise ValueError('result was not produced by the current clean Baseline version')
                    if str(sm.get('status', '')) != 'COMPLETE':
                        raise ValueError('run status is not COMPLETE')
                    if not bool(sm.get('selection_valid', False)):
                        raise ValueError('run is marked invalid for Formal selection')
                    score = float(sm.get('val', {}).get('overall_MAE', float('inf')))
                    if not score < float('inf'):
                        raise ValueError('non-finite validation score')
                    candidates.append((score, d, cp, sm))
                except Exception:
                    continue
        if not candidates:
            raise SystemExit(f'Missing valid current-version Formal run: {m} seed={seed}')
        score, d, cp, sm = sorted(candidates, key=lambda x: (x[0], -x[1].stat().st_mtime))[0]

        if cfg['methods'][m].get('requires_frozen_backbone_for_final_forecast', False):
            if str(sm.get('patchstg_checkpoint')) != str(fb['reference_checkpoint']):
                raise SystemExit(f'{m} seed={seed}: frozen PatchSTG checkpoint mismatch')
            if int(sm.get('patchstg_reference_seed', -1)) != int(fb['reference_seed']):
                raise SystemExit(f'{m} seed={seed}: frozen PatchSTG reference seed mismatch')

        obj['methods'][m][str(seed)] = {
            'run_dir': str(d),
            'checkpoint': str(cp),
            'checkpoint_sha256': sha256_file(cp),
            'val_overall_MAE': score,
            'run_identity_hash': sm.get('run_identity_hash'),
        }

obj['selection_fingerprint'] = canonical_sha256(obj)
out = exp / 'FORMAL_SELECTION.json'
write_json(out, obj)
print('wrote', out)
print('SELECTION_FINGERPRINT=' + obj['selection_fingerprint'])
