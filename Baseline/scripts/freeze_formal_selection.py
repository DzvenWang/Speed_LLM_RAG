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
methods = ['ImputeFormer', 'GinAR', 'RAFT', 'RAST', 'SARAF']
seeds = [int(x) for x in cfg['experiment']['formal_seeds']]
formal_cache = cache_path(cfg, 'formal')
expected = load_cache_identity(formal_cache, cfg['version'])

# Validate Backbone reference once and bind the selection artifact to it.
cm = json.loads((formal_cache / 'cache_manifest.json').read_text())
contract = cfg.get('frozen_backbone_contract', {})
fb = load_frozen_patchstg_registry(
    cfg['project']['frozen_backbone_json'],
    expected_protocol_hash=cm['protocol_stage_hash'],
    expected_time_feature_revision=cm['time_feature_revision'],
    expected_time_feature_fingerprint=cm['time_feature_fingerprint'],
    expected_time_feature_source=cm['time_feature_source'],
    expected_dataset=sg,
    expected_backbone_project_version=str(contract.get('required_project_version', '0.5.0')),
)

obj = {
    'schema_version': 2,
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
}

for m in methods:
    obj['methods'][m] = {}
    for seed in seeds:
        rr = exp / m / sg / 'formal'
        cand = []
        for d in rr.glob(f'*_seed{seed}') if rr.exists() else []:
            sp = d / 'summary.json'; cp = d / 'best.pt'
            if not (sp.exists() and cp.exists()):
                continue
            try:
                sm = json.loads(sp.read_text())
                require_same_identity(sm, expected, label=str(sp))
                if str(sm.get('code_project_version', sm.get('project_version'))) != str(cfg['version']):
                    raise ValueError(
                        f'code project version {sm.get("code_project_version", sm.get("project_version"))} != {cfg["version"]}'
                    )
            except Exception:
                continue
            val = sm.get('val', {})
            score = float(val.get('overall_MAE', val.get('MAE', float('inf'))))
            cand.append((score, d, cp, sm))
        if not cand:
            raise SystemExit(f'Missing current-identity formal COMPLETE run {m} seed={seed}')
        score, d, cp, sm = sorted(cand, key=lambda x: (x[0], -x[1].stat().st_mtime))[0]

        # Methods that depend on PatchSTG must prove they used the same reference checkpoint.
        if cfg['methods'][m].get('requires_frozen_backbone_for_final_forecast', False):
            if str(sm.get('patchstg_checkpoint')) != str(fb['reference_checkpoint']):
                raise SystemExit(
                    f'{m} seed={seed}: PatchSTG checkpoint {sm.get("patchstg_checkpoint")} '
                    f'!= frozen reference {fb["reference_checkpoint"]}'
                )
            if int(sm.get('patchstg_reference_seed', -1)) != int(fb['reference_seed']):
                raise SystemExit(f'{m} seed={seed}: PatchSTG reference seed mismatch')

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
