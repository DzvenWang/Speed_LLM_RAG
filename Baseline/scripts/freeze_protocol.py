#!/usr/bin/env python3
from pathlib import Path
import argparse, json, sys, subprocess

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baseline.utils import load_yaml, write_json
from baseline.fingerprint import sha256_file, canonical_sha256
from baseline.registry import METHODS, ORDER
from baseline.run_identity import load_cache_identity
from baseline.shared_cache import cache_path
from baseline.backbones.frozen_registry import load_frozen_patchstg_registry

p = argparse.ArgumentParser()
p.add_argument('--note', default='Formal baseline protocol frozen after validation-only selection.')
a = p.parse_args()
cfg = load_yaml(ROOT / 'configs/baseline.yaml')
exp = Path(cfg['project']['exp_root'])
formal = cache_path(cfg, 'formal') / 'cache_manifest.json'
sel = exp / 'FORMAL_SELECTION.json'
subprocess.check_call([cfg['project']['python'], str(ROOT / 'scripts/validate_cache.py'), '--stage', 'formal'])
if not sel.exists():
    raise SystemExit('Run freeze_formal_selection.py before protocol freeze')
selection = json.loads(sel.read_text())
identity = load_cache_identity(formal.parent, cfg['version'])
if selection.get('formal_run_identity') != identity:
    raise SystemExit('FORMAL_SELECTION.json does not belong to the current Baseline formal cache identity')

cm = json.loads(formal.read_text())
contract = cfg.get('frozen_backbone_contract', {})
fb = load_frozen_patchstg_registry(
    cfg['project']['frozen_backbone_json'],
    expected_protocol_hash=cm['protocol_stage_hash'],
    expected_time_feature_revision=cm['time_feature_revision'],
    expected_time_feature_fingerprint=cm['time_feature_fingerprint'],
    expected_time_feature_source=cm['time_feature_source'],
    expected_dataset=cfg['experiment']['subgraph'],
    expected_backbone_project_version=str(contract.get('required_project_version', '0.5.0')),
)
if selection.get('frozen_backbone', {}).get('reference_checkpoint_sha256') != fb['reference_checkpoint_sha256']:
    raise SystemExit('FORMAL_SELECTION.json is bound to a different frozen PatchSTG reference checkpoint')

obj = {
    'schema_version': 2,
    'baseline_version': cfg['version'],
    'formal_run_identity': identity,
    'config_sha256': sha256_file(ROOT / 'configs/baseline.yaml'),
    'formal_cache_manifest': cm,
    'official_sources': {
        m: {'commit': METHODS[m]['commit'], 'remote': METHODS[m]['repo']} for m in ORDER
    },
    'frozen_backbone_json': fb,
    'formal_selection': selection,
    'note': a.note,
}
obj['freeze_fingerprint'] = canonical_sha256(obj)
out = exp / 'FROZEN_BASELINE_PROTOCOL.json'
write_json(out, obj)
print('wrote', out)
print('FREEZE_FINGERPRINT=' + obj['freeze_fingerprint'])
