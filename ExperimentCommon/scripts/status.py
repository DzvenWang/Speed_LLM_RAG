#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
for p in [ROOT, REPO_ROOT / 'Protocol', REPO_ROOT / 'TraffiDent_SupportBench']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiment_common.cache import cache_dir, evaluation_access_granted, load_protocol_stage_metadata
from experiment_common.utils import load_yaml

p = argparse.ArgumentParser()
p.add_argument('--config', default=str(ROOT / 'configs' / 'experiment_contract.yaml'))
a = p.parse_args()
cfg = load_yaml(a.config)
print('Protocol:', cfg['protocol']['name'])
print('Severity:', float(cfg['experiment']['severity']))
for stage in cfg['protocol'].get('allowed_stages', []):
    try:
        pm = load_protocol_stage_metadata(cfg, stage)
        out = cache_dir(cfg, stage, protocol_stage_hash=str(pm['stage_hash']))
        ready = out / 'READY.json'
        if ready.exists():
            r = json.loads(ready.read_text(encoding='utf-8'))
            extra = ''
            if stage == 'evaluation':
                extra = f" access_state={r.get('evaluation_access_state')} materialized_while_locked={r.get('materialized_while_locked')}"
            print(stage, 'READY', r.get('cache_fingerprint'), out, extra)
        else:
            print(stage, 'NOT_BUILT', out)
    except Exception as e:
        print(stage, 'UNAVAILABLE', type(e).__name__, str(e))
if 'evaluation' in cfg['protocol'].get('allowed_stages', []):
    print('evaluation_access_granted=', evaluation_access_granted(cfg))
