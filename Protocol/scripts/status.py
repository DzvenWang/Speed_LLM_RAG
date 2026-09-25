#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from shared_protocol.utils import load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT / 'configs' / 'protocol_s40.yaml'))
    a = ap.parse_args()
    cfg = load_yaml(a.config)
    base = Path(cfg['project']['artifact_root']) / cfg['experiment']['dataset'] / cfg['protocol']['name']
    print('Protocol:', base)
    print('severity=', cfg['experiment']['severity'])
    print('allowed_stages=', cfg['protocol']['allowed_stages'])
    for stage in ['development', 'formal', 'evaluation']:
        sd = base / stage
        frozen = sd / 'FROZEN.json'
        mf = sd / 'stage_manifest.json'
        if frozen.exists() and mf.exists():
            j = json.loads(mf.read_text(encoding='utf-8'))
            print(stage, 'FROZEN', j['stage_hash'], j['splits'])
        else:
            print(stage, 'NOT_BUILT')
    pair_root = Path(cfg['project']['artifact_root']) / cfg['experiment']['dataset'] / '_paired' / cfg['protocol']['paired_group_id']
    print('pairing_frozen=', (pair_root / 'PAIRING_FROZEN.json').exists())
    print('evaluation_access_policy=direct_unlocked')
    print('legacy_evaluation_gate_present=', (pair_root / 'EVALUATION_ACCESS_GRANTED.json').exists())


if __name__ == '__main__':
    main()
