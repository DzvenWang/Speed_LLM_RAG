#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from shared_protocol.builder import validate_stage
from shared_protocol.utils import load_yaml

p = argparse.ArgumentParser()
p.add_argument('--stage', choices=['development', 'formal', 'evaluation'], required=True)
p.add_argument('--config', default=str(ROOT / 'configs' / 'protocol_s40.yaml'))
a = p.parse_args()
cfg = load_yaml(a.config)
cfg['_config_path'] = str(Path(a.config).resolve())
m = validate_stage(cfg, a.stage)
print('PASS', cfg['protocol']['name'], a.stage, m['stage_hash'])
print('PROTOCOL VALIDATION PASS')
