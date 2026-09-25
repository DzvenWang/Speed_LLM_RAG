#!/usr/bin/env python3
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared_protocol.builder import validate_paired_evaluations
from shared_protocol.utils import load_yaml

configs = []
for s in ('20', '40', '60', '90'):
    p = ROOT / 'configs' / f'protocol_s{s}.yaml'
    cfg = load_yaml(p)
    cfg['_config_path'] = str(p.resolve())
    configs.append(cfg)
report = validate_paired_evaluations(configs)
print(json.dumps(report, indent=2))
print('PAIRED EVALUATION VALIDATION PASS')
