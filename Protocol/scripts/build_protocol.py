#!/usr/bin/env python3
from pathlib import Path
import argparse, sys

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Protocol')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'TraffiDent_SupportBench'))
sys.dont_write_bytecode = True
from shared_protocol.builder import build_stage, validate_stage
from shared_protocol.utils import load_yaml
from supportbench.dataset import SupportBench

p = argparse.ArgumentParser()
p.add_argument('--stage', choices=['development', 'formal', 'evaluation'], required=True)
p.add_argument('--force', action='store_true')
a = p.parse_args()
cfg = load_yaml(ROOT / 'configs/protocol.yaml')
b = SupportBench(cfg['experiment']['dataset'], root=cfg['project']['benchmark_root'])
out = build_stage(b, cfg, a.stage, force=a.force)
try:
    manifest = validate_stage(cfg, a.stage)
except Exception:
    # Never leave a failed stage marked immutable/frozen.
    (out / 'FROZEN.json').unlink(missing_ok=True)
    raise
print('PROTOCOL_STAGE=' + a.stage)
print('PROTOCOL_STAGE_HASH=' + manifest['stage_hash'])
print('PROTOCOL_FROZEN=' + str(out / 'FROZEN.json'))
print('PROTOCOL BUILD PASS')
