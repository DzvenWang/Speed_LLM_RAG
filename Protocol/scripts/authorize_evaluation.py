#!/usr/bin/env python3
from pathlib import Path
import argparse, datetime, sys
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Protocol'); sys.path.insert(0,str(ROOT))
from shared_protocol.utils import load_yaml, write_json
p=argparse.ArgumentParser()
p.add_argument('--acknowledge-all-methods-frozen', action='store_true')
a=p.parse_args()
if not a.acknowledge_all_methods_frozen:
    raise SystemExit('REFUSE: pass --acknowledge-all-methods-frozen only after Backbone/Baseline/Ours configurations are frozen.')
cfg=load_yaml(ROOT/'configs/protocol.yaml')
base=Path(cfg['project']['artifact_root'])/cfg['experiment']['dataset']/cfg['protocol']['name']
if not (base/'formal'/'FROZEN.json').exists(): raise SystemExit('REFUSE: formal protocol is not frozen.')
write_json(base/'EVALUATION_AUTHORIZED.json',{
    'acknowledged':True,
    'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'statement':'All method selection/configuration is frozen; Test protocol may now be materialized.'
})
print('EVALUATION AUTHORIZED',base/'EVALUATION_AUTHORIZED.json')
