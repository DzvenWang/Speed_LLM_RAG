#!/usr/bin/env python3
"""CPU-only, no GPU queue: check available checkpoint and family masks."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from fare.config import cache_dir
from fare.protocol import SupportDataset
from fare.diagnostics_once.native_export import _family_column

def validate(checkpoint,config):
    cp=Path(checkpoint).resolve()
    if not cp.is_file():raise FileNotFoundError(cp)
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    if not all(k in ck for k in ('config','metadata','state_dict')):raise RuntimeError('Not FARE checkpoint')
    cfg=ck['config'];target=cfg['experiment']['target'];info={'target':target,'checkpoint':str(cp),'severities':[]}
    if cfg['experiment']['seed'] != config['seed']:raise ValueError('Seed mismatch')
    if cfg['experiment']['dataset']!=config['dataset']:raise ValueError('Dataset mismatch')
    for field in ('backbone_checkpoint','memory_path'):
        v=Path(ck['metadata'][field]);marker=v/'manifest.json' if field=='memory_path' else v
        if not marker.exists():raise FileNotFoundError(f'Checkpoint dependency {field} absent: {marker}')
    for severity in config['expected_severities']:
        directory=cache_dir(cfg,'evaluation',severity)
        ds=SupportDataset(directory,config['split'])
        col=_family_column(ds)
        counts=ds.meta.iloc[ds.ix][col].astype(str).value_counts().to_dict()
        missing=[f for f in config['expected_families'] if not counts.get(f)]
        if missing:raise ValueError(f'{target} s{severity} missing {missing}; available {counts}')
        info['severities'].append({'severity':severity,'rows':len(ds),'families':counts,'family_column':col})
    return info

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='FARE/configs/diagnose_once.json');p.add_argument('--speed-checkpoint',required=True);p.add_argument('--flow-checkpoint',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text())
    if cfg['split']=='test' and not cfg.get('allow_exploratory_test',False):raise ValueError('Explicit test opt-in required')
    report=[validate(a.speed_checkpoint,cfg),validate(a.flow_checkpoint,cfg)]
    print('FARE_DIAG_CPU_PREFLIGHT_PASS '+json.dumps(report),flush=True)
if __name__=='__main__':main()
