#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from supportbench.config import load_config
from supportbench.dataset import SupportBench

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',default='TraffiDent_Net_06'); args=ap.parse_args(); cfg=load_config(args.config)
    b=SupportBench(args.subgraph,config=args.config)
    df=b.views_for('test',family='regional_outage',severity=float(cfg['sensing']['main_severity']),distribution='id')
    if df.empty: raise RuntimeError('No smoke-test view')
    s=b.sample(df.iloc[0].to_dict()); v=b.model_view(s); ep=b.memory_episode(0); patch=b.local_patch_nodes(0)
    print('PASS API')
    print('history',v['history'].shape,'mask',v['observation_mask'].shape,'future',v['future_speed'].shape)
    print('view',s.view_id,'effective_missing',float(s.artificial_mask.mean()))
    print('memory history',ep['history_speed'].shape,'future',ep['future_speed'].shape,'patch0',patch.tolist())
if __name__=='__main__': main()
