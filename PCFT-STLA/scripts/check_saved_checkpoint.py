#!/usr/bin/env python3
"""Validate existing PCFT v1.0 best.pt and its Train-only memory index.

Checks metadata without allocating GPU or launching model training.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pcft_stla.config import file_sha256
from pcft_stla.memory import LatentMemoryIndex


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--backbone-checkpoint')
    p.add_argument('--target',choices=('speed','flow'))
    a=p.parse_args()
    cp=Path(a.checkpoint).expanduser().resolve()
    if not cp.is_file():raise FileNotFoundError(cp)
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    if str(ck.get('version')) not in ('2.0.0','3.0.0'):raise RuntimeError('Expected compatible v2.0/v3.0 checkpoint')
    if ck.get('model')!='PCFT-STLA':raise RuntimeError(f'Wrong model {ck.get("model")}')
    target=ck['config']['experiment']['target']
    if a.target and a.target!=target:raise RuntimeError(f'Wrong target {target}')
    bp=Path(a.backbone_checkpoint or ck['backbone_checkpoint']).expanduser()
    if not bp.is_absolute():bp=Path(ck['config']['project']['root'])/bp
    if not bp.is_file():raise FileNotFoundError(f'Original STLA backbone not found: {bp}')
    actual=file_sha256(bp)
    if ck.get('backbone_sha256') and ck['backbone_sha256']!=actual:
        raise RuntimeError('Backbone SHA256 changed; wrong pretrained STLA weights')
    index_path=cp.parent/'best_memory_index.npz'
    if ck['config']['training'].get('ablation','full')!='no_retrieval':
        if not index_path.is_file():raise FileNotFoundError(index_path)
        index=LatentMemoryIndex.load(index_path)
        if len(index.starts_np)<1:raise RuntimeError('Empty Train-only memory index')
        if not np.isfinite(index.temporal_np).all():raise RuntimeError('Non-finite memory temporal keys')
        rows=len(index.starts_np)
    else:rows=0
    print(f'CHECKPOINT_RECOVERY_READY target={target} best_epoch={ck.get("best_epoch")} '
          f'val_mae={ck.get("best_val_mae")} memory_rows={rows} '
          f'checkpoint={cp} index={index_path} backbone={bp}',flush=True)

if __name__=='__main__':main()
