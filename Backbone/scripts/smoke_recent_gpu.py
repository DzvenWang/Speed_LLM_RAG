#!/usr/bin/env python3
from pathlib import Path
import argparse, sys
import torch
from torch.utils.data import DataLoader
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT.parent/'TraffiDent_SupportBench'))
from backbone.adapters import build_adapter
from backbone.data import CachedSplit
from backbone.utils import count_parameters, load_yaml
from backbone.shared_cache import cache_path
from supportbench.dataset import SupportBench

ap=argparse.ArgumentParser(); ap.add_argument('--model',required=True,choices=['PatchSTG','STLAformer']); args=ap.parse_args()
if not torch.cuda.is_available(): raise SystemExit('GPU smoke requires cuda:0')
cfg=load_yaml(ROOT/'configs/backbone_selection.yaml'); sg=cfg['experiment']['subgraph']
b=SupportBench(sg,root=cfg['project']['benchmark_root']); st=b.stats['channels'][b.manifest['primary_channel']]
cache=cache_path(cfg,'screening'); val=CachedSplit(cache,'val')
mc=dict(cfg['models'][args.model]); bs=min(2,int(mc['batch_size']))
loader=DataLoader(val,batch_size=bs,shuffle=False,num_workers=0)
batch=next(iter(loader)); device=torch.device('cuda:0')
adapter=build_adapter(args.model,ROOT/'vendor',b.N,device,float(st['mean']),float(st['std']),mc,b.adj,b.nodes)
adapter.model.eval()
with torch.no_grad(): pred=adapter.eval_forward(batch)
expected=tuple(batch['y_raw'].shape)
if tuple(pred.shape)!=expected: raise SystemExit(f'FAIL {args.model} shape={tuple(pred.shape)} expected={expected}')
if not torch.isfinite(pred).all(): raise SystemExit(f'FAIL {args.model} non-finite prediction')
print(f'RECENT GPU SMOKE PASS model={args.model} shape={tuple(pred.shape)} params={count_parameters(adapter.model):,} dtype={pred.dtype}')
print('adapter_revision='+str(getattr(adapter,'adapter_revision','base')))
print('adapter_metadata='+str(getattr(adapter,'adapter_metadata',{})))
