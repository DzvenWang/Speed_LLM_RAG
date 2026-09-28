#!/usr/bin/env python3
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from supportbench.config import load_config, discover_source_root
p=argparse.ArgumentParser(); p.add_argument('--dataset',default='TraffiDent_Net_06',choices=[f'TraffiDent_Net_{x}' for x in ['01','02','03','04','06']]); p.add_argument('--config',default=str(ROOT/'configs'/'base_supportbench.yaml')); p.add_argument('--target',choices=['speed','flow'],default='speed'); a=p.parse_args()
cfg=load_config(a.config); cfg['source']['primary_channel']=a.target; src=discover_source_root(cfg); d=src/a.dataset
req=['traffic_2024.npy','nodes_target.csv','adj_target.npy']
print('Base_SupportBench preflight'); print('TARGET',a.target); print('PROJECT_ROOT',cfg['project']['project_root']); print('SOURCE_ROOT',src); print('DATASET',a.dataset)
for f in req:
    q=d/f; print(('PASS' if q.exists() else 'FAIL'),q)
    if not q.exists(): raise SystemExit(2)
# Freeze the original SupportBench General-track sensing-generation contract.
s=cfg['sensing']
expected={
 'global_seed':20260905,'general_families':['entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage'],
 'severities':[0.20,0.40,0.60,0.90],'train_severities':[0.20,0.40,0.60],'main_severity':0.40,
 'train_views_per_window':4,'val_instances_per_family_severity':1,'test_instances_per_family_severity':1,'paired_main_instances':2,
 'temporal_node_fraction':0.75,'ood_severities':[0.40],'regional_seen_fraction':0.70,
 'regional_id_layouts_per_severity':32,'regional_ood_layouts_per_severity':16,'regional_min_id_layouts_per_severity':16,'regional_min_ood_layouts_per_severity':8,
 'directional_id_layouts_per_severity':32,'directional_ood_layouts_per_severity':16,'directional_min_id_layouts_per_severity':16,'directional_min_ood_layouts_per_severity':8,
 'effective_ratio_tolerance':0.04,'regional_min_largest_component_fraction':0.50}
for k,v in expected.items():
    if s.get(k)!=v:
        print('FAIL sensing contract',k,'actual=',s.get(k),'expected=',v); raise SystemExit(2)
if s.get('layout_min_overrides')!={'s90':{'regional_id':8,'directional_id':8}}:
    print('FAIL sensing contract layout_min_overrides',s.get('layout_min_overrides')); raise SystemExit(2)
print('PASS original SupportBench general sensing contract 20/40/60/90')
# Target channel must exist in the raw mapping/default contract.
from supportbench.utils import load_channel_names
import numpy as np
shape=np.load(d/cfg['source']['traffic_file'],mmap_mode='r').shape
channels=load_channel_names(d,int(shape[-1]),cfg['source'].get('channel_mapping_file','channel_mapping.json'))
if a.target not in channels:
    print('FAIL target channel',a.target,'channels=',channels); raise SystemExit(2)
print('PASS target channel',a.target,'index=',channels.index(a.target),'channels=',channels)

print('PREFLIGHT PASS')
