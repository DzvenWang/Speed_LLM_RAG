#!/usr/bin/env python3
from __future__ import annotations
import shutil, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
TMP=ROOT/'tests'/'_synthetic_runtime'
if TMP.exists(): shutil.rmtree(TMP)
src=TMP/'data'/'processed'/'traffident'/'subgraph_datasets_final'/'TraffiDent_Net_06'
src.mkdir(parents=True)
T,N,C=720,12,3
rng=np.random.default_rng(7)
t=np.arange(T,dtype=np.float32)
base=60+8*np.sin(2*np.pi*t/288)
traffic=np.empty((T,N,C),np.float32)
for n in range(N):
    speed=base + 2*np.sin(2*np.pi*(t-n*2)/96) - 0.3*n + rng.normal(0,0.5,T)
    traffic[:,n,2]=np.maximum(speed,5)
    traffic[:,n,0]=1500+100*np.sin(2*np.pi*t/288)+10*n+rng.normal(0,5,T)
    traffic[:,n,1]=0.15+0.02*np.sin(2*np.pi*t/96)+0.001*n
np.save(src/'traffic_2024.npy',traffic)
nodes=pd.DataFrame({'ID':np.arange(N),'Fwy':['I5']*6+['US101']*6,'Direction':['N']*6+['S']*6,'Lat':np.r_[np.linspace(34,34.5,6),np.linspace(35,34.5,6)],'Lng':[-118.0]*N,'Abs PM':np.r_[np.arange(6),np.arange(6)]})
nodes.to_csv(src/'nodes_target.csv',index=False)
adj=np.zeros((N,N),np.float32)
for a,b in list(zip(range(5),range(1,6)))+[(5,6)]+list(zip(range(6,11),range(7,12))): adj[a,b]=adj[b,a]=1
np.save(src/'adj_target.npy',adj)
dist=np.where(adj>0,1.0,0.0).astype(np.float32); np.save(src/'distance_target.npy',dist)
(src/'channel_mapping.json').write_text('{"output_channel_names":["flow","occupancy","speed"]}')

cfg=yaml.safe_load((ROOT/'configs'/'supportbench.yaml').read_text())
cfg['project']['project_root']=str(TMP)
cfg['project']['process_root']=str(ROOT)
cfg['project']['data_root']=str(TMP/'data')
cfg['project']['benchmark_root']=str(TMP/'data'/'processed'/'traffident'/'supportbench')
cfg['project']['diagnostics_root']=str(TMP/'data'/'diagnostics'/'traffident'/'supportbench')
cfg['project']['derived_root']=str(TMP/'artifacts'/'supportbench')
cfg['source']['source_root']=str(TMP/'data'/'processed'/'traffident'/'subgraph_datasets_final')
cfg['source']['expected_subgraphs']=['TraffiDent_Net_06']
cfg['source']['link_mode']='copy'

cfg['sensing']['regional_id_layouts_per_severity']=1
cfg['sensing']['regional_ood_layouts_per_severity']=1
cfg['sensing']['directional_id_layouts_per_severity']=2
cfg['sensing']['directional_ood_layouts_per_severity']=1

cfg['sensing']['predictive_regional_id_layouts_per_severity']=2
cfg['sensing']['predictive_regional_ood_layouts_per_severity']=1
cfg['sensing']['predictive_regional_min_id_layouts_per_severity']=1
cfg['sensing']['predictive_regional_min_ood_layouts_per_severity']=1
cfg['sensing']['predictive_directional_id_layouts_per_severity']=2
cfg['sensing']['predictive_directional_ood_layouts_per_severity']=1
cfg['sensing']['predictive_directional_min_id_layouts_per_severity']=1
cfg['sensing']['predictive_directional_min_ood_layouts_per_severity']=1
cfg['dependency']['min_pairs']=8
cfg['budgets']['development_train_windows']=100
cfg['budgets']['development_val_windows']=20
cfg['budgets']['development_test_windows']=20
cfg_path=TMP/'synthetic.yaml'; cfg_path.write_text(yaml.safe_dump(cfg,sort_keys=False))

env=dict(**__import__('os').environ); env['PYTHONPATH']=str(ROOT)+(':'+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
for script in ['01_build_core.py','02_build_windows.py','03_build_sensing_views.py','04_build_support_memory.py','05_build_protocols.py','06_audit.py','07_smoke_api.py']:
    cmd=[sys.executable,str(ROOT/'build'/script),'--config',str(cfg_path),'--subgraph','TraffiDent_Net_06']
    print('+',' '.join(cmd)); subprocess.check_call(cmd,env=env,cwd=ROOT)
print('SYNTHETIC BUILD PASS',TMP)
