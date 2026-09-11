#!/usr/bin/env python3
"""Static/deterministic contract smoke for the PatchSTG recent adapter."""
from pathlib import Path
import sys, numpy as np, pandas as pd, torch
GADI_ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Backbone')
ROOT=GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backbone.adapters import _node_coordinate_array
from backbone.utils import load_yaml
cfg=load_yaml(ROOT/'configs/backbone_selection.yaml')
N=388
nodes=pd.DataFrame({'lat':np.linspace(-37.9,-37.0,N),'lng':np.linspace(144.5,145.5,N),'postmile':np.arange(N)})
loc=_node_coordinate_array(nodes,N)
assert loc.shape==(2,N) and np.isfinite(loc).all()
p=cfg['models']['PatchSTG']; assert int(p['tem_patchsize'])*int(p['tem_patchnum'])==12
assert 2**int(p['spatial_recur'])==int(p['spa_patchnum']) and int(p['spa_patchnum'])%int(p['factors'])==0
print('RECENT MODEL CONTRACT SMOKE PASS')
