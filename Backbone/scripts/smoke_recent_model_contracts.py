#!/usr/bin/env python3
"""Static/deterministic contract smoke for recent Backbone adapters."""
from pathlib import Path
import sys, numpy as np, pandas as pd, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backbone.adapters import _node_coordinate_array, stlaformer_features
from backbone.utils import load_yaml
cfg=load_yaml(ROOT/'configs/backbone_selection.yaml')
N=388
nodes=pd.DataFrame({'lat':np.linspace(-37.9,-37.0,N),'lng':np.linspace(144.5,145.5,N),'postmile':np.arange(N)})
loc=_node_coordinate_array(nodes,N)
assert loc.shape==(2,N) and np.isfinite(loc).all()
p=cfg['models']['PatchSTG']; assert int(p['tem_patchsize'])*int(p['tem_patchnum'])==12
assert 2**int(p['spatial_recur'])==int(p['spa_patchnum']) and int(p['spa_patchnum'])%int(p['factors'])==0
print('RECENT MODEL CONTRACT SMOKE PASS')


# STLAformer pinned official constructor defaults and canonical TOW bridge.
st=cfg['models']['STLAformer']
assert int(st['input_dim'])==2
assert int(st['steps_per_week'])==7*288
assert int(st['input_embedding_dim'])==32
assert int(st['tow_embedding_dim'])==32
assert int(st['batch_size'])==16 and int(st['max_epochs'])==200
assert str(st['scheduler']).lower()=='plateau' and int(st['scheduler_patience'])==20
assert int(st['num_heads'])==4 and int(st['num_layers'])==6
B,T,N2=2,12,5
batch={
    'x':torch.zeros(B,T,N2),
    'tod_hist':torch.arange(T).view(1,T).expand(B,T),
    'dow_hist':torch.tensor([[0]*T,[6]*T]),
}
f=stlaformer_features(batch,288,7,torch.device('cpu'))
assert tuple(f.shape)==(B,T,N2,2)
assert torch.allclose(f[0,:,0,1],torch.arange(T,dtype=torch.float32)/2016.0)
assert torch.all(f[1,:,0,1] >= (6*288)/2016.0)
# Exhaustive float32 round-trip: vendor uses (tow*2016).long().
slots=torch.arange(2016)
full={
    'x':torch.zeros(1,2016,1),
    'tod_hist':(slots%288).view(1,-1),
    'dow_hist':(slots//288).view(1,-1),
}
ff=stlaformer_features(full,288,7,torch.device('cpu'))
recovered=(ff[0,:,0,1]*2016).long()
assert torch.equal(recovered,slots), 'STLAformer fractional TOW must round-trip exactly to official embedding indices'
print('STLAFORMER CONTRACT SMOKE PASS')
