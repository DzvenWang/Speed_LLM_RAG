import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from pcft_stla.v6.graph import build_or_load_graph,graph_audit,physical_only_graph
from pcft_stla.v6.graph_build.physical import build_physical,EDGE_FIELDS
from pcft_stla.v6.graph_build.functional import build_functional,observed_train_series

class Bench:
    def __init__(self,sg):
        self.sg_dir=Path(sg);self.N=5
        self.nodes=pd.DataFrame(dict(ID=[100,101,102,103,104],Fwy=[1,1,1,1,2],
            Direction=['N']*5,Lat=[-37.0,-36.999,-36.998,-36.997,-36.996],
            Lng=[145.0]*5,**{'Abs PM':[10.,10.1,10.2,10.3,10.4]}))
        self.adj=np.zeros((5,5),dtype=np.float32)
        self.adj[0,1]=1;self.adj[1,2]=1;self.adj[2,3]=1;self.adj[3,4]=1
        self.sg_dir.mkdir(parents=True,exist_ok=True)
        (self.sg_dir/'graph').mkdir(exist_ok=True)
        raw=np.full((5,5),-1,dtype=np.float32)
        raw[0,1]=-1.;raw[1,0]=.1;raw[1,2]=.1;raw[2,1]=.1
        np.save(self.sg_dir/'graph/distance.npy',raw)

class Tiny:
    def __init__(self,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.target='speed';self.target_idx=0
        self.channels=['speed','flow','occupancy']
        self.meta=pd.DataFrame({'window_start':np.arange(150)})
        self.x=np.zeros((150,12,5,3),np.float32)
        self.obs=np.zeros_like(self.x,dtype=np.uint8)
        for w in range(150):
            t=np.arange(w,w+12)
            for node in range(5):
                self.x[w,:,node,0]=np.sin((t-node*2)/7.)
                self.obs[w,:,node,0]=1
        # Old views overlap heavily. No raw complete val/test data exists in this fixture.
        (self.root/'train_manifest.csv').write_text('view_id,window_start\n'+''.join(f'{i},{i}\n' for i in range(150)))
    def __len__(self):return len(self.meta)

def test_physical_builder_rejects_cross_road_and_no_invalid_distance_boost(tmp_path):
    bench=Bench(tmp_path)
    ei,ea,audit=build_physical(bench)
    assert ei.shape[0]==2 and ea.shape[1]==len(EDGE_FIELDS)
    assert ei.shape[1]==6 # pairs 0-1, 1-2, 2-3; cross-freeway 3-4 rejected
    assert audit['raw_distance_invalid']>0
    assert not np.isinf(ea).any() and np.isfinite(ea).all()
    ix=next(k for k in range(ei.shape[1]) if tuple(ei[:,k])==(0,1))
    assert ea[ix,0]==0 # raw distance marked unknown
    assert ea[ix,2]>0 and ea[ix,3]==1 # independent geo distance retained
    assert ea[ix,6]==1 and ea[ix,7]==0
    assert (ei[0]!=ei[1]).all()

def test_functional_train_observed_and_overlap_dedup(tmp_path):
    ds=Tiny(tmp_path/'train')
    x,m,st=observed_train_series(ds,max_timestamps=100)
    assert st['train_timestamps']<=100
    assert st['retained_observations']<=100*5
    pe=np.array([[0,1,2],[1,2,3]],dtype=np.int64)
    ei,ea,audit=build_functional(ds,pe,topk=2,max_lag=3,max_timestamps=100,
                                 min_pairs=6,min_abs_corr=0.01,shortlist=3)
    assert ei.shape[0]==2 and ea.shape[1]==6
    assert audit['source']=='train_s40_observed_only'
    assert audit['unique_train_windows']<=101
    assert np.isfinite(ea).all()

def test_functional_removed_mask_cannot_invent_corr(tmp_path):
    ds=Tiny(tmp_path/'train')
    ds.obs[:]=0
    ei,ea,_=build_functional(ds,np.empty((2,0),np.int64),topk=2,max_lag=3,
                             max_timestamps=100,min_pairs=3,min_abs_corr=0.001,shortlist=3)
    assert ei.shape==(2,0) and ea.shape==(0,6)

def test_v6_graph_cache_immutable_identity(tmp_path,monkeypatch):
    bench=Bench(tmp_path/'repo'/'base'/'TraffiDent_Net_06')
    ds=Tiny(tmp_path/'target')
    (tmp_path/'multicache').mkdir()
    (tmp_path/'multicache'/'identity.json').write_text('{"x":"train"}')
    (bench.sg_dir/'core').mkdir()
    (bench.sg_dir/'core/nodes.csv').write_text('id\n1\n')
    np.save(bench.sg_dir/'graph/adjacency.npy',bench.adj)
    cfg={'graph':{'max_geo_km':15.,'max_pm_gap_km':15.,'functional_topk':2,
                  'max_train_timestamps':100,'functional_shortlist':3,
                  'min_corr_pairs':6,'min_abs_corr':0.01},'model':{'max_lag':3}}
    g=build_or_load_graph(bench,ds,tmp_path/'repo',cfg,tmp_path/'multicache')
    graph_audit(bench,g)
    g2=build_or_load_graph(bench,ds,tmp_path/'repo',cfg,tmp_path/'multicache')
    assert g.identity==g2.identity
    assert g.edge_index.equal(g2.edge_index)
    # Input graph changed => identity and graph cache invalidated.
    bench.adj[0,2]=1
    np.save(bench.sg_dir/'graph/adjacency.npy',bench.adj)
    g3=build_or_load_graph(bench,ds,tmp_path/'repo',cfg,tmp_path/'multicache')
    assert g3.identity!=g.identity

def test_no_unknown_distance_as_direct_weight(tmp_path):
    b=Bench(tmp_path);ei,ea,_=build_physical(b)
    # There must be no inverse-distance feature with 0.99 normalized fallback weights.
    assert np.all(ea[:,8]==1.)
    assert np.all(ea[:,2]<1.)

def test_graph_source_integrity_disallows_duplicate_feature_semantics(tmp_path):
    b=Bench(tmp_path)
    _,a,_=build_physical(b)
    assert (a[:,6]+a[:,7]==1.).all()
    assert np.all(a[:,0]>=0.)

def test_logging_timestamps_every_line(capsys):
    import sys,re
    from pcft_stla.v6.timestamp import TimestampWriter
    class Target:
        def __init__(self):self.buffer=''
        def write(self,s):self.buffer+=s
        def flush(self):pass
    target=Target();w=TimestampWriter(target)
    w.write('first\nsecond');w.write(' part\nthird\n')
    lines=target.buffer.splitlines()
    assert len(lines)==3
    assert all(re.match(r'^\[\d{4}-\d{2}-\d{2}T',line) for line in lines)
    assert 'second part' in lines[1]
