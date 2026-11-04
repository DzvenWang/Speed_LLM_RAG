"""Immutable audited PCFT v6 graph contract and isolated graph cache."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from .config import sha256,json_write
from .graph_build.physical import build_physical,EDGE_FIELDS
from .graph_build.functional import build_functional,FUNCTIONAL_FIELDS

SCHEMA='pcft_v6_audited_graph_v1'

@dataclass
class DirectedGraph:
    edge_index:torch.Tensor
    edge_attr:torch.Tensor
    num_nodes:int
    functional_index:torch.Tensor|None=None
    functional_attr:torch.Tensor|None=None
    identity:str='synthetic'
    def __post_init__(self):
        if self.functional_index is None:self.functional_index=torch.empty((2,0),dtype=torch.long)
        if self.functional_attr is None:self.functional_attr=torch.empty((0,len(FUNCTIONAL_FIELDS)),dtype=torch.float32)
        if self.edge_index.shape[0]!=2 or self.functional_index.shape[0]!=2:raise ValueError('Edge index must be [2,E]')
        if self.edge_attr.shape!=(self.edge_index.shape[1],len(EDGE_FIELDS)):
            raise ValueError('Physical edge feature schema mismatch')
        if self.functional_attr.shape!=(self.functional_index.shape[1],len(FUNCTIONAL_FIELDS)):
            raise ValueError('Functional edge feature schema mismatch')
        if not torch.isfinite(self.edge_attr).all() or not torch.isfinite(self.functional_attr).all():
            raise ValueError('Nonfinite graph features')
        for index in (self.edge_index,self.functional_index):
            if index.numel() and (int(index.min())<0 or int(index.max())>=self.num_nodes):
                raise ValueError('Out-of-range graph edge')
    def to(self,device):
        return DirectedGraph(self.edge_index.to(device),self.edge_attr.to(device),self.num_nodes,
                             self.functional_index.to(device),self.functional_attr.to(device),self.identity)


def _hash(*parts):
    h=hashlib.sha256()
    for x in parts:h.update(x if isinstance(x,bytes) else str(x).encode())
    return h.hexdigest()


def graph_root(repo,dataset,target):
    return Path(repo)/'PCFT-STLA_cache'/'v6'/'graphs'/dataset/target


def source_identity(bench,ds,config):
    root=Path(bench.sg_dir)
    physical_files=['core/nodes.csv','graph/adjacency.npy','graph/distance.npy']
    src={x:sha256(root/x) if (root/x).is_file() else 'missing' for x in physical_files}
    src['train_manifest_sha256']=sha256(ds.root/'train_manifest.csv')
    src['multi_input_sha256']=sha256(Path(config['multi_cache_root'])/'identity.json')
    src['target']=ds.target
    src['node_count']=int(bench.N)
    src['schema']=SCHEMA
    src['graph_config']=config['graph_config']
    return src


def build_or_load_graph(bench,ds,repo,cfg,multi_root,*,force=False):
    config=dict(graph_config=cfg['graph'],multi_cache_root=str(multi_root))
    sources=source_identity(bench,ds,config)
    key=_hash(json.dumps(sources,sort_keys=True))
    out=graph_root(repo,bench.sg_dir.name,ds.target)
    bundle=out/f'{key[:16]}.npz';meta=out/f'{key[:16]}.json'
    if bundle.is_file() and meta.is_file() and not force:
        j=json.loads(meta.read_text())
        if j.get('source_identity')!=sources or j.get('sha256')!=sha256(bundle):
            raise RuntimeError(f'Graph cache identity mismatch: {bundle}')
        with np.load(bundle) as z:
            pe=z['physical_index'];pa=z['physical_attr'];fi=z['functional_index'];fa=z['functional_attr']
    else:
        pe,pa,pm=build_physical(bench,max_geo_km=float(cfg['graph']['max_geo_km']),
                               max_pm_gap_km=float(cfg['graph']['max_pm_gap_km']))
        fi,fa,fm=build_functional(ds,pe,topk=int(cfg['graph']['functional_topk']),
                                 max_lag=int(cfg['model']['max_lag']),
                                 max_timestamps=int(cfg['graph']['max_train_timestamps']),
                                 shortlist=int(cfg['graph']['functional_shortlist']),
                                 min_pairs=int(cfg['graph']['min_corr_pairs']),
                                 min_abs_corr=float(cfg['graph']['min_abs_corr']))
        out.mkdir(parents=True,exist_ok=True)
        tmp=bundle.with_name(bundle.stem+'.partial.npz')
        np.savez_compressed(tmp,physical_index=pe,physical_attr=pa,
                            functional_index=fi,functional_attr=fa)
        os.replace(tmp,bundle)
        summary=dict(schema=SCHEMA,source_identity=sources,physical_audit=pm,functional_audit=fm,
                     physical_feature_schema=list(EDGE_FIELDS),functional_feature_schema=list(FUNCTIONAL_FIELDS),
                     sha256=sha256(bundle),physical_edges=int(pe.shape[1]),functional_edges=int(fi.shape[1]))
        # Node identity retained for human verification, but do not copy huge arrays into checkpoints.
        json_write(meta,summary)
    if fi.shape[1]==0:
        print('WARN_V6_FUNCTIONAL_GRAPH_EMPTY: check observed TRAIN visibility/correlation thresholds; no physical edge is invented',flush=True)
    else:
        print('GRAPH_V6_READY physical_edges=',pe.shape[1],'functional_edges=',fi.shape[1],'cache=',bundle,flush=True)
    return DirectedGraph(torch.from_numpy(pe.copy()).long(),torch.from_numpy(pa.copy()).float(),int(bench.N),
                         torch.from_numpy(fi.copy()).long(),torch.from_numpy(fa.copy()).float(),
                         _hash(key,sha256(bundle)))


def physical_only_graph(bench):
    pe,pa,_=build_physical(bench)
    return DirectedGraph(torch.from_numpy(pe),torch.from_numpy(pa),int(bench.N))


def graph_audit(bench,graph):
    n=int(bench.N)
    if graph.num_nodes!=n:raise RuntimeError('Graph node mismatch')
    pe=graph.edge_index.cpu().numpy();pa=graph.edge_attr.cpu().numpy()
    pairs={(int(a),int(b)) for a,b in pe.T}
    if len(pairs)!=len(pe.T):raise RuntimeError('Duplicate directed physical edges')
    if any((b,a) not in pairs for a,b in pairs):raise RuntimeError('Missing reverse physical candidate')
    if ((pa[:,6]+pa[:,7])!=1).any():raise RuntimeError('Ambiguous forward/reverse edge flags')
    if ((pa[:,0]<0)|(pa[:,0]>1)|(pa[:,3]<0)|(pa[:,3]>1)).any():raise RuntimeError('Invalid graph masks')
    return {'physical_edges':len(pairs),'functional_edges':int(graph.functional_index.shape[1]),
            'physical_direction_types':{'forward':int((pa[:,6]>0).sum()),'reverse':int((pa[:,7]>0).sum())},
            'invalid_raw_distances':int((pa[:,0]==0).sum()),'graph_sha256':graph.identity}
