from __future__ import annotations
import configparser, importlib.util, sys, types, warnings
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn


def load_module(name,path,prepend=None):
    if prepend is not None and str(prepend) not in sys.path: sys.path.insert(0,str(prepend))
    spec=importlib.util.spec_from_file_location(name,str(path)); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


class _UnusedSTDNLegacyGCNConv(nn.Module):
    """Guard for STDN's unused legacy PyG helper.

    The pinned STDN first_model used by this benchmark constructs its dynamic
    graph with the repository's own ``gcn`` implementation.  The top-level
    imports of PyG's GCNConv/Data only support a separate legacy ``GCN`` class
    that is not instantiated on this execution path.  Gadi's optional
    torch-scatter binary can require a newer GLIBC, so importing PyG solely for
    that unused helper creates a misleading warning.  We isolate the import and
    fail loudly if the unused helper is ever reached.
    """
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'The pinned STDN legacy torch_geometric.GCN helper was unexpectedly '
            'instantiated. The Shared Protocol STDN path must use the official '
            'first_model dynamic gcn implementation instead.'
        )


class _UnusedSTDNLegacyData:
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            'The pinned STDN legacy torch_geometric.Data helper was unexpectedly used.'
        )


@contextmanager
def _isolate_stdn_unused_pyg_imports():
    """Load pinned STDN without importing optional PyG compiled extensions."""
    names = ('torch_geometric', 'torch_geometric.nn', 'torch_geometric.data')
    old = {name: sys.modules.get(name) for name in names}
    root = types.ModuleType('torch_geometric')
    nn_mod = types.ModuleType('torch_geometric.nn')
    data_mod = types.ModuleType('torch_geometric.data')
    nn_mod.GCNConv = _UnusedSTDNLegacyGCNConv
    data_mod.Data = _UnusedSTDNLegacyData
    root.nn = nn_mod
    root.data = data_mod
    sys.modules['torch_geometric'] = root
    sys.modules['torch_geometric.nn'] = nn_mod
    sys.modules['torch_geometric.data'] = data_mod
    try:
        yield
    finally:
        for name in names:
            if old[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old[name]


def st_features(batch,steps_per_day,device):
    speed=batch['x'].to(device,non_blocking=True).float() # B,T,N
    B,T,N=speed.shape
    tod=(batch['tod_hist'].to(device).float()/float(steps_per_day)).unsqueeze(-1).expand(B,T,N)
    dow=batch['dow_hist'].to(device).float().unsqueeze(-1).expand(B,T,N)
    return torch.stack([speed,tod,dow],dim=-1)


class BaseAdapter:
    def __init__(self,model,device,mean,std): self.model=model.to(device); self.device=device; self.mean=float(mean); self.std=float(std)
    def raw(self,pred_norm): return pred_norm*self.std+self.mean
    def train_forward(self,batch,batches_seen): return self.forward(batch,batches_seen,True)
    def eval_forward(self,batch): return self.forward(batch,None,False)


class STAEformerAdapter(BaseAdapter):
    name='STAEformer'
    @classmethod
    def build(cls,vendor,num_nodes,device,mean,std,cfg,adj):
        mod=load_module('vendor_staeformer',Path(vendor)/'model'/'STAEformer.py',Path(vendor)/'model')
        m=mod.STAEformer(num_nodes=num_nodes,in_steps=12,out_steps=12,steps_per_day=288,input_dim=int(cfg.get('input_dim',3)),output_dim=1,
            input_embedding_dim=int(cfg['input_embedding_dim']),tod_embedding_dim=int(cfg['tod_embedding_dim']),dow_embedding_dim=int(cfg['dow_embedding_dim']),spatial_embedding_dim=int(cfg['spatial_embedding_dim']),adaptive_embedding_dim=int(cfg['adaptive_embedding_dim']),feed_forward_dim=int(cfg['feed_forward_dim']),num_heads=int(cfg['num_heads']),num_layers=int(cfg['num_layers']),dropout=float(cfg['dropout']),use_mixed_proj=bool(cfg['use_mixed_proj']))
        return cls(m,device,mean,std)
    def forward(self,batch,batches_seen=None,training=False):
        x=st_features(batch,288,self.device); return self.model(x).squeeze(-1)


class HimNetAdapter(BaseAdapter):
    name='HimNet'
    @classmethod
    def build(cls,vendor,num_nodes,device,mean,std,cfg,adj):
        mod=load_module('vendor_himnet',Path(vendor)/'model'/'HimNet.py',Path(vendor)/'model')
        m=mod.HimNet(num_nodes=num_nodes,input_dim=3,output_dim=1,out_steps=12,hidden_dim=int(cfg['hidden_dim']),num_layers=int(cfg['num_layers']),cheb_k=int(cfg['cheb_k']),ycov_dim=2,tod_embedding_dim=int(cfg['tod_embedding_dim']),dow_embedding_dim=int(cfg['dow_embedding_dim']),node_embedding_dim=int(cfg['node_embedding_dim']),st_embedding_dim=int(cfg['st_embedding_dim']),tf_decay_steps=int(cfg['tf_decay_steps']),use_teacher_forcing=bool(cfg['use_teacher_forcing']))
        return cls(m,device,mean,std)
    def forward(self,batch,batches_seen=None,training=False):
        x=st_features(batch,288,self.device); B,_,N,_=x.shape
        tod=(batch['tod_future'].to(self.device).float()/288.0).unsqueeze(-1).unsqueeze(-1).expand(B,12,N,1)
        dow=batch['dow_future'].to(self.device).float().unsqueeze(-1).unsqueeze(-1).expand(B,12,N,1)
        ycov=torch.cat([tod,dow],dim=-1)
        labels=batch['y_norm'].to(self.device).float().unsqueeze(-1) if training else None
        out=self.model(x,ycov,labels=labels,batches_seen=batches_seen)
        return out.squeeze(-1)


def canonicalize_bigst_prediction(pred, batch):
    """Convert BigST's native node-first forecast to the shared [B, P, N] contract.

    The pinned BigST training code transposes the target to [B, N, P].  The
    shared Backbone cache and every other adapter use [B, P, N].  Keep the
    official model untouched and perform the layout conversion only at this
    adapter boundary.
    """
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if pred.ndim != 3:
        raise RuntimeError(
            f'BigST prediction must be rank-3 after optional singleton squeeze; '
            f'got shape={tuple(pred.shape)}'
        )

    expected = tuple(batch['y_norm'].shape)  # shared contract: [B, P, N]
    if tuple(pred.shape) == expected:
        return pred

    b, p, n = expected
    if tuple(pred.shape) == (b, n, p):
        return pred.transpose(1, 2).contiguous()

    raise RuntimeError(
        'BigST prediction shape is incompatible with the shared forecast contract: '
        f'native={tuple(pred.shape)} expected={expected} or {(b, n, p)}'
    )


class BigSTAdapter(BaseAdapter):
    name='BigST'
    adapter_revision='bigst_bnp_to_bpn_hotfix1'
    @classmethod
    def build(cls,vendor,num_nodes,device,mean,std,cfg,adj):
        mod=load_module('vendor_bigst',Path(vendor)/'model.py',Path(vendor))
        a=SimpleNamespace(tau=float(cfg['tau']),num_layers=int(cfg['num_layers']),random_feature_dim=int(cfg['random_feature_dim']),use_residual=bool(cfg['use_residual']),use_bn=bool(cfg['use_bn']),use_spatial=bool(cfg['use_spatial']),use_long=bool(cfg['use_long']),dropout=float(cfg['dropout']),time_num=288,week_num=7,num_nodes=num_nodes,node_dim=int(cfg['node_dim']),time_dim=int(cfg['time_dim']),output_length=12,in_dim=3,hid_dim=int(cfg['hid_dim']))
        A=np.asarray(adj,np.float32); rs=A.sum(1,keepdims=True); An=A/np.maximum(rs,1e-6); support=torch.tensor(An,dtype=torch.float32,device=device); edges=torch.nonzero(support>0)
        m=mod.BigST(a,supports=[support],edge_indices=edges)
        return cls(m,device,mean,std)
    def forward(self,batch,batches_seen=None,training=False):
        x=st_features(batch,288,self.device).transpose(1,2) # B,N,T,D
        out=self.model(x)
        pred=out[0] if isinstance(out,tuple) else out
        return canonicalize_bigst_prediction(pred,batch)


class STDNAdapter(BaseAdapter):
    name='STDN'
    @classmethod
    def build(cls,vendor,num_nodes,device,mean,std,cfg,adj):
        # Keep the pinned vendor source untouched.  Its first_model path uses the
        # repository's own dynamic ``gcn``; isolate the unrelated legacy PyG
        # imports so Gadi does not try to load an incompatible torch-scatter .so.
        with _isolate_stdn_unused_pyg_imports():
            mod=load_module('vendor_stdn_model',Path(vendor)/'model.py',Path(vendor))
        mod.device=device
        cp=configparser.ConfigParser(); cp['Data']={'dataset_name':'TraffiDent','time_slice_size':'5','num_of_vertices':str(num_nodes)}
        cp['Training']={'gpu':'1' if device.type=='cuda' else '0','L':str(cfg['L']),'K':str(cfg['K']),'d':str(cfg['d']),'node_miss_rate':str(cfg['node_miss_rate']),'T_miss_len':str(cfg['T_miss_len']),'order':str(cfg['order']),'reference':str(cfg['reference']),'num_his':'12','num_pred':'12','in_channels':'1','out_channels':'1'}
        m=mod.make_model(cp,bn_decay=float(cfg['bn_decay']))
        # Use the official utils.cal_lape implementation.
        util=load_module('vendor_stdn_utils',Path(vendor)/'utils.py',Path(vendor)); A=(np.asarray(adj)!=0).astype(np.float32)
        # Official utils maps inf degree factors for isolated nodes back to zero
        # on the next line.  Suppress only NumPy's expected divide-by-zero notice.
        with np.errstate(divide='ignore', invalid='ignore'):
            lape=util.cal_lape(A.copy())
        if lape.shape[1]<32:
            pad=np.zeros((num_nodes,32-lape.shape[1]),np.float32); lape=np.concatenate([lape,pad],axis=1)
        obj=cls(m,device,mean,std); obj.lape=torch.tensor(lape[:,:32],dtype=torch.float32,device=device); return obj
    def forward(self,batch,batches_seen=None,training=False):
        X=batch['x'].to(self.device).float().unsqueeze(-1)
        # Official TE order is [day-of-week, time-of-day slot] for history+future.
        dow=torch.cat([batch['dow_hist'],batch['dow_future']],dim=1).to(self.device).long()
        tod=torch.cat([batch['tod_hist'],batch['tod_future']],dim=1).to(self.device).long()
        TE=torch.stack([dow,tod],dim=-1)
        # Pinned STDN wraps an already-existing tensor with torch.tensor().
        # That emits a copy-construction warning but does not change this path's
        # numerical result.  Filter only that exact vendor warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                'ignore',
                message=r'To copy construct from a tensor, it is recommended to use sourceTensor\.detach\(\)\.clone\(\).*',
                category=UserWarning,
            )
            return self.model(X,TE,self.lape,'train' if training else 'test').squeeze(-1)



def _node_coordinate_array(nodes, num_nodes):
    """Return deterministic [2,N] static coordinates for PatchSTG's official KDTree.

    SupportBench canonical nodes.csv normally contains lat/lng.  Missing values are
    filled only from static node metadata (postmile, then local index); no traffic
    values or Val/Test outcomes are used.
    """
    if nodes is None:
        raise RuntimeError('PatchSTG requires SupportBench nodes.csv metadata.')
    if len(nodes) != int(num_nodes):
        raise RuntimeError(f'PatchSTG nodes metadata length {len(nodes)} != N={num_nodes}')

    def col(names):
        for name in names:
            if name in nodes.columns:
                return np.asarray(nodes[name], dtype=np.float64)
        return np.full(num_nodes, np.nan, dtype=np.float64)

    lat = col(['lat', 'Lat', 'latitude'])
    lng = col(['lng', 'Lng', 'lon', 'longitude'])
    pm = col(['postmile', 'Abs PM', 'abs_pm'])
    idx = np.arange(num_nodes, dtype=np.float64)

    # Preserve all observed geographic metadata.  For rare missing entries,
    # derive a deterministic static fallback; this is not learned from outcomes.
    for arr, other, sign in ((lat, pm, 1.0), (lng, pm, -1.0)):
        bad = ~np.isfinite(arr)
        if np.any(bad):
            finite = arr[np.isfinite(arr)]
            base = float(np.median(finite)) if len(finite) else 0.0
            scale = 1e-4
            fallback = np.where(np.isfinite(other), other, idx)
            arr[bad] = base + sign * scale * fallback[bad]
    return np.stack([lat, lng], axis=0)


def _patchstg_layout(vendor, nodes, adj, num_nodes, cfg):
    util = load_module('vendor_patchstg_utils', Path(vendor)/'lib'/'utils.py', Path(vendor))
    recur = int(cfg['spatial_recur'])
    sps = int(cfg['spa_patchsize'])
    spn = int(cfg['spa_patchnum'])
    factors = int(cfg['factors'])
    if 2 ** recur != spn:
        raise RuntimeError(f'PatchSTG spa_patchnum={spn} must equal 2**spatial_recur={2**recur}')
    if spn % factors != 0:
        raise RuntimeError(f'PatchSTG spa_patchnum={spn} must be divisible by factors={factors}')
    locations = _node_coordinate_array(nodes, num_nodes)
    parts = util.kdTree(locations, recur, 0)
    max_leaf = max(len(x) for x in parts)
    if max_leaf > sps:
        raise RuntimeError(
            f'PatchSTG spa_patchsize={sps} is smaller than KDTree max leaf size={max_leaf}; '
            'increase spa_patchsize or spatial_recur.'
        )
    affinity = np.nan_to_num(np.abs(np.asarray(adj, dtype=np.float64)), nan=0.0, posinf=0.0, neginf=0.0)
    if affinity.shape != (num_nodes, num_nodes):
        raise RuntimeError(f'PatchSTG adjacency shape {affinity.shape} != {(num_nodes,num_nodes)}')
    np.fill_diagonal(affinity, 0.0)
    ori, reo, all_idx = util.reorderData(parts, affinity, sps)
    expected_padded = spn * sps
    if len(all_idx) != expected_padded:
        raise RuntimeError(f'PatchSTG padded index count {len(all_idx)} != {expected_padded}')
    import hashlib
    payload = np.concatenate([np.asarray(ori,np.int64), np.asarray(reo,np.int64), np.asarray(all_idx,np.int64)]).tobytes()
    fp = hashlib.sha256(payload).hexdigest()
    return np.asarray(ori,np.int64), np.asarray(reo,np.int64), np.asarray(all_idx,np.int64), fp


class PatchSTGAdapter(BaseAdapter):
    name = 'PatchSTG'
    adapter_revision = 'patchstg_official_kdtree_traffident_v1'

    @classmethod
    def build(cls, vendor, num_nodes, device, mean, std, cfg, adj, nodes, layout_override=None):
        try:
            import timm  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                'PatchSTG requires timm==1.0.12. Install requirements_extra.txt in the TFP environment.'
            ) from e
        mod = load_module('vendor_patchstg_model', Path(vendor)/'models'/'model.py', Path(vendor))
        if layout_override is None:
            ori, reo, all_idx, fp = _patchstg_layout(vendor, nodes, adj, num_nodes, cfg)
        else:
            from .patchstg import normalize_layout_override
            ori, reo, all_idx, fp = normalize_layout_override(layout_override, num_nodes, cfg)
        m = mod.PatchSTG(
            tem_patchsize=int(cfg['tem_patchsize']),
            tem_patchnum=int(cfg['tem_patchnum']),
            node_num=num_nodes,
            spa_patchsize=int(cfg['spa_patchsize']),
            spa_patchnum=int(cfg['spa_patchnum']),
            tod=int(cfg.get('tod_slots', 288)),
            dow=int(cfg.get('dow_slots', 7)),
            layers=int(cfg['layers']),
            factors=int(cfg['factors']),
            input_dims=int(cfg['input_dims']),
            node_dims=int(cfg['node_dims']),
            tod_dims=int(cfg['tod_dims']),
            dow_dims=int(cfg['dow_dims']),
            ori_parts_idx=ori,
            reo_parts_idx=reo,
            reo_all_idx=all_idx,
        )
        obj = cls(m, device, mean, std)
        obj.adapter_metadata = {
            'spatial_layout_fingerprint': fp,
            'spatial_recur': int(cfg['spatial_recur']),
            'spa_patchnum': int(cfg['spa_patchnum']),
            'spa_patchsize': int(cfg['spa_patchsize']),
            'factors': int(cfg['factors']),
            'coordinate_source': 'SupportBench/core/nodes.csv static lat/lng with deterministic static fallback only',
        }
        return obj

    def forward(self, batch, batches_seen=None, training=False):
        x = batch['x'].to(self.device, non_blocking=True).float().unsqueeze(-1)  # [B,T,N,1]
        B,T,N,_ = x.shape
        tod = batch['tod_hist'].to(self.device, non_blocking=True).long().unsqueeze(-1).expand(B,T,N)
        dow = batch['dow_hist'].to(self.device, non_blocking=True).long().unsqueeze(-1).expand(B,T,N)
        te = torch.stack([tod, dow], dim=-1)
        pred = self.model(x, te)
        if pred.ndim == 4 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        return pred



def stlaformer_features(batch, steps_per_day, days_per_week, device):
    """Build the pinned STLAformer input [B,T,N,2].

    Official STLAformer consumes traffic value plus a fractional time-of-week
    channel.  The same time-of-week channel is also used by the model's
    embedding lookup.  TraffiDent canonical TOD/DOW are converted without
    using future timestamps or target values.
    """
    speed = batch['x'].to(device, non_blocking=True).float()  # [B,T,N]
    B,T,N = speed.shape
    tod = batch['tod_hist'].to(device, non_blocking=True).long()
    dow = batch['dow_hist'].to(device, non_blocking=True).long()
    steps_per_week = int(steps_per_day) * int(days_per_week)
    tow_slot = dow * int(steps_per_day) + tod
    if torch.any(tow_slot < 0) or torch.any(tow_slot >= steps_per_week):
        raise RuntimeError(
            f'STLAformer time-of-week slot out of range [0,{steps_per_week}): '
            f'min={int(tow_slot.min())} max={int(tow_slot.max())}'
        )
    # The vendor model reconstructs the embedding index with
    # ``(tow * steps_per_week).long()``.  A plain float32 division can land
    # infinitesimally below an integer for a few slots (e.g. 255/2016), which
    # would shift the official embedding index by one after truncation.  Move
    # to the next representable float toward +inf so the encoded slot round-trip
    # is exact while preserving the fractional TOW semantics.
    frac = tow_slot.float() / float(steps_per_week)
    frac = torch.nextafter(frac, torch.full_like(frac, float('inf')))
    tow = frac.unsqueeze(-1).expand(B,T,N)
    return torch.stack([speed, tow], dim=-1)


class STLAformerAdapter(BaseAdapter):
    name = 'STLAformer'
    adapter_revision = 'stlaformer_official_tow_traffident_v1'

    @classmethod
    def build(cls, vendor, num_nodes, device, mean, std, cfg, adj):
        model_dir = Path(vendor) / 'model'
        model_file = model_dir / 'STLformer.py'
        if not model_file.exists():
            raise RuntimeError(f'Missing pinned STLAformer model source: {model_file}')
        mod = load_module('vendor_stlaformer', model_file, model_dir)
        m = mod.STLformer(
            num_nodes=num_nodes,
            in_steps=12,
            out_steps=12,
            steps_per_week=int(cfg.get('steps_per_week', 2016)),
            input_dim=int(cfg.get('input_dim', 2)),
            output_dim=1,
            input_embedding_dim=int(cfg.get('input_embedding_dim', 32)),
            tow_embedding_dim=int(cfg.get('tow_embedding_dim', 32)),
            feed_forward_dim=int(cfg.get('feed_forward_dim', 256)),
            num_heads=int(cfg.get('num_heads', 4)),
            num_layers=int(cfg.get('num_layers', 3)),
            dropout=float(cfg.get('dropout', 0.3)),
        )
        obj = cls(m, device, mean, std)
        obj.steps_per_day = int(cfg.get('tod_slots', 288))
        obj.days_per_week = int(cfg.get('dow_slots', 7))
        obj.adapter_metadata = {
            'official_class': 'STLformer',
            'input_contract': 'normalized_speed_plus_fractional_time_of_week',
            'steps_per_week': int(cfg.get('steps_per_week', 2016)),
            'future_time_features_used': False,
            'adjacency_used': False,
        }
        return obj

    def forward(self, batch, batches_seen=None, training=False):
        x = stlaformer_features(batch, self.steps_per_day, self.days_per_week, self.device)
        pred = self.model(x)
        if pred.ndim == 4 and pred.shape[-1] == 1:
            pred = pred.squeeze(-1)
        return pred


ADAPTERS={'STAEformer':STAEformerAdapter,'HimNet':HimNetAdapter,'BigST':BigSTAdapter,'STDN':STDNAdapter,'PatchSTG':PatchSTGAdapter,'STLAformer':STLAformerAdapter}

def build_adapter(name,vendor_root,num_nodes,device,mean,std,cfg,adj,nodes=None,layout_override=None):
    if name not in ADAPTERS: raise KeyError(name)
    vendor = Path(vendor_root) / name
    if name == 'PatchSTG':
        return ADAPTERS[name].build(vendor,num_nodes,device,mean,std,cfg,adj,nodes,layout_override=layout_override)
    return ADAPTERS[name].build(vendor,num_nodes,device,mean,std,cfg,adj)
