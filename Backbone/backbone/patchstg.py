from __future__ import annotations
import hashlib, importlib.util, sys
from pathlib import Path
import numpy as np
import torch


def load_module(name, path, prepend=None):
    if prepend is not None and str(prepend) not in sys.path:
        sys.path.insert(0, str(prepend))
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PatchSTGAdapter:
    name = 'PatchSTG'
    adapter_revision = 'patchstg_official_kdtree_traffident_v2_layout_snapshot'
    def __init__(self, model, device, mean, std):
        self.model = model.to(device); self.device = device; self.mean = float(mean); self.std = float(std)
    def raw(self, pred_norm): return pred_norm * self.std + self.mean
    def train_forward(self, batch, batches_seen=None): return self.forward(batch)
    def eval_forward(self, batch): return self.forward(batch)
    def forward(self, batch):
        x = batch['x'].to(self.device, non_blocking=True).float().unsqueeze(-1)
        B,T,N,_ = x.shape
        tod = batch['tod_hist'].to(self.device, non_blocking=True).long().unsqueeze(-1).expand(B,T,N)
        dow = batch['dow_hist'].to(self.device, non_blocking=True).long().unsqueeze(-1).expand(B,T,N)
        te = torch.stack([tod, dow], dim=-1)
        pred = self.model(x, te)
        if pred.ndim == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
        return pred


def _node_coordinate_array(nodes, num_nodes):
    if nodes is None: raise RuntimeError('PatchSTG requires SupportBench nodes.csv metadata.')
    if len(nodes) != int(num_nodes): raise RuntimeError(f'nodes metadata length {len(nodes)} != N={num_nodes}')
    def col(names):
        for name in names:
            if name in nodes.columns: return np.asarray(nodes[name], dtype=np.float64)
        return np.full(num_nodes, np.nan, dtype=np.float64)
    lat = col(['lat','Lat','latitude']); lng = col(['lng','Lng','lon','longitude']); pm = col(['postmile','Abs PM','abs_pm']); idx = np.arange(num_nodes, dtype=np.float64)
    for arr, other, sign in ((lat,pm,1.0),(lng,pm,-1.0)):
        bad = ~np.isfinite(arr)
        if np.any(bad):
            finite = arr[np.isfinite(arr)]; base = float(np.median(finite)) if len(finite) else 0.0
            fallback = np.where(np.isfinite(other), other, idx)
            arr[bad] = base + sign * 1e-4 * fallback[bad]
    return np.stack([lat,lng], axis=0)


def spatial_layout_fingerprint(ori, reo, all_idx):
    payload = np.concatenate([
        np.asarray(ori, np.int64).reshape(-1),
        np.asarray(reo, np.int64).reshape(-1),
        np.asarray(all_idx, np.int64).reshape(-1),
    ]).tobytes()
    return hashlib.sha256(payload).hexdigest()


def compute_patchstg_layout(vendor, nodes, adj, num_nodes, cfg):
    util = load_module('vendor_patchstg_utils', Path(vendor)/'lib'/'utils.py', Path(vendor))
    recur = int(cfg['spatial_recur']); sps = int(cfg['spa_patchsize']); spn = int(cfg['spa_patchnum']); factors = int(cfg['factors'])
    if 2 ** recur != spn: raise RuntimeError(f'spa_patchnum={spn} must equal 2**spatial_recur={2**recur}')
    if spn % factors != 0: raise RuntimeError(f'spa_patchnum={spn} must be divisible by factors={factors}')
    locations = _node_coordinate_array(nodes, num_nodes)
    parts = util.kdTree(locations, recur, 0)
    max_leaf = max(len(x) for x in parts)
    if max_leaf > sps: raise RuntimeError(f'spa_patchsize={sps} smaller than KDTree max leaf={max_leaf}')
    affinity = np.nan_to_num(np.abs(np.asarray(adj, dtype=np.float64)), nan=0.0, posinf=0.0, neginf=0.0)
    if affinity.shape != (num_nodes,num_nodes): raise RuntimeError(f'adjacency shape {affinity.shape} != {(num_nodes,num_nodes)}')
    np.fill_diagonal(affinity, 0.0)
    ori, reo, all_idx = util.reorderData(parts, affinity, sps)
    if len(all_idx) != spn * sps: raise RuntimeError(f'padded index count {len(all_idx)} != {spn*sps}')
    ori = np.asarray(ori, np.int64); reo = np.asarray(reo, np.int64); all_idx = np.asarray(all_idx, np.int64)
    return ori, reo, all_idx, spatial_layout_fingerprint(ori, reo, all_idx)


def normalize_layout_override(layout, num_nodes, cfg):
    if layout is None:
        return None
    if not isinstance(layout, dict):
        raise TypeError(f'layout override must be a dict, got {type(layout)!r}')
    required = ('ori_parts_idx', 'reo_parts_idx', 'reo_all_idx')
    missing = [k for k in required if k not in layout]
    if missing:
        raise RuntimeError(f'layout override missing keys: {missing}')
    ori = np.asarray(layout['ori_parts_idx'], dtype=np.int64).reshape(-1)
    reo = np.asarray(layout['reo_parts_idx'], dtype=np.int64).reshape(-1)
    all_idx = np.asarray(layout['reo_all_idx'], dtype=np.int64).reshape(-1)
    expected_padded = int(cfg['spa_patchnum']) * int(cfg['spa_patchsize'])
    if len(all_idx) != expected_padded:
        raise RuntimeError(f'layout override padded index count {len(all_idx)} != {expected_padded}')
    if len(ori) != int(num_nodes) or len(reo) != int(num_nodes):
        raise RuntimeError(
            f'layout override original/reorder lengths {(len(ori), len(reo))} != num_nodes={num_nodes}'
        )
    if np.any(ori < 0) or np.any(ori >= int(num_nodes)):
        raise RuntimeError('layout override ori_parts_idx contains out-of-range node indices')
    if np.any(reo < 0) or np.any(reo >= expected_padded):
        raise RuntimeError('layout override reo_parts_idx contains out-of-range padded indices')
    if np.any(all_idx < 0) or np.any(all_idx >= int(num_nodes)):
        raise RuntimeError('layout override reo_all_idx contains out-of-range node indices')
    fp = spatial_layout_fingerprint(ori, reo, all_idx)
    expected_fp = layout.get('spatial_layout_fingerprint')
    if expected_fp not in (None, '') and str(expected_fp) != fp:
        raise RuntimeError(f'layout override fingerprint mismatch: declared={expected_fp} computed={fp}')
    return ori, reo, all_idx, fp


def layout_from_checkpoint(ck):
    layout = ck.get('spatial_layout') if isinstance(ck, dict) else None
    if not isinstance(layout, dict):
        return None
    return {
        'ori_parts_idx': layout.get('ori_parts_idx'),
        'reo_parts_idx': layout.get('reo_parts_idx'),
        'reo_all_idx': layout.get('reo_all_idx'),
        'spatial_layout_fingerprint': layout.get('spatial_layout_fingerprint') or (
            ck.get('adapter_metadata', {}).get('spatial_layout_fingerprint')
            if isinstance(ck.get('adapter_metadata'), dict) else None
        ),
    }


def load_layout_snapshot(path):
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        layout = {
            'ori_parts_idx': np.asarray(z['ori_parts_idx'], dtype=np.int64),
            'reo_parts_idx': np.asarray(z['reo_parts_idx'], dtype=np.int64),
            'reo_all_idx': np.asarray(z['reo_all_idx'], dtype=np.int64),
        }
        if 'spatial_layout_fingerprint' in z.files:
            raw = z['spatial_layout_fingerprint']
            layout['spatial_layout_fingerprint'] = str(raw.item() if getattr(raw, 'ndim', 0) == 0 else raw.tolist())
    return layout


def save_layout_snapshot(path, layout, metadata=None):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    ori = np.asarray(layout['ori_parts_idx'], dtype=np.int64).reshape(-1)
    reo = np.asarray(layout['reo_parts_idx'], dtype=np.int64).reshape(-1)
    all_idx = np.asarray(layout['reo_all_idx'], dtype=np.int64).reshape(-1)
    fp = layout.get('spatial_layout_fingerprint') or spatial_layout_fingerprint(ori, reo, all_idx)
    np.savez_compressed(
        p,
        ori_parts_idx=ori,
        reo_parts_idx=reo,
        reo_all_idx=all_idx,
        spatial_layout_fingerprint=np.asarray(str(fp)),
    )
    return str(fp)



def checkpoint_layout_or_sidecar(ck, checkpoint_path=None):
    """Return an embedded/sibling recovered layout, or None for legacy checkpoints."""
    embedded = layout_from_checkpoint(ck)
    if embedded is not None:
        return embedded, 'checkpoint_embedded'
    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        for name in ('spatial_layout.npz', 'PATCHSTG_LAYOUT.npz'):
            p = cp.parent / name
            if p.exists():
                return load_layout_snapshot(p), f'snapshot:{p.resolve()}'
    return None, 'current_benchmark'


def require_checkpoint_layout_match(ck, adapter, *, context='PatchSTG evaluation'):
    old = None
    if isinstance(ck, dict) and isinstance(ck.get('adapter_metadata'), dict):
        old = ck['adapter_metadata'].get('spatial_layout_fingerprint')
    new = getattr(adapter, 'adapter_metadata', {}).get('spatial_layout_fingerprint')
    if old and new and str(old) != str(new):
        raise RuntimeError(
            f'{context} spatial layout mismatch: checkpoint={old} effective={new}. '
            'Recover the historical spatial_layout.npz before evaluation.'
        )
    return old, new

def build_patchstg(vendor, num_nodes, device, mean, std, cfg, adj, nodes, layout_override=None):
    try: import timm  # noqa
    except Exception as e: raise RuntimeError('PatchSTG requires timm==1.0.12; install requirements_extra.txt') from e
    mod = load_module('vendor_patchstg_model', Path(vendor)/'models'/'model.py', Path(vendor))
    source = 'current_benchmark'
    if layout_override is None:
        ori, reo, all_idx, fp = compute_patchstg_layout(vendor, nodes, adj, num_nodes, cfg)
    else:
        ori, reo, all_idx, fp = normalize_layout_override(layout_override, num_nodes, cfg)
        source = 'checkpoint_or_snapshot'
    model = mod.PatchSTG(
        tem_patchsize=int(cfg['tem_patchsize']), tem_patchnum=int(cfg['tem_patchnum']), node_num=num_nodes,
        spa_patchsize=int(cfg['spa_patchsize']), spa_patchnum=int(cfg['spa_patchnum']),
        tod=int(cfg.get('tod_slots',288)), dow=int(cfg.get('dow_slots',7)), layers=int(cfg['layers']),
        factors=int(cfg['factors']), input_dims=int(cfg['input_dims']), node_dims=int(cfg['node_dims']),
        tod_dims=int(cfg['tod_dims']), dow_dims=int(cfg['dow_dims']), ori_parts_idx=ori,
        reo_parts_idx=reo, reo_all_idx=all_idx,
    )
    obj = PatchSTGAdapter(model, device, mean, std)
    obj.spatial_layout = {
        'ori_parts_idx': ori.copy(),
        'reo_parts_idx': reo.copy(),
        'reo_all_idx': all_idx.copy(),
        'spatial_layout_fingerprint': fp,
    }
    obj.adapter_metadata = {
        'spatial_layout_fingerprint': fp, 'spatial_recur': int(cfg['spatial_recur']),
        'spa_patchnum': int(cfg['spa_patchnum']), 'spa_patchsize': int(cfg['spa_patchsize']),
        'factors': int(cfg['factors']), 'spatial_layout_source': source,
        'coordinate_source': 'SupportBench/core/nodes.csv static lat/lng with deterministic static fallback only',
    }
    return obj
