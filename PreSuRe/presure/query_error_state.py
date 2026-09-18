from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .models import NodeSensingEncoder
from .features import _masked_slope, _trailing_missing, row_normalize_adjacency, NODE_SENSE_DIM

RECENT_ERROR_DIM = 8
PHYSICAL_TRANSITION_DIM = 12


def _last_observed(x, obs):
    B,H,N=x.shape
    idx=np.where(obs, np.arange(H,dtype=np.int16)[None,:,None], -1).max(1)
    safe=np.maximum(idx,0)
    out=np.take_along_axis(x, safe[:,None,:], axis=1)[:,0,:]
    return np.where(idx>=0,out,0.0).astype(np.float32)


def _last_k_observed(x, obs, k=3):
    B,H,N=x.shape
    vals=np.zeros((B,k,N),np.float32); ok=np.zeros((B,k,N),bool)
    for j in range(k):
        rank=np.cumsum(obs[:,::-1,:],axis=1)[:,::-1,:]
        target=j+1
        mask=obs & (rank==target)
        has=mask.any(1); idx=mask.argmax(1)
        vals[:,j,:]=np.where(has,np.take_along_axis(x,idx[:,None,:],axis=1)[:,0,:],0.0)
        ok[:,j,:]=has
    return vals,ok


def build_physical_transition_features(x, obs, adj):
    """Strictly query-history-only physical transition descriptors [M,N,12]."""
    xx=np.asarray(x,np.float32); mm=np.asarray(obs,bool); A=row_normalize_adjacency(adj)
    B,H,N=xx.shape
    xobs=xx*mm.astype(np.float32)
    den=mm.sum(1).clip(min=1)
    mean=xobs.sum(1)/den
    var=(((xx-mean[:,None,:])**2)*mm).sum(1)/den
    std=np.sqrt(np.maximum(var,0)).astype(np.float32)
    slope=_masked_slope(xx,mm)
    last=_last_observed(xx,mm)
    vals,ok=_last_k_observed(xx,mm,3)
    d1=np.where(ok[:,0]&ok[:,1], vals[:,0]-vals[:,1], 0.0)
    d2=np.where(ok[:,0]&ok[:,1]&ok[:,2], vals[:,0]-2*vals[:,1]+vals[:,2], 0.0)
    shock=last-mean
    nbr_last=(last @ A.T).astype(np.float32)
    nbr_slope=(slope @ A.T).astype(np.float32)
    spatial_grad=nbr_last-last
    spatial_slope=nbr_slope-slope
    obsfrac=mm.mean(1).astype(np.float32)
    trail=_trailing_missing(mm)
    current=mm[:,-1,:].astype(np.float32)
    nbr_current=(current @ A.T).astype(np.float32)
    out=np.stack([last,mean,std,slope,d1,d2,shock,spatial_grad,spatial_slope,obsfrac,trail,nbr_current],-1)
    return np.nan_to_num(out.astype(np.float32),nan=0.0,posinf=8.0,neginf=-8.0)


def infer_row_stride(tod_hist, dow_hist, max_stride=144):
    tod=np.asarray(tod_hist[:,-1],np.int64); dow=np.asarray(dow_hist[:,-1],np.int64)
    week=dow*288+tod
    d=(week[1:]-week[:-1])%(7*288)
    d=d[(d>0)&(d<=int(max_stride))]
    if len(d)==0: return None,0.0
    vals,cnt=np.unique(d,return_counts=True); j=int(np.argmax(cnt))
    return int(vals[j]),float(cnt[j]/len(d))


def _absolute_anchor_steps(ds, step_minutes=5):
    """Best-effort absolute query anchor from the cache manifest, otherwise None.

    No label/future field is inspected. The returned values only establish chronology.
    """
    meta=getattr(ds,'meta',None)
    if meta is None: return None,None
    numeric_names=['query_index','query_idx','anchor_index','anchor_idx','history_end_idx','history_end_index','time_index','t_idx','forecast_start_idx']
    for c in numeric_names:
        if c in meta.columns:
            try:
                z=np.asarray(meta[c],np.float64)
                if np.isfinite(z).mean()>0.99 and len(np.unique(z[np.isfinite(z)]))>0.9*len(z):
                    return z, f'manifest_numeric:{c}'
            except Exception: pass
    datetime_names=['query_time','anchor_time','history_end_time','timestamp','datetime','date_time','forecast_start_time']
    for c in datetime_names:
        if c in meta.columns:
            try:
                import pandas as pd
                z=pd.to_datetime(meta[c],errors='coerce')
                if float(z.notna().mean())>0.99:
                    ns=z.astype('int64').to_numpy(dtype=np.float64)
                    return ns/(60e9*float(step_minutes)), f'manifest_datetime:{c}'
            except Exception: pass
    return None,None


def build_recent_causal_error_features(ds, base_pred, expected_stride=12, min_stride_confidence=0.98,
                                       max_lag_queries=4, max_gap_steps=48, step_minutes=5):
    """Build recent *realized* backbone-error state without touching current-query future.

    Preferred chronology comes from an absolute cache-manifest anchor. If unavailable, a
    conservative TOD/DOW row-order proof is attempted. A previous query contributes only
    horizons already realized by the current query time. Current-query y is never read into
    the feature vector.
    """
    y=np.asarray(ds.y,np.float32); base=np.asarray(base_pred,np.float32); valid=np.asarray(ds.valid,bool)
    M,P,N=y.shape; err=y-base
    abs_anchor,source=_absolute_anchor_steps(ds,step_minutes)
    if abs_anchor is None:
        stride,conf=infer_row_stride(ds.tod,ds.dow,max_stride=max(288,int(expected_stride)*4))
        if stride is None or conf < float(min_stride_confidence) or stride<=0:
            return np.zeros((M,N,RECENT_ERROR_DIM),np.float32), {
                'available':False,'reason':f'cannot prove chronology (stride={stride}, confidence={conf:.4f})',
                'chronology_source':'none','inferred_stride':stride,'stride_confidence':conf,
            }
        if int(expected_stride)>0 and int(stride)!=int(expected_stride):
            return np.zeros((M,N,RECENT_ERROR_DIM),np.float32), {
                'available':False,'reason':f'inferred stride {stride} != expected {expected_stride}',
                'chronology_source':'tod_dow_row_order','inferred_stride':stride,'stride_confidence':conf,
            }
        week=np.asarray(ds.dow[:,-1],np.int64)*288+np.asarray(ds.tod[:,-1],np.int64)
        # unwrap using the proven modal stride; exact checks below stop at sequence boundaries.
        abs_anchor=np.zeros(M,np.float64); abs_anchor[0]=week[0]
        for i in range(1,M):
            d=(int(week[i])-int(week[i-1]))%(7*288)
            abs_anchor[i]=abs_anchor[i-1]+d
        source='tod_dow_row_order'
        conf=float(conf)
    else:
        d=np.diff(abs_anchor); good=d[(d>0)&(d<=int(max_gap_steps))]
        conf=float(len(good)/max(len(d),1))
        stride=int(np.median(good)) if len(good) else None

    out=np.zeros((M,N,RECENT_ERROR_DIM),np.float32); denom=float(max(P*max_lag_queries,1)); usable_rows=0
    for i in range(M):
        vals=[[] for _ in range(N)]; latest=np.zeros(N,np.float32); latest_ok=np.zeros(N,bool)
        used_any=False
        for lag in range(1,int(max_lag_queries)+1):
            j=i-lag
            if j<0: break
            delta=int(round(float(abs_anchor[i]-abs_anchor[j])))
            if delta<=0 or delta>int(max_gap_steps): break
            realized=min(P,delta)
            if realized<=0: continue
            ev=err[j,:realized,:]; vv=valid[j,:realized,:]; used_any=True
            if not latest_ok.all():
                for n in range(N):
                    if latest_ok[n]: continue
                    ids=np.where(vv[:,n])[0]
                    if len(ids): latest[n]=ev[ids[-1],n]; latest_ok[n]=True
            for n in range(N):
                z=ev[:,n][vv[:,n]]
                if len(z): vals[n].extend(z.tolist())
        if used_any: usable_rows+=1
        for n in range(N):
            if not vals[n]: continue
            z=np.asarray(vals[n],np.float32)
            out[i,n,0]=latest[n] if latest_ok[n] else z[-1]
            out[i,n,1]=float(z.mean()); out[i,n,2]=float(np.mean(np.abs(z))); out[i,n,3]=float(z.std())
            out[i,n,4]=float(np.mean(z>0)); out[i,n,5]=float(np.max(np.abs(z)))
            out[i,n,6]=min(float(len(z))/denom,1.0); out[i,n,7]=1.0
    available=usable_rows>=max(1,int(np.ceil(0.02*M)))
    if not available: out.fill(0.0)
    return np.nan_to_num(out,nan=0.0,posinf=8.0,neginf=-8.0), {
        'available':bool(available),'reason':'causally realized previous-query residuals' if available else 'too few temporally adjacent prior queries',
        'chronology_source':source,'inferred_stride':stride,'stride_confidence':conf,
        'max_lag_queries':int(max_lag_queries),'max_gap_steps':int(max_gap_steps),'usable_row_fraction':float(usable_rows/max(M,1)),
    }


class QueryErrorStateNet(nn.Module):
    """v0.9.5 query error-state encoder with explicit predictive/causal/physical evidence."""
    def __init__(self,din,num_nodes,P,base_start,hidden_dim,recent_dim,physical_dim,
                 use_hidden=True,use_recent=True,use_physical=True,sense_dim=NODE_SENSE_DIM,
                 encoder_hidden=192,state_dim=96,node_dim=24,horizon_dim=24,extra_dim=48,hidden=192,dropout=0.1):
        super().__init__(); self.P=int(P); self.N=int(num_nodes); self.base_start=int(base_start); self.sense_dim=int(sense_dim)
        self.use_hidden=bool(use_hidden); self.use_recent=bool(use_recent); self.use_physical=bool(use_physical)
        self.encoder=NodeSensingEncoder(int(din),self.N,hidden=int(encoder_hidden),demb=int(state_dim),node_dim=int(node_dim),dropout=float(dropout))
        def enc(d): return nn.Sequential(nn.LayerNorm(int(d)),nn.Linear(int(d),int(extra_dim)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(extra_dim),int(extra_dim)),nn.GELU())
        self.hidden_enc=enc(hidden_dim) if self.use_hidden else None
        self.recent_enc=enc(recent_dim) if self.use_recent else None
        self.phys_enc=enc(physical_dim) if self.use_physical else None
        self.h_emb=nn.Embedding(self.P,int(horizon_dim))
        local_dim=4+self.sense_dim
        total=int(state_dim)+int(horizon_dim)+local_dim+int(extra_dim)*(self.use_hidden+self.use_recent+self.use_physical)
        self.trunk=nn.Sequential(nn.Linear(total,int(hidden)),nn.LayerNorm(int(hidden)),nn.GELU(),nn.Dropout(float(dropout)),nn.Linear(int(hidden),96),nn.GELU())
        self.error_head=nn.Linear(96,1); self.direction_head=nn.Linear(96,1)

    def _local(self,x):
        base=x[:,:,self.base_start:self.base_start+self.P]
        prev=torch.cat([base[:,:,:1],base[:,:,:-1]],-1); nxt=torch.cat([base[:,:,1:],base[:,:,-1:]],-1)
        dprev=base-prev; dnext=nxt-base; curve=dnext-dprev
        sense=x[:,:,-self.sense_dim:][:,:,None,:].expand(-1,-1,self.P,-1)
        return torch.cat([torch.stack([base,dprev,dnext,curve],-1),sense],-1)

    def forward(self,x,hidden_state,recent_error,physical):
        B,N,_=x.shape; z=self.encoder(x); parts=[z]
        if self.use_hidden: parts.append(self.hidden_enc(hidden_state))
        if self.use_recent: parts.append(self.recent_enc(recent_error))
        if self.use_physical: parts.append(self.phys_enc(physical))
        node=torch.cat(parts,-1)[:,:,None,:].expand(-1,-1,self.P,-1)
        he=self.h_emb.weight[None,None,:,:].expand(B,N,-1,-1)
        h=self.trunk(torch.cat([node,he,self._local(x)],-1))
        err=self.error_head(h).squeeze(-1).permute(0,2,1).contiguous()
        logit=self.direction_head(h).squeeze(-1).permute(0,2,1).contiguous()
        return err,logit

# -----------------------------------------------------------------------------
# v0.9.5.1 repaired causal backbone-error state
# -----------------------------------------------------------------------------
RECENT_ERROR_REPAIRED_DIM = 12


def _manifest_anchor_steps(ds, history: int):
    """Resolve absolute query anchor steps from audit-friendly manifest fields.

    ``window_start`` is the preferred SupportBench contract field. Retrieval already
    treats it as an absolute underlying-window coordinate, so the query anchor is
    ``window_start + history``. We deliberately do not infer chronology from row order
    when this field is available because multiple synthetic views can be interleaved.
    """
    meta = getattr(ds, 'meta', None)
    if meta is None:
        return None, {'available': False, 'reason': 'cache split has no manifest table'}
    if 'window_start' in meta.columns:
        try:
            w = np.asarray(meta['window_start'], np.float64)
            good = np.isfinite(w)
            if good.mean() > 0.99:
                return w + float(history), {
                    'available': True,
                    'source': 'manifest_window_start_plus_history',
                    'history': int(history),
                    'finite_fraction': float(good.mean()),
                }
        except Exception:
            pass
    # Fall back to explicit anchor fields only. These are stronger than a row-order/TOD
    # guess and remain valid even when benchmark views are shuffled.
    z, source = _absolute_anchor_steps(ds)
    if z is not None:
        return np.asarray(z, np.float64), {
            'available': True, 'source': source, 'history': int(history),
            'finite_fraction': float(np.isfinite(z).mean()),
        }
    return None, {
        'available': False,
        'reason': 'manifest has neither window_start nor an explicit absolute query anchor',
        'columns': list(meta.columns),
    }


def _canonical_rows_by_anchor(ds, anchors):
    """Choose one deterministic benchmark view per historical anchor.

    SupportBench may contain multiple incomplete-sensing views of the same underlying
    window. Averaging all of those counterfactual views would overstate what one online
    deployment could observe. v0.9.5.1 therefore uses exactly one deterministic row per
    historical anchor: the lexicographically smallest ``view_group_id`` when present,
    otherwise the smallest row index. The policy never inspects future error magnitude.
    """
    anchors = np.asarray(anchors, np.float64)
    meta = getattr(ds, 'meta', None)
    valid = np.where(np.isfinite(anchors))[0]
    groups = {}
    for i in valid:
        groups.setdefault(float(anchors[i]), []).append(int(i))
    out = []
    for a, ids in groups.items():
        if meta is not None and 'view_group_id' in meta.columns:
            ids = sorted(ids, key=lambda j: (str(meta.iloc[j]['view_group_id']), j))
        else:
            ids = sorted(ids)
        out.append((float(a), int(ids[0]), len(ids)))
    out.sort(key=lambda x: x[0])
    return out


def _node_seq_features(seq, ages, max_items):
    """Convert chronological query-level node bias sequence to 12 safe features."""
    # seq [L,N] with NaN for unavailable node errors; ages [L] in base time steps.
    L, N = seq.shape if seq.ndim == 2 else (0, 0)
    out = np.zeros((N, RECENT_ERROR_REPAIRED_DIM), np.float32)
    if L == 0:
        return out
    for n in range(N):
        ok = np.isfinite(seq[:, n])
        if not np.any(ok):
            continue
        z = seq[ok, n].astype(np.float32)
        ag = np.asarray(ages, np.float32)[ok]
        z6 = z[-6:]; z3 = z[-3:]
        # Query-level bias trend over the last <=6 realised prior forecasts.
        if len(z6) >= 2:
            xx = np.arange(len(z6), dtype=np.float32)
            xc = xx - xx.mean(); den = float(np.sum(xc*xc))
            slope = float(np.sum(xc*(z6-z6.mean()))/den) if den > 0 else 0.0
        else:
            slope = 0.0
        signs = np.sign(z6)
        out[n, 0] = float(z[-1])                         # latest realised bias
        out[n, 1] = float(z3.mean())                     # mean last 3 query biases
        out[n, 2] = float(z6.mean())                     # mean last 6 query biases
        out[n, 3] = float(np.mean(np.abs(z3)))           # recent MAE
        out[n, 4] = float(z6.std())                      # recent bias volatility
        out[n, 5] = float(np.mean(z6 > 0))               # underprediction fraction
        out[n, 6] = float(abs(np.mean(signs)))           # sign persistence [0,1]
        out[n, 7] = slope                                # bias trend
        out[n, 8] = float(np.max(np.abs(z6)))            # maximum recent |bias|
        out[n, 9] = min(float(len(z))/max(float(max_items), 1.0), 1.0)
        out[n,10] = min(float(ag[-1])/288.0, 2.0) if len(ag) else 2.0  # latest age in days
        out[n,11] = 1.0                                  # availability
    return out


def build_repaired_causal_error_features(pool_ds, pool_base_pred, target_ds=None,
                                          history: int = 12, max_lag_queries: int = 6,
                                          max_gap_steps: int = 288,
                                          target_uses_pool_labels_only: bool = True):
    """Build v0.9.5.1 recent backbone-error features from proven manifest chronology.

    Parameters
    ----------
    pool_ds / pool_base_pred:
        Historical forecast rows whose realised outcomes may be used. For Train targets
        this is the Train split itself. For Validation targets the caller passes Train as
        the pool, so Validation labels never become features for other Validation rows.
    target_ds:
        Query rows receiving features. ``None`` means ``pool_ds`` (Train causal mode).

    Causality
    ---------
    For a prior query with anchor ``a_j`` and a current target anchor ``a_i``, only
    horizons with target time <= ``a_i`` are aggregated. Rows at the same anchor are
    excluded. Current-target future labels are never read.
    """
    if target_ds is None:
        target_ds = pool_ds
    pool_anchor, pman = _manifest_anchor_steps(pool_ds, history)
    target_anchor, tman = _manifest_anchor_steps(target_ds, history)
    N = int(pool_ds.y.shape[2]); M = int(target_ds.y.shape[0])
    out = np.zeros((M, N, RECENT_ERROR_REPAIRED_DIM), np.float32)
    if pool_anchor is None or target_anchor is None:
        return out, {
            'available': False,
            'reason': 'cannot establish manifest chronology',
            'pool_chronology': pman, 'target_chronology': tman,
            'validation_history_policy': 'pool labels only',
        }

    y = np.asarray(pool_ds.y, np.float32)
    base = np.asarray(pool_base_pred, np.float32)
    valid = np.asarray(pool_ds.valid, bool)
    err = y - base
    canon = _canonical_rows_by_anchor(pool_ds, pool_anchor)
    if not canon:
        return out, {'available': False, 'reason': 'no canonical chronological pool rows'}
    times = np.asarray([x[0] for x in canon], np.float64)
    rows = np.asarray([x[1] for x in canon], np.int64)
    duplicate_counts = np.asarray([x[2] for x in canon], np.int64)
    usable = 0

    for i, ai in enumerate(np.asarray(target_anchor, np.float64)):
        if not np.isfinite(ai):
            continue
        # Strictly earlier anchors only. A prior forecast can contribute only realised
        # horizons; same-anchor counterfactual views can never be used.
        end = int(np.searchsorted(times, ai, side='left'))
        if end <= 0:
            continue
        cand = []
        for k in range(end-1, -1, -1):
            delta = float(ai - times[k])
            if delta <= 0:
                continue
            if delta > float(max_gap_steps):
                break
            j = int(rows[k])
            realised = min(int(err.shape[1]), int(np.floor(delta + 1e-8)))
            if realised <= 0:
                continue
            ev = err[j, :realised, :]
            vv = valid[j, :realised, :]
            # One query-level bias per node; later features operate across prior queries,
            # not across duplicated synthetic views or unrealisable future horizons.
            den = vv.sum(0)
            bias = np.full(N, np.nan, np.float32)
            ok = den > 0
            if np.any(ok):
                bias[ok] = (ev * vv.astype(np.float32)).sum(0)[ok] / den[ok]
            cand.append((bias, delta))
            if len(cand) >= int(max_lag_queries):
                break
        if not cand:
            continue
        cand.reverse()  # chronological oldest -> newest
        seq = np.stack([x[0] for x in cand], 0)
        ages = np.asarray([x[1] for x in cand], np.float32)
        out[i] = _node_seq_features(seq, ages, int(max_lag_queries))
        if np.any(out[i, :, -1] > 0):
            usable += 1

    frac = float(usable / max(M, 1))
    available = usable > 0 and frac >= 0.02
    if not available:
        out.fill(0.0)
    return np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0), {
        'available': bool(available),
        'reason': 'manifest-window causal realised prior forecast biases' if available else 'too few targets have a realised prior forecast in the allowed gap',
        'pool_chronology': pman,
        'target_chronology': tman,
        'canonical_view_policy': 'one deterministic row per anchor; lexicographic view_group_id then row index; future error never used for choice',
        'unique_pool_anchors': int(len(canon)),
        'mean_views_per_anchor': float(duplicate_counts.mean()) if len(duplicate_counts) else 0.0,
        'max_lag_queries': int(max_lag_queries),
        'max_gap_steps': int(max_gap_steps),
        'usable_row_fraction': frac,
        'validation_history_policy': 'features use pool labels only; caller should use Train as pool for Validation',
        'feature_dim': RECENT_ERROR_REPAIRED_DIM,
    }
