from __future__ import annotations
import numpy as np
import torch


def perfect_completion_item(item):
    """Post-prediction diagnostic only: restore artificial missing entries from x_full."""
    qx=item['x_obs'].numpy().astype(np.float32).copy()
    xf=item['x_full'].numpy().astype(np.float32)
    target=item['artificial_missing'].numpy().astype(bool) & np.isfinite(xf)
    qx[target]=xf[target]
    out=dict(item); out['x_obs']=torch.from_numpy(qx)
    return out,target


def retrieved_completion_oracles(item, completion):
    """Post-hoc completion opportunity from values already present in retrieved Top-K.

    `case` chooses one coherent retrieved case. `entry` chooses the closest retrieved
    value independently per missing entry. Hidden current history is used only here,
    after the real prediction. Reconstruction MAE is evaluated over the entire target
    missing set, so low-coverage candidates cannot look artificially good.
    """
    qx=item['x_obs'].numpy().astype(np.float32)
    xf=item['x_full'].numpy().astype(np.float32)
    target=item['artificial_missing'].numpy().astype(bool) & np.isfinite(xf)
    vals=np.asarray(completion.get('candidate_values',[]),np.float32)
    avail=np.asarray(completion.get('candidate_available',[]),bool)
    if vals.ndim!=3 or len(vals)==0 or not np.any(target): return None

    best_j=None; best_mae=float('inf'); best_aug=None; best_fill=None
    for j in range(len(vals)):
        aug=qx.copy(); m=target & avail[j] & np.isfinite(vals[j]); aug[m]=vals[j][m]
        mae=float(np.mean(np.abs(aug[target]-xf[target])))
        if mae<best_mae: best_mae=mae; best_j=j; best_aug=aug; best_fill=m
    case_aug=best_aug if best_aug is not None else qx.copy(); case_fill=best_fill if best_fill is not None else np.zeros_like(target)

    entry_aug=qx.copy(); entry_fill=np.zeros_like(target)
    err=np.abs(vals-xf[None,:,:]); err[~avail]=np.inf; err[:,~target]=np.inf
    jbest=np.argmin(err,axis=0); ebest=np.min(err,axis=0); m=target & np.isfinite(ebest)
    ti,ni=np.where(m); entry_aug[ti,ni]=vals[jbest[ti,ni],ti,ni]; entry_fill[m]=True
    entry_mae=float(np.mean(np.abs(entry_aug[target]-xf[target])))

    return {
        'case_x_aug':case_aug.astype(np.float32),'case_fill':case_fill,'case_best_local_rank':None if best_j is None else int(best_j),
        'case_recon_mae_norm':best_mae,
        'entry_x_aug':entry_aug.astype(np.float32),'entry_fill':entry_fill,'entry_recon_mae_norm':entry_mae,
        'target_count':int(target.sum()),'case_fill_count':int(case_fill.sum()),'entry_fill_count':int(entry_fill.sum()),
    }


def retrieved_entry_oracles_by_k(item, completion_pool, ks):
    """Post-hoc entry-wise opportunity for ranked candidate prefixes.

    Hidden x_full is used only after the real prediction. Uncovered target entries
    retain the observed placeholder and remain in the reconstruction error.
    Returns augmented histories for each requested K so downstream forecast ceiling
    can be measured with the same online PatchSTG state.
    """
    qx=item['x_obs'].numpy().astype(np.float32)
    xf=item['x_full'].numpy().astype(np.float32)
    target=item['artificial_missing'].numpy().astype(bool) & np.isfinite(xf)
    vals=np.asarray(completion_pool.get('candidate_values',[]),np.float32)
    avail=np.asarray(completion_pool.get('candidate_available',[]),bool)
    out={}
    if vals.ndim!=3 or len(vals)==0 or not np.any(target):
        return out
    for kval in sorted({int(k) for k in ks if int(k)>0}):
        k=min(kval,len(vals))
        v=vals[:k]; a=avail[:k]
        aug=qx.copy(); err=np.abs(v-xf[None,:,:]); err[~a]=np.inf; err[:,~target]=np.inf
        jbest=np.argmin(err,axis=0); ebest=np.min(err,axis=0); m=target & np.isfinite(ebest)
        ti,ni=np.where(m); aug[ti,ni]=v[jbest[ti,ni],ti,ni]
        recon=float(np.mean(np.abs(aug[target]-xf[target])))
        out[kval]={'x_aug':aug.astype(np.float32),'fill_mask':m,'fill_count':int(m.sum()),'target_count':int(target.sum()),
                   'recon_mae_norm':recon,'effective_k':k}
    return out
