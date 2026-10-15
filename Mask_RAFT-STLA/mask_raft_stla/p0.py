"""P0 retrieval diagnostics for a *saved*, unchanged v2.3 checkpoint.

Offline paired metrics use exactly the same test query, future-valid mask and
prediction horizons. Candidate probe is a separately invoked inference-only
study and NEVER uses target futures to select retrieval evidence at inference.
Labels may only be used for post-hoc oracle/utility statistics.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

SCHEMA = 'p0-v1'


def read_predictions(checkpoint: Path, severity: int, mode: str, expected_hash: str | None = None):
    out = checkpoint.resolve().parent / 'evaluation' / f's{severity}' / mode
    pred_path = out / 'test_predictions.npz'
    if not pred_path.is_file():
        raise FileNotFoundError(f'Missing {mode} evaluation for s{severity}: {pred_path}')
    sm = out / 'summary.json'
    if not sm.is_file():
        raise FileNotFoundError(f'Evaluation has no checkpoint provenance: {sm}')
    meta = json.loads(sm.read_text(encoding='utf-8'))
    if expected_hash and meta.get('checkpoint_sha256') != expected_hash:
        raise RuntimeError(f'Checkpoint fingerprint mismatch for {pred_path}. Re-evaluate that checkpoint.')
    with np.load(pred_path, allow_pickle=False) as arr:
        return np.asarray(arr['prediction'], np.float32), np.asarray(arr['base_prediction'], np.float32)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for part in iter(lambda: fh.read(4 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def validate_pair(full_ck, independent_ck, method):
    """Independent no-retrieval comparison is impossible without a second checkpoint."""
    full = torch.load(full_ck, map_location='cpu', weights_only=False)
    if full.get('version') != '2.3.0' or full['config']['training'].get('ablation', 'full') != 'full':
        raise ValueError('Full checkpoint must be an independently trained v2.3 full checkpoint.')
    if independent_ck is None:
        return full, None
    ctrl = torch.load(independent_ck, map_location='cpu', weights_only=False)
    if ctrl.get('version') != '2.3.0' or ctrl['config']['training'].get('ablation') != 'no_retrieval':
        raise ValueError('Control checkpoint must be a separately trained no_retrieval v2.3 run.')
    if full.get('backbone_sha256') != ctrl.get('backbone_sha256'):
        raise ValueError('Full and independent controls must share the EXACT initial backbone SHA256.')
    for key in ('target', 'dataset', 'seed', 'history', 'horizon', 'stride', 'backbone_mode'):
        if full['config']['experiment'].get(key) != ctrl['config']['experiment'].get(key):
            raise ValueError(f'Independent control protocol mismatch on {key}')
    for field in ('batch_size', 'warmup_epochs', 'partial_epochs', 'max_epochs',
                  'patience', 'lr_backbone', 'lr_new', 'weight_decay', 'grad_clip'):
        if full['config']['training'].get(field) != ctrl['config']['training'].get(field):
            raise ValueError(f'Training budget/hyperparameter mismatch for {field}')
    return full, ctrl


def query_metrics(y, valid, full, base, cross, disabled, control=None, direct=None,
                  margin=1e-8):
    """Per-query paired summaries, shape all forecasts [Q,P,N]."""
    y = np.asarray(y, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    variants = {'full': full, 'base': base, 'cross': cross, 'disabled': disabled}
    if control is not None: variants['independent'] = control
    for key, val in variants.items():
        if np.shape(val) != y.shape:
            raise ValueError(f'Nonaligned {key} forecasts {np.shape(val)} != truth {y.shape}')
    if valid.shape != y.shape:raise ValueError('Future valid mask differs from label array')
    if direct is not None and np.shape(direct) != (y.shape[0], y.shape[2]):
        raise ValueError('Direct mask shape mismatch')
    # Common finite prediction support, preventing unfair denominators.
    common = valid & np.isfinite(y)
    for item in variants.values(): common &= np.isfinite(item)
    errors = {name:np.abs(np.asarray(pred, np.float32) - y) for name,pred in variants.items()}
    n = common.sum(axis=(1,2))
    def weighted(x, m=common):
        cnt = m.sum(axis=(1,2))
        return np.divide((np.where(m,x,0)).sum(axis=(1,2),dtype=np.float64), cnt,
                         out=np.full(y.shape[0], np.nan), where=cnt>0)
    result = {'valid_count':n}
    for k,v in errors.items():
        result[f'{k}_mae']=weighted(v)
        result[f'{k}_rmse']=np.sqrt(weighted(v*v))
        # Same raw-unit denominator rule as v2.3 evaluation MAPE: |y|>1.
        eligible=common & (np.abs(y)>1.0)
        result[f'{k}_mape_pct']=100.0*weighted(np.divide(v,np.maximum(np.abs(y),1.0)),eligible)
    delta = errors['base'] - errors['full']
    result.update(gain_mae=weighted(delta),
                  beneficial_element_rate=weighted((delta>margin).astype(np.float32)),
                  harmful_element_rate=weighted((delta<-margin).astype(np.float32)),
                  neutral_element_rate=weighted((np.abs(delta)<=margin).astype(np.float32)),
                  positive_gain_mean=weighted(np.where(delta>margin,delta,0)),
                  negative_gain_mean=weighted(np.where(delta<-margin,delta,0)),
                  corrected_element_rate=weighted((np.abs(full-base)>margin).astype(np.float32)))
    # The *conditioned* positive/negative magnitude is distinct from the
    # all-element positive/negative contribution returned above.
    p = common & (delta>margin); m = common & (delta<-margin)
    result['positive_gain_conditional'] = weighted(np.maximum(delta,0),p)
    result['negative_loss_conditional'] = weighted(np.maximum(-delta,0),m)
    result['cross_history_gain']=result['cross_mae']-result['full_mae']
    result['disabled_gain']=result['disabled_mae']-result['full_mae']
    if control is not None:
        result['independent_gain']=result['independent_mae']-result['full_mae']
    if direct is not None:
        md = common & np.asarray(direct, bool)[:,None,:]
        nd = common & ~np.asarray(direct,bool)[:,None,:]
        for label,mask in [('direct_masked',md),('not_direct_masked',nd)]:
            result[f'{label}_count']=mask.sum(axis=(1,2))
            result[f'{label}_full_mae']=weighted(errors['full'],mask)
            result[f'{label}_base_mae']=weighted(errors['base'],mask)
            result[f'{label}_gain']=weighted(delta,mask)
    return pd.DataFrame(result)


def support_stratum(coverage):
    if not np.isfinite(coverage):return 'unknown'
    if coverage < .10:return 'coverage_<10%'
    if coverage < .30:return 'coverage_10-30%'
    if coverage < .50:return 'coverage_30-50%'
    if coverage < .70:return 'coverage_50-70%'
    return 'coverage_>=70%'


def summary_table(frame):
    """Paired query-level summary (each query equal weight) with nested strata."""
    group_keys=[['severity'],['severity','family'],['severity','support_bin'],
                ['severity','family','support_bin'], ['severity','backbone_mode']]
    result=[]
    metric_cols=['full_mae','base_mae','cross_mae','disabled_mae','independent_mae',
                 'gain_mae','cross_history_gain','disabled_gain','independent_gain',
                 'full_rmse','base_rmse','cross_rmse','disabled_rmse','independent_rmse',
                 'full_mape_pct','base_mape_pct','cross_mape_pct','disabled_mape_pct','independent_mape_pct',
                 'beneficial_element_rate','harmful_element_rate','neutral_element_rate',
                 'positive_gain_conditional','negative_loss_conditional','direct_masked_gain',
                 'not_direct_masked_gain','coverage','zero_history_node_fraction']
    for keys in group_keys:
        if not set(keys).issubset(frame.columns):continue
        for grouping,part in frame.groupby(keys,dropna=False):
            if not isinstance(grouping,tuple):grouping=(grouping,)
            row={'grouping':'+'.join(keys),'queries':int(len(part))}
            row.update(zip(keys,grouping))
            for col in metric_cols:
                if col in part:
                    values=part[col].to_numpy(dtype=float)
                    row[col]=float(values[np.isfinite(values)].mean()) if np.isfinite(values).any() else float('nan')
            row['beneficial_query_rate']=float((part['gain_mae']>0).mean())
            row['harmful_query_rate']=float((part['gain_mae']<0).mean())
            result.append(row)
    return pd.DataFrame(result)


def candidate_probe_batch(output, batch, max_delta, margin=.02):
    """Compute hindsight candidate diagnostics WITHOUT using label during selection.

    The hindsight 'best of individual candidate or zero' number is an
    unattainable per-element supervised diagnostic. It is not an upper bound
    on mixtures of candidates nor is it a deployable retrieval score.
    """
    if output.get('candidate_residuals') is None:
        raise RuntimeError('P0 candidate probe requires a FULL retrieval forward output')
    with torch.no_grad():
        r = output['candidate_residuals'].float()
        base = output['base_prediction'].float()
        final = output['prediction'].float()
        y = batch['y_norm'].float()
        valid = batch['future_valid'].bool() & torch.isfinite(y)
        available = output['candidate_available'].bool() & valid[:,None]
        if r.ndim!=4 or base.ndim!=3 or y.shape!=base.shape:
            raise RuntimeError('Invalid candidate diagnostics tensor shape')
        if not bool(torch.isfinite(r).all().item()):
            raise RuntimeError('Non-finite candidate residual detected during P0 oracle diagnostics')
        n = valid.sum((1,2)).clamp_min(1)
        base_err=(base-y).abs(); full_err=(final-y).abs()
        max_delta = float(max_delta)
        if max_delta <=0:raise ValueError('max_delta must be >0')
        direct_candidate = max_delta*torch.tanh(r/max_delta)
        raw_gain = base_err[:,None]-(base[:,None]+direct_candidate-y[:,None]).abs()
        gain_valid = raw_gain.masked_fill(~available,-float('inf'))
        best_gain=gain_valid.max(1).values.clamp_min(0)
        best_gain=torch.where(valid,best_gain,torch.zeros_like(best_gain))
        any_good=(gain_valid>0).any(1) & valid
        oracle_loss=base_err-best_gain
        correction = final-base
        gate=output.get('risk_probability')
        ungated=output.get('ungated_correction')
        if gate is None or ungated is None:
            raise RuntimeError('Full model must expose risk gate/proposal for P0 calibration')
        # Calibration label mirrors training objective, in NORMALIZED units.
        proposal=base+ungated.float()
        true_benefit=(base_err-(proposal-y).abs())>float(margin)
        w = output['candidate_weights'].float()
        k = r.shape[1]
        if w.shape[1] != k+1:raise RuntimeError('Expected candidates plus None')
        winner=w[:,:k].argmax(1)
        selected_r=torch.gather(direct_candidate,1,winner[:,None]).squeeze(1)
        selected_available=torch.gather(available,1,winner[:,None]).squeeze(1)
        selected_gain=base_err-(base+selected_r-y).abs()
        none_w=w[:,-1]
        non_none_w=w[:,:k]
        selected_is_candidate=(non_none_w.max(1).values>none_w)&selected_available
        selected_gain=torch.where(selected_is_candidate,selected_gain,torch.zeros_like(selected_gain))
        weighted_gain=(non_none_w*torch.where(available,raw_gain,torch.zeros_like(raw_gain))).sum(1)
        risk_prob=gate.float().clamp(0,1)
        def mean_q(t,vm=valid):
            return ((t.float()*vm).sum((1,2))/vm.sum((1,2)).clamp_min(1)).cpu().numpy()
        row={
            'n_valid':valid.sum((1,2)).cpu().numpy(),
            'base_mae_norm':mean_q(base_err),
            'full_mae_norm':mean_q(full_err),
            'best_single_candidate_or_none_mae_norm':mean_q(oracle_loss),
            'best_single_candidate_gain_norm':mean_q(best_gain),
            'oracle_minus_full_gain_norm':mean_q(full_err-oracle_loss),
            'actual_full_gain_norm':mean_q(base_err-full_err),
            'weighted_single_candidate_gain_norm':mean_q(weighted_gain),
            'selected_candidate_gain_norm':mean_q(selected_gain),
            'positive_candidate_available_rate':mean_q(any_good),
            'candidate_available_rate':mean_q(available.any(1)),
            'risk_prob_mean':mean_q(risk_prob),
            'risk_label_rate':mean_q(true_benefit),
            'risk_brier':mean_q((risk_prob-true_benefit.float()).square()),
            'gate_mean':mean_q(output['gate']),
            'none_weight_mean':mean_q(none_w),
            'correction_abs_norm':mean_q(correction.abs()),
            'full_beneficial_element_rate':mean_q((base_err>full_err)),
            'full_harmful_element_rate':mean_q((base_err<full_err)),
            'selected_candidate_useful_rate':mean_q((selected_gain>0)&selected_is_candidate),
        }
        if output.get('router') is not None:
            router=output['router'].float()
            if router.shape[-1]==4:
                for j,name in enumerate(('temporal','spatial','regional','none')):
                    row[f'router_{name}_mean']=mean_q(router[...,j])
        # Clip risk by prediction-valid; produce per-bin calibration rows on CPU.
        edges=np.linspace(0.,1.,11)
        pro=risk_prob[valid].cpu().numpy()
        target=true_benefit[valid].float().cpu().numpy()
        ids=np.minimum(np.searchsorted(edges,pro,side='right')-1,9).clip(0,9)
        counts=np.bincount(ids,minlength=10)
        probs=np.bincount(ids,weights=pro,minlength=10)
        labels=np.bincount(ids,weights=target,minlength=10)
        bins=[dict(bin=i,low=float(edges[i]),high=float(edges[i+1]),count=int(counts[i]),
                   sum_probability=float(probs[i]),sum_positive=float(labels[i])) for i in range(10)]
        return row,bins


def aggregate_calibration(rows):
    if not rows:return pd.DataFrame()
    f=pd.DataFrame(rows)
    keys=['severity','bin']
    tab=f.groupby(keys,as_index=False)[['count','sum_probability','sum_positive']].sum()
    tab['confidence']=np.divide(tab.sum_probability,tab['count'],out=np.full(len(tab),np.nan),where=tab['count']>0)
    tab['empirical_positive']=np.divide(tab.sum_positive,tab['count'],out=np.full(len(tab),np.nan),where=tab['count']>0)
    tab['absolute_calibration_error']=(tab.confidence-tab.empirical_positive).abs()
    return tab
