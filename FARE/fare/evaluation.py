from __future__ import annotations
"""FARE evaluation using the single Base_SupportBench metric implementation.

No local MAE/RMSE/MAPE formula is maintained here. All headline and family
scores use shared micro-aggregation and exact shared masks; H3/H6/H12 refer to
single forecast steps, not cumulative horizons.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import insert_paths, json_write
from .data import loader, to_device


@lru_cache(maxsize=1)
def common_metrics():
    """Fail early if Base_SupportBench's MAPE extension has not been deployed."""
    insert_paths()
    try:
        from experiment_common.metrics import (
            metric_bundle, per_query_metrics, MAPE_PROTOCOL,
            MAPE_MIN_TARGET_ABS,
        )
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            'FARE requires the 2026-10-12 Base_SupportBench MAPE update. '
            'Merge Traffic_ThreeProjects_MAPE_20261012(1).zip at repository root '
            '(Base_SupportBench, Backbone, Baseline) before running FARE.'
        ) from exc
    expected = 'raw_target_percent_exclude_abs_le_1_v1'
    if MAPE_PROTOCOL != expected or float(MAPE_MIN_TARGET_ABS) != 1.0:
        raise RuntimeError(
            'Unexpected Base_SupportBench MAPE protocol: '
            f'{MAPE_PROTOCOL} / threshold {MAPE_MIN_TARGET_ABS}. '
            'Do not mix differently defined MAE/RMSE/MAPE evaluations.'
        )
    return metric_bundle, per_query_metrics, MAPE_PROTOCOL, MAPE_MIN_TARGET_ABS


def _metrics(pred, true, valid, direct):
    """Call canonical shared metric_bundle; add familiar FARE/Backbone aliases.

    Returned keys keep ALL canonical scopes, horizons, counts and protocol
    metadata, plus top-level MAE/RMSE/MAPE for existing report consumers.
    """
    bundle, _, _, _ = common_metrics()
    pred = np.asarray(pred)
    true = np.asarray(true)
    valid = np.asarray(valid, dtype=bool)
    direct = np.asarray(direct, dtype=bool)
    if pred.shape != true.shape or valid.shape != true.shape:
        raise ValueError(f'Evaluation shape mismatch: pred={pred.shape}, target={true.shape}, valid={valid.shape}')
    if pred.ndim != 3 or direct.shape != (pred.shape[0], pred.shape[2]):
        raise ValueError(f'Evaluation direct_mask shape mismatch: direct={direct.shape}, pred={pred.shape}')
    result = dict(bundle(pred, true, valid, direct))
    required = (
        'metrics_revision', 'mape_protocol', 'mape_min_target_abs',
        'overall_MAE', 'overall_RMSE', 'overall_MAPE',
        'masked_MAE', 'masked_RMSE', 'masked_MAPE',
        'overall_MAPE_count', 'overall_MAPE_coverage',
        'masked_MAPE_count', 'masked_MAPE_coverage',
        'H3_overall_MAPE', 'H6_overall_MAPE', 'H12_overall_MAPE',
    )
    missing = [k for k in required if k not in result]
    if missing:
        raise RuntimeError(f'Base_SupportBench metrics bundle lacks MAPE protocol fields: {missing}')
    for metric in ('MAE', 'RMSE', 'MAPE', 'count', 'MAPE_count', 'MAPE_coverage'):
        result[metric] = result[f'overall_{metric}']
        result[f'directly_masked_{metric}'] = result[f'masked_{metric}']
    for horizon in (3, 6, 12):
        for metric in ('MAE', 'RMSE', 'MAPE', 'count', 'MAPE_count', 'MAPE_coverage'):
            result[f'H{horizon}_{metric}'] = result[f'H{horizon}_overall_{metric}']
            result[f'H{horizon}_directly_masked_{metric}'] = result[f'H{horizon}_masked_{metric}']
    return result


@torch.no_grad()
def evaluate(model, memory, dataset, device, batch_size, mean, std, run_dir=None,
             tag='val', save_arrays=False, workers=0, mode='full', refresh_memory=True):
    # Check shared metrics BEFORE GPU evaluation (fail-fast if only FARE was deployed).
    _, per_query_metrics, _, _ = common_metrics()
    model.eval()
    if mode != 'no_memory' and refresh_memory:
        model.refresh(memory)
    p=[]; b=[]; c=[]; y=[]; v=[]; d=[]; rows=[]
    all_gates=[]; all_states=[]; all_art=[]; all_full=[]
    retrieval_ids=[]; retrieval_weights=[]; retrieval_valid=[]; diag=[]; effective_gates=[]
    for batch in loader(dataset, batch_size, workers=workers):
        bt=to_device(batch, device)
        out=model(bt, memory, mode=mode)
        for name in ('prediction', 'base_prediction', 'candidate_prediction'):
            if not torch.isfinite(out[name]).all():
                raise FloatingPointError(f'FARE_NONFINITE_EVALUATE tag={tag} mode={mode} key={name}')
        p.append((out['prediction']*std+mean).cpu().numpy())
        b.append((out['base_prediction']*std+mean).cpu().numpy())
        c.append((out['candidate_prediction']*std+mean).cpu().numpy())
        y.append(batch['y_raw'].numpy())
        v.append(batch['future_valid'].numpy())
        d.append(batch['direct_mask'].numpy())
        all_gates.append(out['final_gate'].mean().item() if 'final_gate' in out else 0.0)
        if 'final_gate' in out:
            effective_gates.append(out['final_gate'].detach().cpu().numpy())
        if 'retrieved_ids' in out:
            retrieval_ids.append(out['retrieved_ids'].cpu().numpy())
            retrieval_weights.append(out['retrieved_weights'].cpu().numpy())
            retrieval_valid.append(out['retrieval_valid'].cpu().numpy())
            diag.append({
                'state_gate':float(out['state_gate'].mean()),
                'innovation_abs':float(out['latent_innovation'].abs().mean()),
                'retrieved_token_abs':float(out['retrieved_token_abs'].mean()),
                'confidence':float(out['retrieval_confidence'].mean()),
                'predicted_utility':float(out['predicted_utility'].mean()),
                'predicted_risk':float(out['predicted_risk'].mean()),
            })
        if 'recovered_state' in out:
            all_states.append(out['recovered_state'].detach().cpu().numpy())
            all_art.append(batch['artificial_mask'].numpy())
            all_full.append(batch['x_full'].numpy())
        rows.extend(batch['row_id'].tolist())
    if not p:
        raise RuntimeError('Empty evaluation split')
    p,b,c,y,v,d=[np.concatenate(a) for a in (p,b,c,y,v,d)]
    result={
        'final':_metrics(p,y,v,d),
        'base':_metrics(b,y,v,d),
        'candidate':_metrics(c,y,v,d),
        'gate_mean':float(np.mean(all_gates)),
        'rows':len(p),'tag':tag,
        'metrics_source':'Base_SupportBench.experiment_common.metrics.metric_bundle',
    }
    if effective_gates:
        g=np.concatenate(effective_gates)
        result['gate_statistics']={
            'mean':float(g.mean()),'p10':float(np.percentile(g,10)),
            'p50':float(np.percentile(g,50)),'p90':float(np.percentile(g,90)),
            'max':float(g.max()),'nonzero_fraction':float(np.mean(g>0)),
        }
    if diag:
        w=np.concatenate(retrieval_weights)
        gvalid=np.concatenate(retrieval_valid)
        result['retrieval_diagnostics']={
            'valid_group_fraction':float(gvalid.mean()),
            'effective_k':float(np.exp(-(w*np.log(np.maximum(w,1e-9))).sum(-1)).mean()),
            **{k:float(np.mean([s[k] for s in diag])) for k in diag[0]},
        }
    if all_states:
        restored=np.concatenate(all_states)
        full=np.concatenate(all_full)
        am=np.concatenate(all_art).astype(bool)
        if am.any():
            dif=restored[am]-full[am]
            result['recovery']={
                'MAE_normalized':float(np.mean(np.abs(dif))),
                'RMSE_normalized':float(np.sqrt(np.mean(dif*dif))),
                'count':int(am.sum()),
            }
    frame = dataset.meta.iloc[rows]
    if getattr(dataset,'family_column',None):
        family=frame[dataset.family_column].astype(str).to_numpy()
        result['families']={
            f:_metrics(p[family==f],y[family==f],v[family==f],d[family==f])
            for f in sorted(set(family))
        }
    # The shared metric mask also excludes nonfinite labels (and predictions).
    mask=v.astype(bool)&np.isfinite(y)&np.isfinite(p)&np.isfinite(b)&np.isfinite(c)
    if mask.any():
        diff=np.abs(c-y)-np.abs(b-y)
        final_diff=np.abs(p-y)-np.abs(b-y)
        result['corrections']={
            'helpful_fraction':float(np.mean(diff[mask]<0)),
            'harmful_fraction':float(np.mean(diff[mask]>0)),
            'candidate_MAE_gain':float(np.mean(-diff[mask])),
            'final_helpful_fraction':float(np.mean(final_diff[mask]<0)),
            'final_harmful_fraction':float(np.mean(final_diff[mask]>0)),
            'final_MAE_gain':float(np.mean(-final_diff[mask])),
            'candidate_delta_abs':float(np.abs(c-b)[mask].mean()),
        }
    if run_dir:
        path=Path(run_dir);path.mkdir(parents=True,exist_ok=True)
        json_write(path/f'{tag}_summary.json',result)
        if save_arrays:
            kw={
                'prediction':p.astype(np.float32),
                'base':b.astype(np.float32),
                'candidate':c.astype(np.float32),
                'target':y.astype(np.float32),
                'valid':v.astype(bool),
                'direct':d.astype(bool),
                'view_id':frame.view_id.astype(str).to_numpy() if 'view_id' in frame else np.asarray(rows).astype(str),
            }
            if effective_gates:
                kw['effective_gate']=np.concatenate(effective_gates).astype(np.float32)
            if getattr(dataset,'family_column',None):
                kw['family']=frame[dataset.family_column].astype(str).to_numpy()
            np.savez_compressed(path/f'{tag}_predictions.npz',**kw)
            # Public per_query_metrics is for diagnostics only; headline MAPE
            # must be micro-aggregated via metric_bundle, NEVER averaged here.
            query_stats=per_query_metrics(p,y,v,d)
            for i,stat in enumerate(query_stats):
                stat['row_id']=int(rows[i])
                if 'view_id' in frame:
                    stat['view_id']=str(frame.iloc[i]['view_id'])
                if getattr(dataset,'family_column',None):
                    stat['family']=str(frame.iloc[i][dataset.family_column])
            pd.DataFrame(query_stats).to_csv(path/f'{tag}_per_query_metrics.csv',index=False)
        if retrieval_ids:
            np.savez_compressed(
                path/f'{tag}_retrieval.npz',
                candidate_ids=np.concatenate(retrieval_ids),
                candidate_weights=np.concatenate(retrieval_weights),
                valid_groups=np.concatenate(retrieval_valid),
            )
        if 'families' in result:
            pd.DataFrame([{'family':key,**val} for key,val in result['families'].items()]).to_csv(
                path/f'{tag}_family_metrics.csv',index=False
            )
    return result
