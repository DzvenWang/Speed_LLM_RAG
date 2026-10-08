from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd, torch
from .data import loader,to_device
from .config import insert_paths,json_write

def _metrics(pred,true,valid,direct):
    insert_paths()
    from experiment_common.metrics import backbone_metric_summary
    return backbone_metric_summary(pred,true,valid,direct)

@torch.no_grad()
def evaluate(model,memory,dataset,device,batch_size,mean,std,run_dir=None,tag='val',save_arrays=False,workers=0,mode='full'):
    model.eval()
    if mode!='no_memory':model.refresh(memory)
    p=[];b=[];c=[];y=[];v=[];d=[];rows=[];all_gates=[];all_states=[];all_art=[];all_full=[];retrieval_ids=[];retrieval_weights=[];retrieval_valid=[];diag=[];effective_gates=[]
    for batch in loader(dataset,batch_size,workers=workers):
        bt=to_device(batch,device)
        out=model(bt,memory,mode=mode)
        # Base_SupportBench's canonical evaluator masks nonfinite predictions.
        # Reject a broken model before it can silently reduce metric counts.
        for name in ('prediction','base_prediction','candidate_prediction'):
            if not torch.isfinite(out[name]).all():
                raise FloatingPointError(f'FARE_NONFINITE_EVALUATE tag={tag} mode={mode} key={name}')
        p.append((out['prediction']*std+mean).cpu().numpy());b.append((out['base_prediction']*std+mean).cpu().numpy());c.append((out['candidate_prediction']*std+mean).cpu().numpy())
        y.append(batch['y_raw'].numpy());v.append(batch['future_valid'].numpy());d.append(batch['direct_mask'].numpy())
        all_gates.append(out['final_gate'].mean().item() if 'final_gate' in out else 0.0)
        if 'final_gate' in out:effective_gates.append(out['final_gate'].detach().cpu().numpy())
        if 'retrieved_ids' in out:
            retrieval_ids.append(out['retrieved_ids'].cpu().numpy())
            retrieval_weights.append(out['retrieved_weights'].cpu().numpy())
            retrieval_valid.append(out['retrieval_valid'].cpu().numpy())
            diag.append({'state_gate':float(out['state_gate'].mean()),'innovation_abs':float(out['latent_innovation'].abs().mean()),'retrieved_token_abs':float(out['retrieved_token_abs'].mean()),'confidence':float(out['retrieval_confidence'].mean()),'predicted_utility':float(out['predicted_utility'].mean()),'predicted_risk':float(out['predicted_risk'].mean())})
        if 'recovered_state' in out:
            rec=out['recovered_state'].detach().cpu().numpy();all_states.append(rec)
            all_art.append(batch['artificial_mask'].numpy());all_full.append(batch['x_full'].numpy())
        rows.extend(batch['row_id'].tolist())
    if not p:raise RuntimeError('Empty split')
    p,b,c,y,v,d=[np.concatenate(a) for a in (p,b,c,y,v,d)]
    result={'final':_metrics(p,y,v,d),'base':_metrics(b,y,v,d),'candidate':_metrics(c,y,v,d),'gate_mean':float(np.mean(all_gates)),'rows':len(p),'tag':tag}
    if effective_gates:
        g=np.concatenate(effective_gates)
        result['gate_statistics']={'mean':float(g.mean()),'p10':float(np.percentile(g,10)),
                                   'p50':float(np.percentile(g,50)),'p90':float(np.percentile(g,90)),
                                   'max':float(g.max()),'nonzero_fraction':float(np.mean(g>0))}
    if diag:
        w=np.concatenate(retrieval_weights);gvalid=np.concatenate(retrieval_valid)
        result['retrieval_diagnostics']={'valid_group_fraction':float(gvalid.mean()),'effective_k':float(np.exp(-(w*np.log(np.maximum(w,1e-9))).sum(-1)).mean()),**{k:float(np.mean([d[k] for d in diag])) for k in diag[0]}}
    if all_states:
        restored=np.concatenate(all_states);full=np.concatenate(all_full);am=np.concatenate(all_art).astype(bool)
        if am.any():
            dif=restored[am]-full[am];result['recovery']={'MAE_normalized':float(np.mean(np.abs(dif))),'RMSE_normalized':float(np.sqrt(np.mean(dif*dif))),'count':int(am.sum())}
    if dataset.family_column:
        family=dataset.meta.iloc[rows][dataset.family_column].astype(str).to_numpy()
        result['families']={f:_metrics(p[family==f],y[family==f],v[family==f],d[family==f]) for f in sorted(set(family))}
    diff=np.abs(c-y)-np.abs(b-y);mask=v.astype(bool)
    final_diff=np.abs(p-y)-np.abs(b-y)
    result['corrections']={'helpful_fraction':float(np.mean((diff[mask]<0))),
                           'harmful_fraction':float(np.mean((diff[mask]>0))),
                           'candidate_MAE_gain':float(np.mean(-diff[mask])),
                           'final_helpful_fraction':float(np.mean(final_diff[mask]<0)),
                           'final_harmful_fraction':float(np.mean(final_diff[mask]>0)),
                           'final_MAE_gain':float(np.mean(-final_diff[mask])),
                           'candidate_delta_abs':float(np.abs(c-b)[mask].mean())}
    if run_dir:
        path=Path(run_dir);path.mkdir(parents=True,exist_ok=True)
        json_write(path/f'{tag}_summary.json',result)
        if save_arrays:
            kw=dict(prediction=p.astype(np.float32),base=b.astype(np.float32),candidate=c.astype(np.float32),
                    target=y.astype(np.float32),valid=v.astype(bool),direct=d.astype(bool),
                    view_id=np.asarray(dataset.meta.iloc[rows].view_id.astype(str)) if 'view_id' in dataset.meta else np.asarray(rows).astype(str))
            if effective_gates:kw['effective_gate']=np.concatenate(effective_gates).astype(np.float32)
            if getattr(dataset,'family_column',None):kw['family']=dataset.meta.iloc[rows][dataset.family_column].astype(str).to_numpy()
            np.savez_compressed(path/f'{tag}_predictions.npz',**kw)
        if retrieval_ids:
            np.savez_compressed(path/f'{tag}_retrieval.npz',candidate_ids=np.concatenate(retrieval_ids),candidate_weights=np.concatenate(retrieval_weights),valid_groups=np.concatenate(retrieval_valid))
        if 'families' in result:pd.DataFrame([{'family':key,**val} for key,val in result['families'].items()]).to_csv(path/f'{tag}_family_metrics.csv',index=False)
    return result
