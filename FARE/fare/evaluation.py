from __future__ import annotations

from collections import Counter,defaultdict
from datetime import datetime
import numpy as np
import pandas as pd

from .cache import CacheSplit
from .inference import ExpertRuntime
from .metrics import error_stats,finalize,restoration_mae_raw,add_stats
from .utils import balanced_indices,safe_corr,severity_tag,write_json,log


def _gain(base,method): return float((base-method)/max(abs(base),1e-12)*100.)
def _qmae(p,y,m):
    mm=np.asarray(m,bool); return float(np.abs(np.asarray(p)-np.asarray(y))[mm].mean()) if np.any(mm) else float('nan')


def evaluate_experts(ctx,severity:int,max_rows:int|None=None,smoke=False):
    d=CacheSplit(ctx.evaluation_cache(severity),'test'); idx=balanced_indices(d.meta,max_rows,ctx.cfg['experiment']['families'],ctx.seed+int(severity)); meta=d.meta.iloc[idx].reset_index(drop=True)
    x=np.asarray(d.x_obs[idx],np.float32).copy(); obs=np.asarray(d.history_observed[idx],bool).copy(); art=np.asarray(d.artificial_missing[idx],bool).copy(); full=np.asarray(d.x_full[idx],np.float32).copy(); y=np.asarray(d.y_raw[idx],np.float32).copy(); valid=np.asarray(d.future_valid[idx],bool).copy(); direct=np.asarray(d.direct_mask[idx],bool).copy(); tod=np.asarray(d.tod_hist[idx],np.int64).copy(); dow=np.asarray(d.dow_hist[idx],np.int64).copy(); starts=np.asarray(meta.window_start,np.int64)
    rt=ExpertRuntime(ctx); restored,raw,diag=rt.restore_all(x,obs,art,tod,dow,starts,smoke=smoke)
    for name,z in restored.items():
        if not np.array_equal(z[~art],x[~art]): raise RuntimeError(f'{name} modified non-artificial cells')
        if not np.array_equal(raw[name][~art],x[~art]): raise RuntimeError(f'{name} raw restoration modified non-artificial cells')
    base=rt.bb.predict_raw(x,tod,dow); preds={k:rt.bb.predict_raw(v,tod,dow) for k,v in restored.items()}; methods={'backbone':base,**preds}
    overall={}; rec_mae={k:np.asarray([restoration_mae_raw(raw[k][q],full[q],art[q],ctx.std) for q in range(len(idx))]) for k in raw}; dm=valid & direct[:,None,:]
    for name,p in methods.items(): overall[name]={**finalize(error_stats(p,y,valid)),"directly_masked":finalize(error_stats(p,y,dm))}
    bmae=overall['backbone']['MAE']; bdm=overall['backbone']['directly_masked']['MAE']
    for name in restored:
        overall[name]['gain_percent']=_gain(bmae,overall[name]['MAE']); overall[name]['directly_masked_gain_percent']=_gain(bdm,overall[name]['directly_masked']['MAE']); overall[name]['raw_restoration_MAE']=float(np.nanmean(rec_mae[name]))
    qrows=[]; expert_routes=Counter(); safe_routes=Counter(); fam_rows=defaultdict(list); expert_oracle_stats={'sum_abs':0.0,'sum_sq':0.0,'count':0}; safe_oracle_stats={'sum_abs':0.0,'sum_sq':0.0,'count':0}
    for q in range(len(idx)):
        qb=_qmae(base[q],y[q],valid[q]); rec={'cache_row':int(idx[q]),'view_id':str(meta.iloc[q].view_id),'family':str(meta.iloc[q].family),'severity':float(meta.iloc[q].severity),'backbone_MAE':qb};
        for en in ('context','retrieval','backbone_aware'):
            if en in diag and isinstance(diag[en],list) and q < len(diag[en]):
                rec[f'{en}_selected_alpha']=float(diag[en][q].get('selected_alpha',0.0)); rec[f'{en}_mean_gate']=float(diag[en][q].get('mean_gate',0.0)); rec[f'{en}_mean_reliability']=float(diag[en][q].get('mean_reliability',0.0))
        best=('none',float('inf')); safe=('backbone',qb)
        for name,p in preds.items():
            qm=_qmae(p[q],y[q],valid[q]); rec[f'{name}_MAE']=qm; rec[f'{name}_gain_percent']=_gain(qb,qm); rec[f'{name}_raw_restoration_MAE']=float(rec_mae[name][q])
            if qm<best[1]: best=(name,qm)
            if qm<safe[1]: safe=(name,qm)
        rec['expert_oracle']=best[0]; rec['expert_oracle_MAE']=best[1]; rec['expert_oracle_gain_percent']=_gain(qb,best[1]); rec['safe_oracle']=safe[0]; rec['safe_oracle_MAE']=safe[1]; rec['safe_oracle_gain_percent']=_gain(qb,safe[1]); expert_routes[best[0]]+=1; safe_routes[safe[0]]+=1
        expert_oracle_stats=add_stats(expert_oracle_stats,error_stats(preds[best[0]][q],y[q],valid[q])); sp=base[q] if safe[0]=='backbone' else preds[safe[0]][q]; safe_oracle_stats=add_stats(safe_oracle_stats,error_stats(sp,y[q],valid[q])); qrows.append(rec); fam_rows[rec['family']].append(rec)
    qdf=pd.DataFrame(qrows); best_single=max(restored.keys(),key=lambda n:overall[n]['gain_percent']); best_single_gain=overall[best_single]['gain_percent']; expert_oracle_metric=finalize(expert_oracle_stats); safe_oracle_metric=finalize(safe_oracle_stats); expert_oracle_gain=_gain(bmae,expert_oracle_metric['MAE']); safe_oracle_gain=_gain(bmae,safe_oracle_metric['MAE'])
    by_family={}
    for fam,rows in fam_rows.items():
        f=pd.DataFrame(rows); cell={'queries':len(f),'expert_oracle_routes':dict(Counter(f.expert_oracle)),'safe_oracle_routes':dict(Counter(f.safe_oracle))}
        for name in ['backbone',*restored.keys()]:
            col='backbone_MAE' if name=='backbone' else f'{name}_MAE'; mae=float(f[col].mean()); cell[name]={'query_mean_MAE':mae,'gain_percent':0.0 if name=='backbone' else _gain(float(f.backbone_MAE.mean()),mae)}
        by_family[fam]=cell
    pairs=[(rec_mae[n][q],qdf.iloc[q][f'{n}_gain_percent']) for n in restored for q in range(len(idx))]; corr=safe_corr([a for a,b in pairs],[b for a,b in pairs])
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S'); out=ctx.exp_root/'four_expert_diagnostic'/severity_tag(severity)/f'{stamp}_seed{ctx.seed}'; out.mkdir(parents=True,exist_ok=True); qdf.to_csv(out/'query_results.csv',index=False)
    summary={'version':'0.2.4','purpose':'forecast_intervention_landscape_plus_hierarchical_prediction_control_diagnostic','target':ctx.target,'dataset':ctx.dataset,'severity':int(severity),'queries':int(len(idx)),'checkpoint':str(ctx.checkpoint),'checkpoint_sha256':ctx.checkpoint_sha256,'overall':overall,'best_single_expert':best_single,'best_single_gain_percent':float(best_single_gain),'expert_oracle':expert_oracle_metric,'safe_oracle':safe_oracle_metric,'expert_oracle_gain_percent':expert_oracle_gain,'safe_oracle_gain_percent':safe_oracle_gain,'expert_oracle_over_best_single_pp':float(expert_oracle_gain-best_single_gain),'safe_oracle_over_best_single_pp':float(safe_oracle_gain-best_single_gain),'expert_oracle_routes':dict(expert_routes),'safe_oracle_routes':dict(safe_routes),'by_family':by_family,'restoration_mae_vs_forecast_gain_pearson':corr,'expert_contract':'Every expert produces dense raw restoration for every artificial-missing cell; neural experts predict a query-level intervention utility landscape and allocate the selected alpha through cell-level reliability','leakage_contract':'x_full/y are formal train/val labels or evaluation diagnostics only; evaluation decisions use inference-visible data and train-only support','diagnostics':diag}
    write_json(out/'summary.json',summary); log('CONTROLLED_EXPERT_SUMMARY '+str({k:summary[k] for k in ['best_single_expert','best_single_gain_percent','expert_oracle_gain_percent','safe_oracle_gain_percent','safe_oracle_over_best_single_pp','safe_oracle_routes']})); log(f'CONTROLLED_EXPERT_PASS {out}'); return summary,out
