"""FARE-only post-training fixed-checkpoint attribution and traffic diagnostics.

NOT equivalent to removing a module and retraining. All contrasts use one model.
"""
from pathlib import Path
import csv,json
import numpy as np, torch
from .evaluation import evaluate,evaluate_modes
from .protocol import make_loader,device_batch
from .config import dump_json

MODES=['full','no_evidence','no_support_demand','no_direction','no_propagation_time',
       'no_boundary_support','no_regime','no_propagation_alignment',
       'no_capability_matching','state_similarity_only',
       'no_residual_information','no_evolution_information',
       'no_residual_value','no_evolution_value',
       'no_set_interaction','no_swap_utility','no_counterfactual_utility','no_null',
       'no_risk_control','no_hidden_injection','no_forecast_correction',
       'random_legal_memory','shuffled_legal_memory']

def write_csv(path,rows):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def valid_metrics(r):
    return float(r.get('overall_MAE',r.get('MAE',np.nan))),float(r.get('overall_RMSE',r.get('RMSE',np.nan)))

def paired_interval(ref,alt,truth,valid,origin,replicates=300,seed=10):
    """Block bootstrap over forecasting-origin days, not correlated individual elements."""
    mask=valid.astype(bool)&np.isfinite(ref)&np.isfinite(alt)&np.isfinite(truth)
    ref_error=np.abs(ref-truth);alt_error=np.abs(alt-truth)
    ref_sq=(ref-truth)**2;alt_sq=(alt-truth)**2
    days=np.asarray(origin)//288
    u,inv=np.unique(days,return_inverse=True)
    block=np.zeros((len(u),5),dtype=np.float64)
    for j in range(len(u)):
        rows=inv==j;vv=mask[rows]
        block[j]=[(alt_error[rows]*vv).sum(),(ref_error[rows]*vv).sum(),
                  (alt_sq[rows]*vv).sum(),(ref_sq[rows]*vv).sum(),vv.sum()]
    if not len(u) or not block[:,4].sum():return {}
    rng=np.random.default_rng(seed);mae=[];rmse=[]
    for _ in range(replicates):
        sample=block[rng.integers(0,len(u),size=len(u))].sum(0)
        den=max(sample[-1],1)
        mae.append((sample[0]-sample[1])/den)
        rmse.append(np.sqrt(sample[2]/den)-np.sqrt(sample[3]/den))
    return {'block_count':int(len(u)),'mae_ci_low':float(np.quantile(mae,.025)),
            'mae_ci_high':float(np.quantile(mae,.975)),
            'rmse_ci_low':float(np.quantile(rmse,.025)),
            'rmse_ci_high':float(np.quantile(rmse,.975))}

def _gain_by_coverage(full,alt):
    v=full['valid'].astype(bool)
    q=full['forward_coverage'].mean((1,2))
    boundary=np.quantile(q,[.25,.5,.75])
    result=[]
    for label,mask in [('low',q<=boundary[0]),('middle',(q>boundary[0])&(q<=boundary[2])),('high',q>boundary[2])]:
        valid=v[mask]
        if valid.sum()==0:continue
        delta=np.abs(alt['prediction'][mask]-alt['target'][mask])-np.abs(full['prediction'][mask]-full['target'][mask])
        result.append({'coverage_group':label,'queries':int(mask.sum()),'module_contribution_mae':float(delta[valid].mean())})
    return result

@torch.no_grad()
def counterfactual_audit(model,memory,ds,device,cfg,path,split=None):
    """Exact shared-query snapshot LOO/Swap + unscaled candidate risk actions.

    All candidate identities are fixed at the Full run. Illegal/repeated swaps
    are explicitly skipped. No retrieval or frozen prefix recomputation per slot.
    """
    model.eval();rows=[];budget=int(cfg['analysis']['max_counterfactual_queries'])
    count=0
    for raw in make_loader(ds,1,shuffle=False,workers=0):
        if count>=budget:break
        batch=device_batch(raw,device)
        full=model(batch,memory,purge=cfg['memory']['purge_steps'])
        y=batch['target'];valid=batch['future_valid'].bool()
        mae=lambda x:float(((x-y).abs()*valid).sum()/valid.sum().clamp_min(1))
        full_mae=mae(full['prediction']);base=mae(full['null_prediction'])
        meta={'split':split,'row_id':int(batch['row_id'][0]),'origin':int(batch['origin'][0]),
              'full_mae_norm':full_mae,'null_mae_norm':base}
        selected=full['selection'];short=full['candidate_pool'];prop=full['proposals']
        prepared=full['prepared'];k=selected['selected_ids'].shape[-1]
        for slot in range(k):
            edit=model.edit_selection(selected,prop,short,drop_slot=slot)
            pred=model.predict_with_selection(batch,prepared,edit)['prediction']
            rows.append({**meta,'operation':'loo','slot':slot,'candidate_position':None,
                         'gain_normalized':mae(pred)-full_mae})
        pool=short['ids'].shape[-1]
        for pos in range(pool):
            # Only meaningful legal replacement of the selected first slot.
            legal=short['valid'][:,:,pos] & (~(selected['selected_ids']==short['ids'][:,:,pos:pos+1]).any(-1))
            if not legal.any():continue
            edit=model.edit_selection(selected,prop,short,replace_slot=0,replace_position=pos)
            pred=model.predict_with_selection(batch,prepared,edit)['prediction']
            rows.append({**meta,'operation':'replacement','slot':0,'candidate_position':pos,
                         'eligible_nodes':int(legal.sum()),
                         'gain_normalized':full_mae-mae(pred)})
        delta=full['candidate_prediction']-full['null_prediction']
        for alpha in (0.,.25,.5,.75,1.,1.25):
            prediction=full['null_prediction']+alpha*delta
            rows.append({**meta,'operation':'scaling','slot':None,'candidate_position':None,
                         'alpha':alpha,'gain_normalized':full_mae-mae(prediction)})
        count+=1
    write_csv(Path(path)/'candidate_counterfactual.csv',rows)
    return rows

@torch.no_grad()
def analyze(model,memory,datasets,device,cfg,out_dir):
    root=Path(out_dir);root.mkdir(parents=True,exist_ok=True)
    summaries=[];traffic=[];interactions=[];risk_rows=[];evidence_rows=[];horizon_rows=[]
    # Post-hoc audit uses an explicitly selected evaluated split; test targets are
    # diagnostic only and must never be used to adjust learned parameters.
    requested_audit=cfg.get('analysis',{}).get('counterfactual_split','s40')
    audit_split=requested_audit if requested_audit in datasets else next(iter(datasets))
    print('FARE_COUNTERFACTUAL_AUDIT_SPLIT '+audit_split,flush=True)
    for tag,ds in datasets.items():
        base_path=root/tag
        # Pilot attribution is budgeted and explicitly labeled. Full predictions
        # are already evaluated on the complete Test dataset by pipeline.py.
        budget=cfg.get('analysis',{}).get('max_intervention_queries')
        if budget and int(budget)<len(ds):
            from torch.utils.data import Subset
            ii=np.linspace(0,len(ds)-1,int(budget),dtype=int).tolist()
            subset=Subset(ds,ii)
        else:subset=ds
        mode_results,mode_data=evaluate_modes(
            model,memory,subset,device,cfg,base_path,MODES,
            save_traces=bool(cfg.get('analysis',{}).get('save_intervention_traces',False)))
        summaries.append({'split':tag,'intervention':'scope_metadata','analyzed_queries':len(subset),
                          'total_test_queries':len(ds),'sampling':'evenly_spaced'})
        ref=mode_data['full'];ref_mae,ref_rmse=valid_metrics(mode_results['full'])
        for mode in MODES:
            mae,rmse=valid_metrics(mode_results[mode])
            ci=paired_interval(ref['prediction'],mode_data[mode]['prediction'],ref['target'],ref['valid'],
                               ref['origin'],cfg['analysis']['bootstrap_samples'],cfg['experiment']['seed'])
            summaries.append({'split':tag,'intervention':mode,'MAE':mae,'RMSE':rmse,
                              'delta_MAE_vs_full':mae-ref_mae,'delta_RMSE_vs_full':rmse-ref_rmse,**ci})
            if mode in ('no_support_demand','no_direction','no_propagation_alignment','no_regime',
                           'no_residual_information','no_evolution_information'):
                for rr in _gain_by_coverage(ref,mode_data[mode]):traffic.append({'split':tag,'intervention':mode,**rr})
        # Query/node/horizon risk diagnosis with unchanged prediction/target masks.
        nul=mode_data['no_evidence'];mask=ref['valid'].astype(bool)
        gain=np.abs(nul['prediction']-ref['target'])-np.abs(ref['prediction']-ref['target'])
        alpha=ref['alpha'].transpose(0,2,1)
        accepted=alpha>0.5
        for title,section in [('H1_4',slice(0,4)),('H5_8',slice(4,8)),('H9_12',slice(8,12))]:
            vv=mask[:,section];gg=gain[:,section];aa=accepted[:,section]
            if vv.sum():
                risk_rows.append({'split':tag,'horizon':title,'forecast_gain_raw':float(gg[vv].mean()),
                                  'accepted_fraction':float(aa[vv].mean()),
                                  'accepted_harmful_fraction':float(((gg<0)&aa&vv).sum()/((aa&vv).sum()+1e-8)),
                                  'rejected_beneficial_fraction':float(((gg>0)&(~aa)&vv).sum()/(((~aa)&vv).sum()+1e-8)),
                                  'alpha_mean':float(alpha[:,section][vv].mean()),
                                  'null_weight_mean':float(ref['null_weight'][:,:,section].mean())})
                horizon_rows.append({'split':tag,'horizon':title,
                                     'MAE':float(np.abs(ref['prediction'][:,section]-ref['target'][:,section])[vv].mean()),
                                     'RMSE':float(np.sqrt(np.square(ref['prediction'][:,section]-ref['target'][:,section])[vv].mean()))})
        evidence_rows.append({'split':tag,'residual_weight_mean':float(ref['residual_weight_mean'].mean()),
                              'evolution_weight_mean':float(ref['evolution_weight_mean'].mean()),
                              'null_weight_mean':float(ref['null_weight'].mean()),
                              'positive_gain_element_fraction':float(((gain>0)&mask).sum()/max(mask.sum(),1)),
                              'negative_gain_element_fraction':float(((gain<0)&mask).sum()/max(mask.sum(),1))})
        # Conditional interaction across the same fixed checkpoint (not retrained ablations)
        losses={k:valid_metrics(mode_results[k])[0] for k in ('full','no_evidence','no_residual_information','no_evolution_information')}
        # no_residual means Evolution only; no_evolution means Residual only
        interaction=losses['no_evolution_information']+losses['no_residual_information']-losses['full']-losses['no_evidence']
        interactions.append({'split':tag,'residual_evolution_interaction_MAE':interaction,
                             'null_MAE':losses['no_evidence'],'full_MAE':losses['full']})
        if tag==audit_split:
            counterfactual_audit(model,memory,subset,device,cfg,root/'counterfactual',split=tag)
    write_csv(root/'module_contributions.csv',summaries)
    write_csv(root/'traffic_support_analysis.csv',traffic)
    write_csv(root/'evidence_interactions.csv',interactions)
    write_csv(root/'risk_diagnostics.csv',risk_rows)
    write_csv(root/'evidence_usage.csv',evidence_rows)
    write_csv(root/'horizon_metrics.csv',horizon_rows)
    # Machine-readable conclusions include uncertainty, never a categorical claim from 1 seed.
    statements=[]
    for r in summaries:
        if r['intervention'] in ('full','scope_metadata'):continue
        lo=r.get('mae_ci_low');hi=r.get('mae_ci_high')
        evidence='inconclusive' if lo is None or hi is None or lo<=0<=hi else ('positive_fixed_checkpoint_dependence' if lo>0 else 'negative_fixed_checkpoint_dependence')
        statements.append({'split':r['split'],'module':r['intervention'],'interpretation':evidence,
                           'delta_MAE':r['delta_MAE_vs_full'],'confidence_interval':[lo,hi]})
    dump_json(root/'research_findings.json',{'scope':'same trained FARE v4 checkpoint; budgeted test interventions; no retraining; no external model comparisons',
                                            'counterfactual_audit_split':audit_split,
                                            'module_findings':statements,'interactions':interactions})
    md=['# FARE internal attribution','',
        'All rows are **fixed-checkpoint inference interventions** on a logged query budget (not retrained architecture ablations). Full Test metrics are produced separately by the evaluation pipeline.','',
        '| Split | Intervention | Delta MAE | Delta RMSE | 95% block bootstrap interval |','|---|---|---:|---:|---|']
    for r in summaries:
        if r['intervention']=='scope_metadata':continue
        md.append(f"| {r['split']} | {r['intervention']} | {r['delta_MAE_vs_full']:+.4f} | {r['delta_RMSE_vs_full']:+.4f} | [{r.get('mae_ci_low',float('nan')):.4f}, {r.get('mae_ci_high',float('nan')):.4f}] |")
    (root/'research_report.md').write_text('\n'.join(md),encoding='utf-8')
    # Self-contained searchable HTML tables; no third-party plotting dependency on Gadi.
    import html
    html_rows=[]
    for r in summaries:
        if r['intervention']=='scope_metadata':continue
        cells=[r['split'],r['intervention'],f"{r['delta_MAE_vs_full']:+.5f}",
               f"{r['delta_RMSE_vs_full']:+.5f}",f"{r.get('mae_ci_low',float('nan')):.5f}",
               f"{r.get('mae_ci_high',float('nan')):.5f}"]
        html_rows.append('<tr>'+''.join('<td>'+html.escape(str(cell))+'</td>' for cell in cells)+'</tr>')
    doc="""<!doctype html><html lang='en'><meta charset='utf-8'><title>FARE internal attribution</title>
<style>body{font:14px system-ui;margin:40px;color:#203044;background:#f7f9fc}h1{font-size:26px}
article{max-width:1100px;background:white;padding:28px;border-radius:12px;box-shadow:0 1px 12px #ddd}
input{padding:10px;width:350px;margin:16px 0}table{border-collapse:collapse;width:100%}th,td{padding:9px;
text-align:left;border-bottom:1px solid #e1e7ec}th{background:#e8f0f8;position:sticky;top:0}
small{color:#667}tr:hover{background:#f3f7ff}</style>
<article><h1>FARE v4 — Internal Attribution</h1><p>Single-checkpoint interventions only; no external-model comparisons or retraining.</p>
<input id='filter' placeholder='Filter by split or intervention' oninput='for(const r of document.querySelectorAll("tbody tr"))r.hidden=!r.innerText.toLowerCase().includes(this.value.toLowerCase())'>
<table><thead><tr><th>Split</th><th>Intervention</th><th>Δ MAE</th><th>Δ RMSE</th><th>95% CI low</th><th>95% CI high</th></tr></thead><tbody>"""+''.join(html_rows)+"</tbody></table><p><small>Positive delta indicates Full FARE performs better than the intervention under this fixed checkpoint.</small></p></article></html>"
    (root/'research_report.html').write_text(doc,encoding='utf-8')
    print('FARE_INTERNAL_ANALYSIS_COMPLETE',flush=True)
    return summaries
