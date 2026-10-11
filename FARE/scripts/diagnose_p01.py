#!/usr/bin/env python3
"""FARE P0.1: full fixed-checkpoint diagnostics on VALIDATION ONLY.

All four severities use the canonical Base_SupportBench sensing generator on
validation windows; this is explicitly diagnostic re-masking, NOT official
benchmark test or a promise of exact paired benchmark severity caches. The s40
validation mask is independently audited against the frozen s40 cache.

Oracle candidate/alpha use validation labels ONLY in offline analysis. Never
use their results as model inputs or an unbiased deployable score.
"""
from __future__ import annotations
import argparse
import math
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import _bootstrap
from fare.config import config_from_checkpoint,json_write
from fare.runtime import setup
from fare.trainer import load_ckpt
from fare.data import loader,to_device
from fare.diagnostics_p01 import (CanonicalValRemask,input_only,make_intervention,
    csv_save,metric_np,relative_difference,oracle_alpha_metrics,ALPHAS,spearman_pairs,small_summary)
from fare.ranking import target_candidate_errors


def result_np(pred, bt,mean,std):
    return (pred.detach().float()*std+mean).cpu().numpy()


def error_stats(a,y,v):
    a=np.asarray(a);y=np.asarray(y);v=np.asarray(v,bool)
    diff=a-y
    return float(np.abs(diff[v]).sum()),float(np.square(diff[v]).sum()),int(v.sum())


def weighted(rows, key):
    n=sum(row.get('count',0) for row in rows if row.get(key) is not None)
    return float(sum(row[key]*row['count'] for row in rows if row.get(key) is not None)/n) if n else None


def no_future_model_inputs(bt):
    inp=input_only(bt)
    assert not any(x in inp for x in ('x_full','y_raw','y_norm','future_valid','artificial_mask'))
    return inp


def retrieve_probe(model,memory,inp):
    return model.retrieve_for_diagnostics(inp,memory,diagnostics=True)


def random_branch(model,memory,inp,ret,ctx,kind,seed):
    new,meta=make_intervention(ret,kind,ctx,model.regions,memory,inp['window_start'],seed)
    out=model(inp,memory,retrieval_override=new,return_diagnostics=True)
    return new,out,meta


def _group_errors(pred,y,valid,group_mask):
    mask=valid&group_mask[None,None,:]
    diff=np.abs(pred-y)
    num=(diff*mask).sum((1,2));den=mask.sum((1,2))
    return np.where(den>0,num/np.maximum(den,1),np.nan),den


def candidate_probes(model,memory,dataset,batch_size,mean,std,probe_rows,probe_regions,probe_candidates,output,seed):
    """Exact local-region counterfactual: change ONLY one region's Top-K weights.

    Other regions and the weights of their retrieved evidence stay fixed; model
    graph adapters can still propagate effects across regions. Scores refer only
    to the intervened region's target nodes. Each probe computes the model's
    full FARE forward; it is intentionally sampled, not a 5000x64 sweep.
    """
    if probe_rows<=0:return {'status':'skipped','rows':[]}
    count=min(probe_rows,len(dataset))
    idx=np.linspace(0,len(dataset)-1,count,dtype=np.int64).tolist()
    chosen=Subset(dataset,idx)
    out_rows=[];rank_rows=[];eligible=0
    region_ids=np.unique(np.linspace(0,model.regions.g-1,min(probe_regions,model.regions.g),dtype=int))
    model.eval()
    with torch.no_grad():
        for j,batch in enumerate(loader(chosen,batch_size,workers=0)):
            bt=to_device(batch,next(model.parameters()).device)
            inp=no_future_model_inputs(bt)
            ctx,ret=retrieve_probe(model,memory,inp)
            selected=model(inp,memory,retrieval_override=ret)
            y=bt['y_raw'].float().cpu().numpy();valid=bt['future_valid'].bool().cpu().numpy()
            base=result_np(selected['base_prediction'],bt,mean,std)
            full=result_np(selected['prediction'],bt,mean,std)
            cand_selected=result_np(selected['candidate_prediction'],bt,mean,std)
            state_err, state_labels=target_candidate_errors(bt,ret,memory,model.regions)
            for g in region_ids:
                region_mask=model.regions.target_mask[g].detach().cpu().numpy().astype(bool)
                base_error,den=_group_errors(base,y,valid,region_mask)
                selected_error,_=_group_errors(full,y,valid,region_mask)
                selected_cand_error,_=_group_errors(cand_selected,y,valid,region_mask)
                best=np.full(len(base_error),np.inf)
                any_candidate=np.zeros(len(base_error),dtype=bool)
                for k in range(min(probe_candidates,ret['ids'].shape[-1])):
                    # One-hot intervention, limited to legal candidates. Other
                    # regions are unchanged. Completely invalid samples skipped.
                    vk=ret['valid'][:,g,k].detach().cpu().numpy().astype(bool)
                    if not vk.any():continue
                    weights=ret['weights'].clone()
                    weights[:,g,:]=0
                    weights[:,g,k]=ret['valid'][:,g,k].float()
                    intervention={**ret,'weights':weights}
                    probe=model(inp,memory,retrieval_override=intervention)
                    y_probe=result_np(probe['prediction'],bt,mean,std)
                    pred_candidate=result_np(probe['candidate_prediction'],bt,mean,std)
                    probe_error,_=_group_errors(y_probe,y,valid,region_mask)
                    probe_cand_error,_=_group_errors(pred_candidate,y,valid,region_mask)
                    for b in np.flatnonzero(vk&(den>0)):
                        best[b]=min(best[b],probe_error[b]);any_candidate[b]=True
                        row={'query_id':int(bt['row_id'][b]),'window_start':int(bt['window_start'][b]),
                             'family':str(dataset.meta.iloc[int(bt['row_id'][b])]['family']),
                             'region':int(g),'slot':int(k),'memory_id':int(ret['ids'][b,g,k]),
                             'memory_start':int(memory.starts[ret['ids'][b,g,k]]),
                             'ranking_score':float(ret['scores'][b,g,k]),
                             'ranking_weight':float(ret['weights'][b,g,k]),
                             'state_error':(float(state_err[b,g,k]) if bool(state_labels[b,g,k]) else None),
                             'base_mae':float(base_error[b]),
                             'selected_final_mae':float(selected_error[b]),
                             'onehot_final_mae':float(probe_error[b]),
                             'onehot_candidate_mae':float(probe_cand_error[b]),
                             'onehot_final_utility':float(base_error[b]-probe_error[b]),
                             'onehot_candidate_utility':float(base_error[b]-probe_cand_error[b]),
                             'count':int(den[b])}
                        out_rows.append(row)
                for b in np.flatnonzero(any_candidate):
                    eligible+=1
                    rank_rows.append({'query_id':int(bt['row_id'][b]),'region':int(g),
                                     'base_mae':float(base_error[b]),
                                     'selected_mae':float(selected_error[b]),
                                     'selected_candidate_mae':float(selected_cand_error[b]),
                                     'best_probe_mae':float(min(best[b],base_error[b])),
                                     'null_can_win':bool(base_error[b]<=best[b]),
                                     'candidate_slots_tested':int(sum(x['query_id']==int(bt['row_id'][b]) and x['region']==int(g) for x in out_rows)),
                                     'count':int(den[b])})
            print(f'FARE_P01_CANDIDATE_BATCH={j} collected={len(out_rows)}',flush=True)
    csv_save(output/'candidate_probes.csv',out_rows)
    csv_save(output/'candidate_oracle_regions.csv',rank_rows)
    correlation=spearman_pairs([x['ranking_score'] for x in out_rows],[x['onehot_final_utility'] for x in out_rows])
    state_correlation=spearman_pairs([-x['state_error'] if x['state_error'] is not None else np.nan for x in out_rows],[x['onehot_final_utility'] for x in out_rows])
    grouped=defaultdict(list)
    for row in out_rows:grouped[(row['query_id'],row['region'])].append(row)
    within=[]
    for candidates in grouped.values():
        rho=spearman_pairs([x['ranking_score'] for x in candidates],
                           [x['onehot_final_utility'] for x in candidates])
        if rho is not None:within.append(rho)
    report={'cohort_rows':count,'regions_probed':list(map(int,region_ids)),
            'per_candidate_records':len(out_rows),'eligible_region_queries':eligible,
            'rank_score_vs_utility_spearman_pooled':correlation,
            'rank_score_vs_utility_spearman_within_query_region_mean':float(np.mean(within)) if within else None,
            'rank_score_vs_utility_spearman_within_query_region_count':len(within),
            'neg_state_error_vs_utility_spearman_pooled':state_correlation,
            'positive_probe_final_utility_fraction':(float(np.mean([x['onehot_final_utility']>0 for x in out_rows])) if out_rows else None),
            'selected_region_mae':weighted(rank_rows,'selected_mae'),
            'oracle_region_mae':weighted(rank_rows,'best_probe_mae'),
            'oracle_optimism_warning':'Validation label-informed local-region candidate selection. Not deployable; not a model score.',
            'counterfactual_scope':'One region slot weight onehot; other regions held fixed; all graph interactions recomputed.'}
    json_write(output/'candidate_utility.json',report)
    return report


def run(args):
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=config_from_checkpoint(ck)
    if cfg['experiment']['target'] != args.target:raise ValueError('Target/checkpoint mismatch')
    model,memory,train,val,device,meta=setup(cfg,ck['metadata']['backbone_checkpoint'],ck.get('memory_path'))
    if meta['backbone_sha256']!=ck['metadata']['backbone_sha256']:raise RuntimeError('Backbone checksum mismatch')
    load_ckpt(args.checkpoint,model,device)
    model.eval();model.refresh(memory)
    outdir=Path(args.output);outdir.mkdir(parents=True,exist_ok=True)
    log={'status':'starting','target':args.target,'dataset':cfg['experiment']['dataset'],
         'checkpoint':str(args.checkpoint),'memory_path':str(ck.get('memory_path')),
         'seed':args.seed,'evaluation':'validation only; diagnostic re-mask per severity',
         'severities':args.severities,'modes':args.modes}
    json_write(outdir/'manifest.json',log)
    full_rows=[];all_reports={}
    for severity in args.severities:
        print(f'FARE_P01_SEVERITY_START target={args.target} severity={severity}',flush=True)
        if severity not in (20,40,60,90):raise ValueError(f'Unsupported severity {severity}')
        ds=CanonicalValRemask(val,severity,cfg['experiment']['dataset'],args.target,
                              verify_rows=args.verify_reference_rows)
        folder=outdir/f's{severity:02d}';folder.mkdir(parents=True,exist_ok=True)
        diag_rows=[];gate_rows=[];all_arrays=defaultdict(list);identity=[]
        gate_buckets=defaultdict(list)
        total_batches=math.ceil(len(ds)/args.batch_size)
        sampled_batches=set(np.linspace(0,total_batches-1,min(args.sensitivity_batches,total_batches),dtype=int).tolist())
        with torch.no_grad():
            for bi,batch in enumerate(loader(ds,args.batch_size,workers=0)):
                if args.max_batches and bi>=args.max_batches:break
                bt=to_device(batch,device);inp=no_future_model_inputs(bt)
                ctx,ret=retrieve_probe(model,memory,inp)
                selected=model(inp,memory,retrieval_override=ret,return_diagnostics=True)
                v=bt['future_valid'].detach().cpu().numpy().astype(bool)
                y=bt['y_raw'].detach().cpu().numpy().astype(np.float32)
                b=result_np(selected['base_prediction'],bt,meta['mean'],meta['std'])
                c=result_np(selected['candidate_prediction'],bt,meta['mean'],meta['std'])
                f=result_np(selected['prediction'],bt,meta['mean'],meta['std'])
                gate=selected['final_gate'].detach().cpu().numpy().astype(np.float32)
                support=ret['group_valid'][:,model.regions.owner][:,None,:].expand_as(selected['final_gate']).detach().cpu().numpy()
                all_arrays['base'].append(b);all_arrays['candidate'].append(c)
                all_arrays['full'].append(f);all_arrays['gate'].append(gate)
                all_arrays['y'].append(y);all_arrays['valid'].append(v)
                all_arrays['direct'].append(bt['direct_mask'].detach().cpu().numpy().astype(bool))
                all_arrays['support'].append(support)
                all_arrays['starts'].append(bt['window_start'].cpu().numpy())
                all_arrays['family'].extend(ds.meta.iloc[bt['row_id'].cpu().numpy()].family.astype(str).tolist())
                if bi in sampled_batches:
                    for j,kind in enumerate(args.modes):
                        if kind=='selected':continue
                        for repeat in range(1 if kind=='uniform_selected' else args.random_repeats):
                            newer,alternative,control=random_branch(model,memory,inp,ret,ctx,kind,
                                    args.seed+int(severity)*100_000+bi*1000+repeat*37+j)
                            change=(newer['ids']!=ret['ids'])
                            overlap_values=[]
                            for bb in range(len(b)):
                                for gg in range(model.regions.g):
                                    selected_ids=set(ret['ids'][bb,gg][ret['valid'][bb,gg]].tolist())
                                    random_ids=set(newer['ids'][bb,gg][newer['valid'][bb,gg]].tolist())
                                    if selected_ids:overlap_values.append(len(selected_ids & random_ids)/len(selected_ids))
                            overlap=float(np.mean(overlap_values)) if overlap_values else 0.
                            cand_alt=result_np(alternative['candidate_prediction'],bt,meta['mean'],meta['std'])
                            final_alt=result_np(alternative['prediction'],bt,meta['mean'],meta['std'])
                            # Verify actual Memory states and latents differ;
                            # changing candidate IDs alone is insufficient.
                            xref=memory.x[ret['ids']]
                            xalt=memory.x[newer['ids']]
                            dstate=relative_difference(xref,xalt)
                            dlatent=(relative_difference(memory.latent[ret['ids']],memory.latent[newer['ids']])
                                     if memory.latent is not None else None)
                            sens={'severity':severity,'batch':bi,'kind':kind,'repeat':repeat,'rows':len(b),
                                  'id_changed_fraction':float(change.float().mean()),
                                  'selected_random_overlap_at_k':float(overlap),
                                  'selected_legal_fraction':float(ret['valid'].float().mean()),
                                  'alternative_legal_fraction':float(newer['valid'].float().mean()),
                                  'time_matching_fallback_groups':control['fallback_groups'],
                                  'historical_state_delta_rel':dstate['delta_rel'],
                                  'historical_latent_delta_rel':dlatent['delta_rel'] if dlatent else None,
                                  'candidate_delta_abs':float(np.abs(c-cand_alt).mean()),
                                  'final_delta_abs':float(np.abs(f-final_alt).mean()),
                                  'selected_MAE':metric_np(f,y,v)['MAE'],
                                  'alternative_MAE':metric_np(final_alt,y,v)['MAE']}
                            for key in ('recovered_state','state_uncertainty','latent_innovation',
                                        'latent_candidate_embedding','retrieved_support_tokens','supported_final_hidden',
                                        'candidate_prediction','prediction','final_gate'):
                                d=relative_difference(selected.get(key),alternative.get(key))
                                if d:
                                    sens[key+'_delta_abs']=d['delta_abs']
                                    sens[key+'_delta_rel']=d['delta_rel']
                            diag_rows.append(sens)
                            if args.source_swaps and kind=='random_global' and repeat==0:
                                # Diagnostic-only counterfactual: keep original
                                # candidate weights / retrieval valid groups, but
                                # swap only state or only latent Memory IDs.
                                # This intentionally breaks State/Latent identity
                                # coherence, and is never used as a valid model.
                                for branch in ('state_ids','latent_ids'):
                                    hybrid={**ret,branch:newer['ids']}
                                    swapped=model(inp,memory,retrieval_override=hybrid,return_diagnostics=True)
                                    brow={'severity':severity,'batch':bi,'kind':'source_swap_'+branch,
                                          'repeat':0,'rows':len(b),
                                          'warning':'Diagnostic-only incoherent cross-source intervention'}
                                    for key in ('recovered_state','latent_innovation','candidate_prediction',
                                                'prediction','supported_final_hidden','final_gate'):
                                        d=relative_difference(selected.get(key),swapped.get(key))
                                        if d:
                                            brow[key+'_delta_abs']=d['delta_abs']
                                            brow[key+'_delta_rel']=d['delta_rel']
                                    diag_rows.append(brow)
                if bi%100==0:print(f'FARE_P01_PROGRESS target={args.target} s{severity} batch={bi}',flush=True)
        if not all_arrays['base']:raise RuntimeError('No validation batches')
        arrays={k:np.concatenate(v,axis=0) for k,v in all_arrays.items() if k!='family'}
        arrays['family']=np.asarray(all_arrays['family'])
        y=arrays['y'];v=arrays['valid'];b=arrays['base'];c=arrays['candidate'];f=arrays['full'];gate=arrays['gate'];support=arrays['support']
        vm=v&arrays['direct'][:,None,:]
        summary={'target':args.target,'severity':severity,'rows':len(y),'reference_s40_mask_audit':ds.audit,
                 'base':metric_np(b,y,v),'candidate':metric_np(c,y,v),'full':metric_np(f,y,v),
                 'directly_masked_base':metric_np(b,y,vm),
                 'directly_masked_candidate':metric_np(c,y,vm),
                 'directly_masked_full':metric_np(f,y,vm),
                 'full_gate':small_summary(gate[v]),
                 'retrieval_support_fraction':float(support.mean()),
                 'correction_direction_agreement':float(np.mean(np.sign((c-b)[v])==np.sign((y-b)[v]))),
                 'helpful_final_fraction':float((np.abs(f-y)[v]<np.abs(b-y)[v]).mean()),
                 'harmful_final_fraction':float((np.abs(f-y)[v]>np.abs(b-y)[v]).mean())}
        # Calibrate only on earlier validation time windows, evaluate on later
        # validation windows, so grid-search scores are not reported in-sample.
        unique=np.unique(arrays['starts']);cut=unique[max(0,int(len(unique)*.6)-1)]
        calibration=arrays['starts']<=cut
        heldout=arrays['starts']>cut
        if not calibration.any() or not heldout.any():raise RuntimeError('Cannot split validation time groups')
        alpha_rows=[]
        for alpha in ALPHAS:
            pp=b+float(alpha)*support*(c-b)
            mr=metric_np(pp[heldout],y[heldout],v[heldout])
            mc=metric_np(pp[calibration],y[calibration],v[calibration])
            alpha_rows.append({'alpha':alpha,'calibration_MAE':mc['MAE'],'heldout_MAE':mr['MAE'],
                               'heldout_RMSE':mr['RMSE']})
        alpha_best=min(alpha_rows,key=lambda z:(z['calibration_MAE'],z['alpha']))
        oracle,oracle_gate=oracle_alpha_metrics(b,c,y,v,max_alpha=model.controller.max_alpha,support=support)
        matched_alpha=float(gate[support.astype(bool)].mean()) if support.any() else 0.
        matched=metric_np(b+matched_alpha*support*(c-b),y,v)
        summary['control']={'alpha_best_on_calibration':alpha_best,'alpha_grid':alpha_rows,
                            'matched_mean_alpha':matched_alpha,'matched_fixed_metrics':matched,
                            'full_heldout':metric_np(f[heldout],y[heldout],v[heldout]),
                            'oracle_pointwise_mae':oracle,'oracle_pointwise_mean_alpha':oracle_gate,
                            'oracle_warning':'Uses validation future labels; not deployable or comparable as a normal model'}
        famrows=[]
        for fam in sorted(set(arrays['family'])):
            mask=arrays['family']==fam
            famrows.append({'severity':severity,'family':fam,'rows':int(mask.sum()),
                            'base_MAE':metric_np(b[mask],y[mask],v[mask])['MAE'],
                            'candidate_MAE':metric_np(c[mask],y[mask],v[mask])['MAE'],
                            'full_MAE':metric_np(f[mask],y[mask],v[mask])['MAE'],
                            'base_RMSE':metric_np(b[mask],y[mask],v[mask])['RMSE'],
                            'candidate_RMSE':metric_np(c[mask],y[mask],v[mask])['RMSE'],
                            'full_RMSE':metric_np(f[mask],y[mask],v[mask])['RMSE'],
                            'masked_full_MAE':metric_np(f[mask],y[mask],vm[mask])['MAE'],
                            'valid_support_fraction':float(support[mask].mean()),
                            'gate_mean':float(gate[mask].mean())})
        groups=[]
        diff=np.abs(b-y)-np.abs(f-y)
        for q in range(5):
            cutlo=np.percentile(gate[v],20*q);cuthi=np.percentile(gate[v],20*(q+1))
            subset=v&(gate>=cutlo)&(gate<=cuthi)
            groups.append({'gate_quintile':q+1,'count':int(subset.sum()),'mean_gain':float(diff[subset].mean()) if subset.any() else None})
        summary['control']['gate_quintile_utility']=groups
        csv_save(folder/'integrity_sensitivity.csv',diag_rows)
        csv_save(folder/'family_metrics.csv',famrows)
        csv_save(folder/'alpha_sweep.csv',alpha_rows)
        json_write(folder/'diagnostic_summary.json',summary)
        if args.save_predictions:
            np.savez_compressed(folder/'diagnostic_val_predictions.npz',**{
                k:val.astype(np.float32) if k not in ('valid','starts','family','support','direct') else val
                for k,val in arrays.items()})
        if args.probe_rows:
            summary['candidate_utility']=candidate_probes(model,memory,ds,args.batch_size,meta['mean'],meta['std'],
                args.probe_rows,args.probe_regions,args.probe_candidates,folder,args.seed+severity)
            json_write(folder/'diagnostic_summary.json',summary)
        full_rows.append({'target':args.target,'severity':severity,'rows':len(y),
                          'base_MAE':summary['base']['MAE'],'candidate_MAE':summary['candidate']['MAE'],
                          'full_MAE':summary['full']['MAE'],'base_RMSE':summary['base']['RMSE'],
                          'full_RMSE':summary['full']['RMSE'],
                          'MAE_gain_pct':100*(summary['base']['MAE']-summary['full']['MAE'])/summary['base']['MAE'],
                          'retrieval_support_fraction':summary['retrieval_support_fraction'],
                          'directly_masked_full_MAE':summary['directly_masked_full']['MAE'],
                          'oracle_pointwise_MAE':oracle['MAE'],
                          'candidate_probed':bool(args.probe_rows)})
        all_reports[f's{severity:02d}']=summary
        print('FARE_P01_SEVERITY_PASS '+json.dumps(full_rows[-1]),flush=True)
    csv_save(outdir/'p01_overview.csv',full_rows)
    report={'status':'PASS','dataset':cfg['experiment']['dataset'],'target':args.target,
            'validation_only':True,'full_per_severity':full_rows,
            'avg_MAE_gain_pct':float(np.mean([x['MAE_gain_pct'] for x in full_rows])),
            'warning':'Oracle labels only in offline analysis; remasked validation is not official test benchmark.',
            'next':'Use retrieval sensitivity and candidate utility to choose v2.0 mechanisms.'}
    json_write(outdir/'p01_report.json',report)
    (outdir/'p01_report.md').write_text('# FARE P0.1 — '+args.target+'\n\nValidation-only diagnostic re-mask, not official Test.\n\n'+
             '|Severity|Base MAE|Candidate MAE|Full MAE|Gain %|\n|---|---:|---:|---:|---:|\n'+
             ''.join(f"|s{r['severity']}|{r['base_MAE']:.4f}|{r['candidate_MAE']:.4f}|{r['full_MAE']:.4f}|{r['MAE_gain_pct']:.3f}|\n" for r in full_rows)+
             '\nDetailed diagnostics are under `sXX/`. Oracle results use validation truth and are not deployable.\n',encoding='utf-8')
    print(f'FARE_P01_TARGET_PASS target={args.target} output={outdir}',flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--target',choices=('speed','flow'),required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--severities',type=int,nargs='+',default=[20,40,60,90])
    p.add_argument('--modes',nargs='+',choices=['uniform_selected','random_top64','random_global','time_matched'],default=['uniform_selected','random_top64','random_global','time_matched'])
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--sensitivity-batches',type=int,default=8,help='Probe N evenly spaced validation batches per severity')
    p.add_argument('--random-repeats',type=int,default=3)
    p.add_argument('--probe-rows',type=int,default=64)
    p.add_argument('--probe-regions',type=int,default=4)
    p.add_argument('--probe-candidates',type=int,default=8)
    p.add_argument('--verify-reference-rows',type=int,default=32)
    p.add_argument('--seed',type=int,default=10)
    p.add_argument('--max-batches',type=int,default=0,help='0=all validation, >0=diagnostic smoke only')
    p.add_argument('--save-predictions',action='store_true')
    p.add_argument('--no-source-swaps',dest='source_swaps',action='store_false')
    p.set_defaults(source_swaps=True)
    args=p.parse_args()
    if args.random_repeats<1 or args.probe_regions<0 or args.sensitivity_batches<0:raise ValueError('Invalid diagnostic budgets')
    if not 1<=args.batch_size<=64:raise ValueError('P0.1 batch size must be 1..64')
    if args.max_batches and args.probe_rows>args.max_batches*args.batch_size:
        print('FARE_P01_WARNING probe_rows exceeds max_batches rows: candidate probe is a separate stratified validation sample',flush=True)
    run(args)

if __name__=='__main__':main()
