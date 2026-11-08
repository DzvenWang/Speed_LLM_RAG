"""Same-cohort post-training CF mechanism diagnostics, tables and HTML report.

All comparisons are retrospective VALIDATION checks, and must not be used to
select test-time candidates. Stage selection used a disjoint chronological
selection subset; the diagnosis below uses chronological CHECK. Controls from
one checkpoint are BEHAVIOR tests, not retrained-architecture ablations.
"""
from __future__ import annotations
import csv,html,json,math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from . import VERSION
from .training import create_model,forward_batch
from .prepare import V9EvidenceDataset
from .utility import oracle_alignment,marginal_targets
from ..v9.graph import GraphBundle
from ..v9.training import to_device
from ..v9.model import utility_targets
from ..v5.calibration import chronological_partitions

CONTROLS={
    'full':{},'normal_only':dict(enable_cf=False),
    'uniform_utility':dict(rank_mode='uniform'),
    'similarity_utility':dict(rank_mode='similarity'),
    'positive_alignment':dict(alignment_mode='positive'),
    'raw_cf':dict(alignment_mode='raw',rank_mode='uniform',null_mode='fixed'),
    'no_cf_graph':dict(graph_mode='none'),
    'fixed_graph':dict(graph_mode='fixed'),
    'gnn_no_reliability':dict(graph_mode='gnn'),
    'no_temporal_cnn':dict(temporal_mode='stats'),
    'fixed_null':dict(null_mode='fixed'),
    'no_recoverability':dict(recoverability_enabled=False),
    'no_cf':dict(enable_cf=False),
    'shuffled_cf':{},
}

def load_checkpoint_model(checkpoint,device):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('version')!=VERSION:raise RuntimeError('Not a v9.2 checkpoint')
    sd=ck['model']
    prefix='normal_core.graph.'
    sign=sd[prefix+'functional_signed'].sign()
    graph=GraphBundle(sd[prefix+'functional'].float(),sd[prefix+'upstream'].float(),
         sd[prefix+'downstream'].float(),functional_sign=sign)
    model=create_model(graph,ck['context_dim'],ck['cfg']).to(device)
    model.load_state_dict(sd,strict=True);model.eval()
    return model,ck


def safe_stat(x):
    a=np.asarray(x,np.float64)
    if len(a)==0:return None
    return float(np.mean(a))


def block_bootstrap(data,block_ids,repeats=200,seed=10):
    arr=np.asarray(data,np.float64); ids=np.asarray(block_ids)
    if len(arr)!=len(ids) or not len(arr):return {'ci95':None,'mean':None,'block_count':0}
    uniq=np.unique(ids)
    if len(uniq)<3:return {'ci95':None,'mean':float(arr.mean()),'block_count':len(uniq)}
    groups=[arr[ids==id] for id in uniq]
    rng=np.random.default_rng(seed);scores=[]
    for _ in range(repeats):
        draw=rng.integers(0,len(uniq),len(uniq))
        sample=np.concatenate([groups[i] for i in draw])
        scores.append(float(sample.mean()))
    return {'ci95':[float(x) for x in np.quantile(scores,[.025,.975])],
            'mean':float(arr.mean()),'block_count':len(uniq)}


def _numpy_json(value):
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)


def diagnose_run(run_identity,max_queries=256,batch_size=8,bootstrap=200,wrong_mask_queries=0):
    run_identity=Path(run_identity)
    info=json.loads(run_identity.read_text())
    if info.get('version')!=VERSION:raise ValueError('v9.2 identity required')
    run=run_identity.parent;outdir=run/'diagnostics';outdir.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=V9EvidenceDataset(info['formal_cache'],'val',(0,),root=info['evidence_root'])
    idx=(list(range(len(ds))) if info['smoke'] else
        chronological_partitions(ds.ds.window_starts[:len(ds.ds)],12,12)['check'])
    idx=idx[:min(max_queries,len(idx))]
    if not idx:raise RuntimeError('Empty diagnostic cohort')
    cohort=np.asarray(idx,np.int64);window_starts=ds.ds.window_starts[cohort]
    import hashlib
    cohort_sha=hashlib.sha256(cohort.astype('<i8').tobytes()).hexdigest()
    dataloader=DataLoader(Subset(ds,idx),batch_size=batch_size,num_workers=0,
        pin_memory=device.type=='cuda')
    checkpoints=[run/f'best_{stage}.pt' for stage in ('normal','align','utility','joint','paired')]
    checkpoints=[p for p in checkpoints if p.is_file()]
    if not checkpoints:raise FileNotFoundError('No trained stage checkpoints to diagnose')
    records=[];component_metrics={};per_query={};alignment_metrics={};utility_metrics={}
    candidate_ids_hash=None
    for ckpath in checkpoints:
        model,ck=load_checkpoint_model(ckpath,device)
        stage=ck['stage'];row_errors=defaultdict(list);row_gains=[];row_mask_frac=[];row_ids=[]
        alpha_vals=[];gate_vals=[];all_null=[];harm_count=0;count=0
        by_horizon_gain=np.zeros(12,np.float64);by_horizon_count=np.zeros(12,np.float64)
        pred_u=[];true_u=[];selected_regret=[];pred_m=[];true_m=[]
        with torch.inference_mode():
            for bb in dataloader:
                b=to_device(bb,device)
                baseargs={key:b[key] for key in ('base','context','x','mask','normal','cf','score','valid','match')}
                with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    full=model(**baseargs,enable_cf=(stage!='normal'))
                    normal=full['normal_prediction']
                    variants={key:(model(**baseargs,**({**arg,'enable_cf':False} if stage=='normal' else arg))['prediction'] if key not in ('full','normal_only','no_cf','shuffled_cf')
                                   else full['prediction'] if key=='full' else normal)
                              for key,arg in CONTROLS.items()}
                    variants['shuffled_cf']=(model(**{**baseargs,'cf':b['cf'].roll(1,dims=1)})['prediction']
                                              if stage!='normal' else normal)
                # Direct failure residual, fixed Normal weights and the SAME candidate pool.
                with torch.autocast(device_type=device.type,enabled=False):
                    wn=full['normal_weights'].float()
                    failure=(b['normal'].float()+b['cf'].float())
                    fraw=(wn[:,:,None,:]*failure).sum(1)
                    direct=b['base'].float()+model.bound_fn(fraw)
                    variants['direct_failure_residual']=direct
                valid=b['future_valid'].bool()
                y=torch.where(valid,b['y_norm'].float(),torch.zeros_like(b['y_norm'].float()))
                nerror=torch.where(valid,(normal.float()-y).abs(),torch.zeros_like(y))
                count+=int(valid.sum())
                harm_count+=int(torch.where(valid,(full['prediction'].float()-y).abs()>nerror+1e-6,False).sum())
                for variant,pred in variants.items():
                    error=torch.where(valid,(pred.float()-y).abs(),torch.zeros_like(y))
                    row=(error.sum((1,2))/valid.sum((1,2)).clamp_min(1)).cpu().numpy()
                    row_errors[variant].extend(map(float,row))
                gain_tensor=nerror-torch.where(valid,(full['prediction'].float()-y).abs(),0)
                row_gains.extend(map(float,(gain_tensor.sum((1,2)) /
                                      valid.sum((1,2)).clamp_min(1)).cpu().numpy()))
                by_horizon_gain += gain_tensor.sum((0,2)).cpu().numpy()
                by_horizon_count += valid.sum((0,2)).cpu().numpy()
                row_mask_frac.extend(map(float,1-b['mask'].float().mean((1,2)).cpu().numpy()))
                row_ids.extend(map(int,b['query_id'].cpu().numpy()))
                if 'alignment' in full:
                    valid_alpha=b['valid'][:,:,None,None].expand_as(full['alignment'])
                    alpha_vals.append(full['alignment'].float()[valid_alpha].flatten().cpu())
                    gate_vals.append(full['graph_gates'].float().reshape(-1,3).cpu())
                    all_null.append(full['cf_weights'][:,-1].float().flatten().cpu())
                    u,uv=utility_targets(full['base'].float(),full['normal_raw'].float(),
                           full['candidate_cf'].float(),b['y_norm'].float(),b['valid'],model.bound,
                           blocks=model.blocks,future_valid=valid,return_valid=True)
                    pred_u.extend(full['cf_utility'][uv].float().cpu().tolist())
                    true_u.extend(u[uv].float().cpu().tolist())
                    if 'marginal_utility' in full:
                        marg,mv=marginal_targets(full,b['y_norm'].float(),valid,b['valid'],model.blocks)
                        pred_m.extend(full['marginal_utility'][mv].float().cpu().tolist())
                        true_m.extend(marg[mv].float().cpu().tolist())
                    u_with_null=torch.cat([u.masked_fill(~uv,-1e6),torch.zeros_like(u[:,:1])],1)
                    selected=full['cf_weights'].argmax(1)
                    score=u_with_null.gather(1,selected[:,None]).squeeze(1)
                    regret=u_with_null.amax(1)-score
                    ok=valid.reshape(len(valid),model.blocks,-1,valid.shape[-1]).any(2)
                    selected_regret.extend(regret[ok].float().cpu().tolist())
        stage_metrics={}
        if list(map(int,row_ids))!=list(map(int,idx)):
            raise AssertionError('Diagnostic cohort order has shifted')
        for name,rows in row_errors.items():
            r=np.asarray(rows);gain=np.asarray(row_errors['normal_only'])-r
            summary=block_bootstrap(gain,np.asarray(window_starts)//288,repeats=bootstrap,seed=info['seed'])
            stage_metrics[name]={'mae_norm':float(r.mean()),'mae_raw':float(r.mean()*ck['cfg'].get('std',1.)),
                 'gain_vs_normal_norm':float(gain.mean()),'gain_ci95_norm':summary['ci95'],
                 'time_block_count':summary['block_count'],'query_win_rate':float((gain>0).mean())}
        if alpha_vals:
            alpha=torch.cat(alpha_vals).numpy();gates=torch.cat(gate_vals).numpy();null=torch.cat(all_null).numpy()
            alignment_metrics[stage]=dict(alpha_mean=float(alpha.mean()) if len(alpha) else None,negative_rate=float((alpha<-.05).mean()) if len(alpha) else None,
               near_zero_rate=float((abs(alpha)<.05).mean()) if len(alpha) else None,saturation_rate=float((abs(alpha)>model.max_align*.95).mean()) if len(alpha) else None,
               source_gate_mean=gates.mean(0).tolist(),null_mean=float(null.mean()),
               physical_graph_available=bool(ck.get('graph_physical',False)))
        if pred_u:
            p=np.asarray(pred_u,np.float32); t=np.asarray(true_u,np.float32)
            corr=float(np.corrcoef(p,t)[0,1]) if p.std()>1e-10 and t.std()>1e-10 else None
            utility_metrics[stage]=dict(utility_mae=float(np.mean(abs(p-t))),utility_pearson=corr,
                oracle_top1_regret=float(np.mean(selected_regret)) if selected_regret else None,
                marginal_utility_mae=(float(np.mean(np.abs(np.asarray(pred_m)-np.asarray(true_m)))) if pred_m else None),
                valid_utility_positions=int(len(t)))
        component_metrics[stage]=dict(stage=stage,checkpoint=str(ckpath),results=stage_metrics,
                      cf_harm_rate=harm_count/max(count,1),cohort_rows=len(idx),
                      cf_gain_by_horizon_norm=(by_horizon_gain/np.maximum(by_horizon_count,1)).tolist())
        per_query[stage]=dict(query_ids=row_ids,cf_gain=row_gains,missing_fraction=row_mask_frac)
        for name,metrics in stage_metrics.items():
            records.append({'stage':stage,'variant':name,**metrics})
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    selection=json.loads((run/'selected_checkpoint.json').read_text())['stage']
    best=component_metrics[selection];main=best['results']
    cf_gain=main['full']['gain_vs_normal_norm']
    graph_gap=main['no_cf_graph']['mae_norm']-main['full']['mae_norm']
    utility_gap=main['uniform_utility']['mae_norm']-main['full']['mae_norm']
    positive_gap=main['positive_alignment']['mae_norm']-main['full']['mae_norm']
    recommendations=[]
    def propose(module,level,message):recommendations.append(dict(module=module,status=level,reason=message))
    ci=main['full']['gain_ci95_norm']
    if ci is None:propose('CF','insufficient_evidence','Fewer than three independent time blocks')
    elif ci[0]>0:propose('CF','positive_validation_evidence','Full model improves fixed Normal on diagnostic cohort')
    elif ci[1]<0:propose('CF','negative_validation_evidence','Full model harms fixed Normal on diagnostic cohort')
    else:propose('CF','inconclusive','Block-bootstrap CI overlaps zero')
    for mod,gap in [('GNN',graph_gap),('Utility',utility_gap),('Signed Alignment',positive_gap)]:
        propose(mod,'behavioral_only',f'Same-checkpoint control MAE gap {gap:+.6f} norm; retraining needed for causal module claim')
    if not alignment_metrics.get(selection,{}).get('physical_graph_available',False):
        propose('Physical Graph','not_active','No grounded upstream/downstream graph supplied; only Functional graph is active')
    propose('Paired Supervision','requires_retrained_ablation','Compare paired training against same-budget unpaired multi-mask training')
    propose('CF Failure Specificity','pending_same_cohort_control','Wrong-mask must rerun the frozen STLA teacher on the identical query subset')
    report=dict(version=VERSION,target=info['target'],dataset=info['dataset'],seed=info['seed'],
       split='validation_chronological_check' if not info['smoke'] else 'smoke_overlap_not_for_research',
       cohort_sha256=cohort_sha,cohort_rows=len(idx),selected_stage=selection,
       scope='Same-checkpoint controls are behavioral tests, not retrained ablations',
       stages=component_metrics,alignment=alignment_metrics,utility=utility_metrics,
       recommendations=recommendations,
       not_run=['same_budget_multimask_retraining','independent_no_gnn_retraining'])
    history_file=run/'history.json'
    if history_file.is_file():
        history=json.loads(history_file.read_text())
        gradient_audit={}
        for stage in ('normal','align','utility','joint','paired'):
            epochs=[e for e in history if e['stage']==stage]
            if epochs:
                gradient_audit[stage]={'trained_epochs':len(epochs),
                   'last_gradient_norm':epochs[-1]['train'].get('gradient_norm',{}),
                   'latest_cf_gain_raw':epochs[-1]['val'].get('cf_gain_raw'),
                   'last_paired_matching_rate':epochs[-1]['train'].get('matching_rate')}
        report['training_integrity']=gradient_audit
        (outdir/'gradient_audit.json').write_text(json.dumps(gradient_audit,indent=2))
    # Retrospective local oracle headroom: future labels are used ONLY in this
    # diagnostic block, never in inference. Local-oracle gain is optimistic.
    if selection!='normal':
        model,_=load_checkpoint_model(run/f'best_{selection}.pt',device)
        oracle_positive=[];oracle_signed=[];ora_count=0
        with torch.inference_mode():
            for i,batch in enumerate(dataloader):
                if ora_count>=min(64,len(idx)):break
                b=to_device(batch,device);out=forward_batch(model,b)
                valid=b['future_valid'].bool()
                norm=out['normal_prediction'].float();y=b['y_norm'].float()
                norm_err=torch.where(valid,(norm-y).abs(),0).reshape(len(y),model.blocks,-1,y.shape[-1])
                count=valid.reshape(len(y),model.blocks,-1,y.shape[-1]).sum(2)
                norm_err=norm_err.sum(2)/count.clamp_min(1)
                signed,svalid,smeta=oracle_alignment(out,b['cf'],y,valid,b['valid'],model.blocks)
                positive,pvalid,pmeta=oracle_alignment(out,b['cf'],y,valid,b['valid'],model.blocks,
                                                 scales=(0.,.5,1.,1.5))
                # Return best single-candidate and Null at each node/block.
                for tlist,metadata in ((oracle_signed,smeta),(oracle_positive,pmeta)):
                    corrected=metadata['signed_best'].masked_fill(~svalid if tlist is oracle_signed else ~pvalid,1e9).amin(1)
                    gain=(norm_err-torch.minimum(norm_err,corrected)).clamp_min(0)
                    observed=count>0
                    if observed.any():
                        (oracle_signed if tlist is oracle_signed else oracle_positive).extend(gain[observed].cpu().tolist())
                ora_count+=len(y)
        report['headroom']={'local_oracle_signed_gain_norm':safe_stat(oracle_signed),
            'local_oracle_positive_gain_norm':safe_stat(oracle_positive),
            'cohort_rows':min(ora_count,64),'warning':'Uses validation outcomes at node/block resolution; optimistic retrospective ceiling, NOT model selection'}
        del model
    else:
        report['headroom']={'status':'not_applicable_normal_only'}
    if wrong_mask_queries>0 and selection!='normal':
        # Recompute historical counterfactual forecasts under a different mask,
        # not a simple CF tensor shuffle, and keep the SAME query/candidate IDs.
        from ..config import load_config
        from ..v5.shared_adapter import initial_model
        from ..evidence import FutureMemory,counterfactual_residuals
        import os
        os.environ['TRAFFIDENT_TARGET']=info['target']
        config=load_config(Path(__file__).resolve().parents[2]/'configs/tcrr_stla_v5.yaml')
        config['experiment'].update(target=info['target'],dataset=info['dataset'],seed=info['seed'])
        config['project']['retrieval_root']=str(Path(config['project']['retrieval_root'])/'v9_2_isolated')
        teacher,_=initial_model(config,Path(info['backbone_checkpoint']),device,Path(info['adapter_checkpoint']))
        teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
        memory=FutureMemory(info['formal_cache'])
        model,_=load_checkpoint_model(run/f'best_{selection}.pt',device)
        correct_errors=[];wrong_errors=[];wrong_query_ids=[];wrong_starts=[]
        subset=DataLoader(Subset(ds,idx[:min(wrong_mask_queries,len(idx))]),batch_size=min(batch_size,4),num_workers=0)
        with torch.inference_mode():
            for raw in subset:
                b=to_device(raw,device)
                baseline=forward_batch(model,b)
                histories=memory.fetch(b['candidate_ids'].long(),device)
                alternate=torch.roll(b['mask'],shifts=max(1,b['mask'].shape[-1]//4),dims=-1)
                alt_res=counterfactual_residuals(teacher,histories,alternate,chunk_size=16,
                                                 strict_source_mask=True)
                alt_cf=(alt_res-b['normal'])*b['valid'][:,:,None,None]
                wrong=forward_batch(model,{**b,'cf':alt_cf})
                y=b['y_norm'].float();fv=b['future_valid'].bool()
                ce=torch.where(fv,(baseline['prediction']-y).abs(),0)
                we=torch.where(fv,(wrong['prediction']-y).abs(),0)
                correction=ce.sum((1,2))/fv.sum((1,2)).clamp_min(1)
                mismatch=we.sum((1,2))/fv.sum((1,2)).clamp_min(1)
                correct_errors.extend(correction.cpu().tolist());wrong_errors.extend(mismatch.cpu().tolist())
                wrong_query_ids.extend(b['query_id'].cpu().tolist())
                wrong_starts.extend(b['window_start'].cpu().tolist())
        delta=np.asarray(wrong_errors)-np.asarray(correct_errors)
        report['wrong_mask_replay']={'cohort_rows':len(wrong_query_ids),'cohort_sha256':hashlib.sha256(
            np.asarray(wrong_query_ids,dtype='<i8').tobytes()).hexdigest(),
            'same_cohort_correct_mae_norm':safe_stat(correct_errors),
            'same_cohort_wrong_mae_norm':safe_stat(wrong_errors),
            'correct_better_gap_norm':safe_stat(delta),
            'gap_ci95_norm':block_bootstrap(delta,np.asarray(wrong_starts)//288,bootstrap,info['seed'])['ci95'],
            'intervention':'node-rolled query failure mask replay; source_valid maintained; same candidate ids'}
        recommendations[:]=[r for r in recommendations if r['module']!='CF Failure Specificity']
        report['recommendations']=recommendations
        gap=report['wrong_mask_replay']['correct_better_gap_norm']
        propose('CF Failure Specificity','behavioral_only',f'Correct-minus-wrong mask replay gap {gap:+.6f} norm; checked same cohort')
        del model,teacher,memory
        if device.type=='cuda':torch.cuda.empty_cache()
    elif wrong_mask_queries>0:
        report['wrong_mask_replay']={'status':'not_applicable_normal_only'}
    else:
        report['not_run'].append('strict_wrong_mask_replay')
    (outdir/'summary.json').write_text(json.dumps(report,indent=2,default=_numpy_json))
    with (outdir/'ablation_results.csv').open('w',newline='') as f:
        fields=['stage','variant','mae_norm','mae_raw','gain_vs_normal_norm','gain_ci95_norm','time_block_count','query_win_rate']
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(records)
    with (outdir/'query_level.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['stage','query_id','missing_fraction','cf_gain_norm'])
        for stage,entries in per_query.items():
            for row in zip(entries['query_ids'],entries['missing_fraction'],entries['cf_gain']):w.writerow([stage,*row])
    # Group by ACTUAL observed missingness, not claimed synthetic severity;
    # official validation view does not have trustworthy s20/s40/s60/s90 tags.
    with (outdir/'failure_groups.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['stage','observed_missing_fraction_bin',
                                            'query_count','mean_cf_gain_norm','query_win_rate'])
        edges=np.asarray([.0,.2,.4,.6,.9,1.01])
        for stage,entries in per_query.items():
            frac=np.asarray(entries['missing_fraction']);gains=np.asarray(entries['cf_gain'])
            for lo,hi in zip(edges[:-1],edges[1:]):
                selected_mask=(frac>=lo)&(frac<hi)
                if selected_mask.any():
                    writer.writerow([stage,f'[{lo:.2f},{hi:.2f})',int(selected_mask.sum()),
                                     float(gains[selected_mask].mean()),float((gains[selected_mask]>0).mean())])
    (outdir/'horizon_analysis.json').write_text(json.dumps({stage:item['cf_gain_by_horizon_norm']
              for stage,item in component_metrics.items()},indent=2))
    (outdir/'recommendations.json').write_text(json.dumps(recommendations,indent=2))
    # HTML contains data summaries only, no external dependencies.
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(record[key]))+'</td>' for key in
       ('stage','variant','mae_norm','gain_vs_normal_norm','gain_ci95_norm'))+'</tr>' for record in records)
    cards=''.join('<li><b>'+html.escape(r['module'])+'</b>: '+html.escape(r['reason'])+'</li>' for r in recommendations)
    page=f'''<!doctype html><html><head><meta charset="utf-8"><title>TCRR v9.2 CF Mechanism Audit</title>
    <style>body{{font:15px Arial,sans-serif;max-width:1240px;margin:32px auto;line-height:1.5}}td,th{{padding:7px 11px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse}}h1,h2{{margin:22px 0 9px}}</style></head><body>
    <h1>TCRR-STLA v9.2 — CF Mechanism Audit</h1><p>Target {html.escape(info['target'])}, dataset {html.escape(info['dataset'])}, selection {html.escape(selection)}.
    <b>These are retrospective validation diagnostics, not prospective prediction results.</b></p>
    <h2>Same-cohort mechanism controls</h2><table><tr><th>Stage</th><th>Variant</th><th>Norm MAE</th><th>Gain vs Normal</th><th>Block-bootstrap 95% CI</th></tr>{rows}</table>
    <h2>Interpretation / next experiments</h2><ul>{cards}</ul><p>See summary.json for Utility correlations, Alignment statistics, source gates and missing controls.</p></body></html>'''
    (outdir/'report.html').write_text(page)
    print('V92_DIAGNOSTIC_PASS '+json.dumps(dict(selected_stage=selection,cf_gain_norm=cf_gain,
       graph_gap=graph_gap,utility_gap=utility_gap,positive_gap=positive_gap,cohort_rows=len(idx))),flush=True)
    return report
