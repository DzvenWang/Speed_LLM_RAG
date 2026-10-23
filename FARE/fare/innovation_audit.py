"""FARE v6.0.1: validation-only, fixed-checkpoint innovation diagnostics.

No optimizer steps, no test labels, no test-set severity tuning.  This module
measures WHAT each frozen intervention changes.  Independent retraining is
still required to establish C1/C2/C3 as publication-level contributions.
"""
from __future__ import annotations
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

from .config import dump_json
from .evaluation import evaluate_modes
from .analysis import write_csv
from .protocol import make_loader, device_batch

# Every extra mode shares the *same* frozen checkpoint and validation windows.
MODES = (
    'full', 'backbone_only', 'support_only',
    'audit_same_pool_random', 'audit_same_pool_similarity', 'audit_static_prior',
    'matched_random', 'state_similarity_only', 'candidate_prior',
    'no_failure', 'shuffled_sensing', 'ordinary_residual',
    'audit_shared_horizon', 'global_selection',
    'no_risk_control', 'no_hidden_injection', 'no_forecast_correction', 'temporal_mean'
)
FIXED_ALPHA_DEFAULT = (0.0, 0.1, 0.25, 0.5, 1.0)


def _assert_validation(ds):
    """Refuse explicit Test/Train split; allow synthetic fixtures without split."""
    source = ds
    while isinstance(source, Subset):
        source = source.dataset
    split = getattr(source, 'split', None)
    if split is not None and split != 'val':
        raise ValueError(f'Innovation Audit requires Validation, got {split!r}')


def _sample_frame(ds):
    """Metadata is used only for sampling; never read target/future arrays."""
    source, indexes = ds, np.arange(len(ds))
    while isinstance(source, Subset):
        indexes = np.asarray(source.indices, dtype=np.int64)[indexes]
        source = source.dataset
    meta = getattr(source, 'meta', None)
    if meta is None or not hasattr(source, 'ix'):
        return None
    frame = meta.iloc[np.asarray(source.ix, dtype=np.int64)[indexes]]
    return frame.reset_index(drop=True)


def _stratified_subset(ds, max_queries, seed):
    """Reproducible, metadata-only severity × family × time-block sampling."""
    _assert_validation(ds)
    count = min(len(ds), int(max_queries))
    if count <= 0:
        raise RuntimeError('Innovation Audit has no Validation queries')
    if count == len(ds):
        return ds
    rng = np.random.default_rng(int(seed))
    frame = _sample_frame(ds)
    if frame is None:
        # Use a different deterministic jitter by seed; avoid repeated fixed
        # equidistant windows and preserve temporal coverage.
        edges = np.linspace(0, len(ds), count + 1, dtype=np.int64)
        indices = [int(rng.integers(a, max(a+1,b))) for a,b in zip(edges[:-1], edges[1:])]
        return Subset(ds, sorted(set(indices)))
    keys=[]
    for key in ('severity','family'):
        if key in frame:
            keys.append(frame[key].fillna('unknown').astype(str).to_numpy())
    if 'window_start' in frame:
        # 288 × 5-minute slots / day. Day groups are for coverage, not labels.
        keys.append((frame.window_start.to_numpy(dtype=np.int64)//288).astype(str))
    groups={}
    for i in range(len(ds)):
        group=tuple(k[i] for k in keys)
        groups.setdefault(group,[]).append(i)
    for members in groups.values():
        rng.shuffle(members)
    order=list(groups)
    rng.shuffle(order)
    chosen=[]
    # Round-robin across family/severity/day strata; if only one severity is
    # present in Validation, record that limitation instead of inventing it.
    while len(chosen)<count:
        progressed=False
        for group in order:
            if groups[group]:
                chosen.append(groups[group].pop())
                progressed=True
                if len(chosen)==count:break
        if not progressed:break
    return Subset(ds,sorted(chosen))


def _sample_errors(trace):
    p=np.asarray(trace['prediction'],np.float64)
    y=np.asarray(trace['target'],np.float64)
    v=np.asarray(trace['valid'],bool)
    if p.shape!=y.shape or p.shape!=v.shape or p.ndim!=3:
        raise ValueError('Invalid aligned prediction/target/valid arrays')
    denom=v.sum((1,2))
    return np.divide((np.abs(p-y)*v).sum((1,2)),np.maximum(denom,1)),denom


def paired_block_stats(a,b,origins,boot=1000,seed=10):
    """Positive means matched B is worse than A. Resample complete day blocks."""
    va,ca=_sample_errors(a);vb,cb=_sample_errors(b)
    if not len(va) or not np.array_equal(a['row_id'],b['row_id']):
        raise ValueError('Paired intervention row IDs differ')
    if not np.array_equal(a['valid'],b['valid']) or not np.array_equal(a['target'],b['target']):
        raise ValueError('Paired intervention targets/validity differ')
    common=(ca>0)&(cb>0)
    diff=(vb-va)[common]
    blocks=(np.asarray(origins)[common]//288).astype(np.int64)
    unique=np.unique(blocks)
    if not len(diff):
        return {'gain':None,'ci95':None,'blocks':0,'query_count':0}
    result={'gain':float(diff.mean()),'ci95':None,'blocks':int(len(unique)),
            'query_count':int(len(diff)), 'unit':'original-target-units'}
    if len(unique)<3:return result
    means=np.asarray([diff[blocks==k].mean() for k in unique]);counts=np.asarray([(blocks==k).sum() for k in unique],float)
    rng=np.random.default_rng(int(seed));boots=int(boot)
    # Do not materialize an enormous bootstrap integer matrix for larger suites.
    vals=[]
    for i in range(0,boots,200):
        samples=rng.integers(0,len(unique),size=(min(200,boots-i),len(unique)))
        vals.extend(((means[samples]*counts[samples]).sum(-1)/counts[samples].sum(-1)).tolist())
    result['ci95']=[float(x) for x in np.quantile(vals,[.025,.975])]
    return result


def _candidate_scores(prepared, batch, count):
    pool=prepared['candidate_pool'];rank=prepared['rank']
    good=pool['valid'][...,:count] & prepared['proposals']['available'][:,:,:count].any((-1,-2))
    learned=rank['score'][...,:count,:].float().mean(-1)
    similar=pool['scores'][...,:count].float()
    # Only historical episode attributes: OOF error and residual coverage.
    cap=pool['candidate_capability'][...,:count,:].float()
    prior=cap[...,7]-0.1*cap[...,3].abs()
    ids=pool['ids'][...,:count].long();row=batch['row_id'].long()[:,None,None]
    node=torch.arange(ids.shape[1],device=ids.device)[None,:,None]
    pseudo=((ids*1103515245+row*12345+node*2654435761)&0x7fffffff).float()/2147483647.0
    return good,{'learned':learned,'similarity':similar,'static_prior':prior,'matched_random':pseudo}


def _nullsafe_top1(scores,available):
    return scores.float().masked_fill(~available,-1e8).argmax(-1)


@torch.no_grad()
def _candidate_probe(model,memory,ds,device,cfg,output=None):
    """Same candidate pool, same adapter, frozen checkpoint and fixed-alpha curves.

    Forced evidence changes all probed nodes in one forward pass.  Gain is
    per-node and per-horizon but may contain spatial interactions with other
    candidate slots; this is NOT an isolated single-node causal intervention.
    """
    ac=cfg.get('innovation_audit',{})
    count=int(ac.get('candidate_queries',8));node_cap=int(ac.get('candidate_nodes',16))
    slots=int(ac.get('candidate_slots',4))
    alphas=tuple(float(x) for x in ac.get('candidate_alphas',FIXED_ALPHA_DEFAULT))
    if any(x<0 or x>1 for x in alphas) or not alphas or len(set(alphas))!=len(alphas):
        raise ValueError('Candidate fixed alphas must be distinct and within [0,1]')
    probe=_stratified_subset(ds,count,int(ac.get('seed',10)))
    totals={name:{'learned':[],'similarity':[],'static_prior':[],'matched_random':[],'oracle_with_null':[]}
            for name in [f'alpha_{a:g}' for a in alphas]+['learned_gate']}
    positive={k:0 for k in totals};harm={k:0 for k in totals}
    recall_hits={name:{pol:{at:0 for at in (1,2,4,8)} for pol in ('learned','similarity','static_prior','matched_random')} for name in totals}
    details=[];used_nodes=0;fwd=0;all_node_ids=set();queried_rows=set()
    model.eval()
    for raw in make_loader(probe,min(2,int(cfg['training']['batch_size'])),workers=0):
        batch=device_batch(raw,device)
        out=model(batch,memory,purge=cfg['memory']['purge_steps'])
        short=out['candidate_pool']
        if short is None:continue
        b,n,k=short['ids'].shape;j=min(k,slots)
        rng=np.random.default_rng(int(ac.get('seed',10))+int(batch['row_id'][0].item()))
        spatial=np.sort(rng.choice(n,min(n,node_cap),replace=False))
        nodes=torch.as_tensor(spatial,device=device,dtype=torch.long)
        good,scores=_candidate_scores(out['prepared'],batch,j)
        good=good.index_select(1,nodes)
        legal=batch['future_valid'].bool().permute(0,2,1).index_select(1,nodes)
        base=out['support_prediction'].permute(0,2,1).index_select(1,nodes)
        target=batch['target'].permute(0,2,1).index_select(1,nodes)
        base_err=(base-target).abs()
        gains={m:[] for m in totals};avail=[]
        for slot in range(j):
            # All candidate IDs/transfer tensors are fixed in the same out.
            pos=torch.full((b,n),slot,dtype=torch.long,device=device)
            solo=model.candidate_single(batch,out['prepared'],pos)
            fwd+=1
            delta=(solo['candidate_prediction']-solo['null_prediction']).permute(0,2,1).index_select(1,nodes)
            selected=solo['selection']['selected_valid'].any((2,3)).index_select(1,nodes)
            valid=legal & selected & good[...,slot,None]
            use=valid.any(-1);avail.append(use)
            denom=valid.sum(-1).clamp_min(1)
            for a in alphas:
                err=(base+a*delta-target).abs()
                gains[f'alpha_{a:g}'].append(((base_err-err)*valid).sum(-1)/denom)
            # Use the *actual learned* risk controller for this forced evidence.
            # No extra STLAformer decode is required.
            risk=model.risk(solo['support'],solo['integrated'],solo['selection'])
            learned_pred=risk['prediction'].permute(0,2,1).index_select(1,nodes)
            gains['learned_gate'].append(((base_err-(learned_pred-target).abs())*valid).sum(-1)/denom)
        if not avail:continue
        V=torch.stack(avail,-1)
        has_any=V.any(-1)
        if not has_any.any():continue
        selected={}
        for pol,score in scores.items():
            selected[pol]=_nullsafe_top1(score.index_select(1,nodes),V)
        identifiers=short['ids'].index_select(1,nodes)[...,:j]
        for name,per_slot in gains.items():
            G=torch.stack(per_slot,-1)
            vmax=G.masked_fill(~V,-1e8).max(-1).values.clamp_min(0)
            relevant=(G>0)&V
            positive[name]+=int(relevant.any(-1)[has_any].sum().item())
            for pol,pol_score in scores.items():
                ranked=pol_score.index_select(1,nodes).masked_fill(~V,-1e8).argsort(-1,descending=True)
                for at in (1,2,4,8):
                    top=ranked[...,:min(at,j)]
                    found=relevant.gather(-1,top).any(-1)&has_any
                    recall_hits[name][pol][at]+=int(found.sum().item())
            for pol,chosen in selected.items():
                policy_gain=G.gather(-1,chosen[...,None]).squeeze(-1)
                totals[name][pol].extend(policy_gain[has_any].cpu().tolist())
                if pol=='learned':harm[name]+=int((policy_gain[has_any]<0).sum().item())
            totals[name]['oracle_with_null'].extend(vmax[has_any].cpu().tolist())
        primary='learned_gate'
        gl=torch.stack(gains[primary],-1)
        primary_oracle=gl.masked_fill(~V,-1e8).max(-1).values.clamp_min(0)
        rows=batch['row_id'].detach().cpu().tolist()
        origins=batch['origin'].detach().cpu().tolist()
        for bi in range(b):
            queried_rows.add(int(rows[bi]))
            for ni,nidx in enumerate(spatial):
                if not bool(has_any[bi,ni]):continue
                all_node_ids.add(int(nidx));used_nodes+=1
                line={'row_id':int(rows[bi]),'origin':int(origins[bi]),'node_id':int(nidx),
                      'positive_any_learned_gate':bool(((gl[bi,ni]>0)&V[bi,ni]).any()),
                      'null_oracle_gain_learned_gate':float(primary_oracle[bi,ni]),
                      'legal_examined_candidates':int(V[bi,ni].sum())}
                for pol,chosen in selected.items():
                    ind=int(chosen[bi,ni])
                    line[f'{pol}_candidate_id']=int(identifiers[bi,ni,ind])
                    line[f'{pol}_gain_learned_gate']=float(gl[bi,ni,ind])
                    for a in alphas:
                        line[f'{pol}_gain_alpha_{a:g}']=float(torch.stack(gains[f'alpha_{a:g}'],-1)[bi,ni,ind])
                details.append(line)
        if len(queried_rows) % max(2,int(ac.get('candidate_progress_every',16))) < b:
            print('FARE6_CANDIDATE_PROGRESS '+json.dumps({'queries':len(queried_rows),
                  'evaluated_nodes':used_nodes,'forward_calls':fwd}),flush=True)
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        if details:
            with (output/'candidate_node_details.csv').open('w',newline='',encoding='utf8') as f:
                w=csv.DictWriter(f,fieldnames=list(details[0]));w.writeheader();w.writerows(details)
    def summary(data):
        out={}
        for key,methods in data.items():
            out[key]={name:(float(np.mean(vals)) if vals else None) for name,vals in methods.items()}
            out[key]['positive_candidate_fraction']=positive[key]/max(1,used_nodes)
            out[key]['positive_examined_recall_at_k']={pol:{f'Recall@{at}':recall_hits[key][pol][at]/max(1,positive[key]) for at in (1,2,4,8)} for pol in recall_hits[key]}
            out[key]['learned_harm_rate']=harm[key]/max(1,used_nodes)
            out[key]['learned_minus_random']=float(np.mean(methods['learned'])-np.mean(methods['matched_random'])) if methods['learned'] else None
            out[key]['learned_minus_similarity']=float(np.mean(methods['learned'])-np.mean(methods['similarity'])) if methods['learned'] else None
        return out
    results=summary(totals)
    candidate_paired={}
    for key in results:
        if not details:
            candidate_paired[key]={'gain':None,'ci95':None,'blocks':0,'query_count':0}
            continue
        # Same query/node paired contrast: +ve means Learned candidate gave
        # more gain (less loss) than Random under identical alpha/gate.
        column_suffix='learned_gate' if key=='learned_gate' else key
        aa=np.asarray([row[f'learned_gain_{column_suffix}']-row[f'matched_random_gain_{column_suffix}'] for row in details])
        oo=np.asarray([row['origin'] for row in details])
        # Reuse block bootstrap with one pseudo observation per evaluated node,
        # then report blocks and node/query counts explicitly.
        uniq=np.unique(oo//288)
        stat={'gain':float(aa.mean()),'ci95':None,'blocks':len(uniq),'query_count':len(queried_rows),
              'evaluated_nodes':len(aa),'unit':'normalized-MAE-gain'}
        if len(uniq)>=3:
            rng=np.random.default_rng(int(ac.get('seed',10)))
            means=np.asarray([aa[(oo//288)==k].mean() for k in uniq])
            counts=np.asarray([(oo//288==k).sum() for k in uniq])
            bs=[]
            for _ in range(int(ac.get('bootstrap_samples',1000))):
                ix=rng.integers(0,len(uniq),size=len(uniq))
                bs.append((means[ix]*counts[ix]).sum()/counts[ix].sum())
            stat['ci95']=[float(z) for z in np.quantile(bs,[.025,.975])]
        candidate_paired[key]=stat
    return {'tested_queries':len(queried_rows),'evaluated_node_queries':used_nodes,
            'distinct_nodes_covered':len(all_node_ids),'tested_candidates_per_node':min(slots, k) if 'k' in locals() else 0,
            'candidate_forward_calls':fwd,'fixed_alphas':alphas,'curves':results,'paired_learned_vs_random':candidate_paired,
            'top1_gain_learned':results['learned_gate']['learned'],
            'top1_gain_similarity':results['learned_gate']['similarity'],
            'top1_gain_matched_random':results['learned_gate']['matched_random'],
            'examined_pool_oracle_gain':results['learned_gate']['oracle_with_null'],
            'fraction_with_positive_examined_candidate':results['learned_gate']['positive_candidate_fraction'],
            'scope':'Validation, fixed checkpoint, examined same shortlist, normalized units, simultaneous spatial candidate-slot intervention; NULL oracle uses max(0, gain)'}


def _signal(stats):
    if stats.get('ci95') is None:return 'INSUFFICIENT_DATA'
    lower,upper=stats['ci95']
    return 'POSITIVE' if lower>0 else ('NEGATIVE' if upper<0 else 'MIXED')


@torch.no_grad()
def audit(model,memory,ds,device,cfg,output,stage=None):
    _assert_validation(ds)
    ac=cfg.get('innovation_audit',{})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    count=int(ac.get('validation_queries',32))
    if stage is not None:count=min(count,int(ac.get('stage_queries',6)))
    subset=_stratified_subset(ds,count,int(ac.get('seed',10)))
    results,traces=evaluate_modes(model,memory,subset,device,cfg,output/'modes',MODES,False,progress_every=int(ac.get('progress_every_batches',4)))
    origins=traces['full']['origin']
    pairs={
        'support_vs_backbone':('support_only','backbone_only'),
        'retrieval_vs_null':('full','support_only'),
        'learned_vs_same_pool_random':('full','audit_same_pool_random'),
        'learned_vs_same_pool_similarity':('full','audit_same_pool_similarity'),
        'learned_vs_static_prior':('full','audit_static_prior'),
        'hidden_injection':('full','no_hidden_injection'),
        'risk_control':('full','no_risk_control'),
        'strict_horizon_selection':('full','audit_shared_horizon'),
        'ordered_temporal':('full','temporal_mean'),
        'forecast_correction':('full','no_forecast_correction'),
        'evidence_failure_delta':('full','no_failure'),
        'ordinary_residual':('full','ordinary_residual'),
        'sensing_identity':('full','shuffled_sensing'),
        # Preserve legacy contrasts while labeling them precisely.
        'legacy_matched_random':('full','matched_random'),
        'legacy_similarity':('full','state_similarity_only'),
        'legacy_global_selection':('full','global_selection'),
    }
    stats={name:paired_block_stats(traces[a],traces[b],origins,
                int(ac.get('bootstrap_samples',1000)),int(ac.get('seed',10)))
           for name,(a,b) in pairs.items()}
    if stage is None:
        candidate=_candidate_probe(model,memory,ds,device,cfg,output)
    elif stage in ('utility','retrieval','joint'):
        stage_cfg={**cfg,'innovation_audit':{**ac,
            'candidate_queries':int(ac.get('stage_candidate_queries',16)),
            'candidate_nodes':int(ac.get('stage_candidate_nodes',16)),
            'candidate_slots':int(ac.get('stage_candidate_slots',4))}}
        candidate=_candidate_probe(model,memory,ds,device,stage_cfg,output)
    else:
        candidate=None
    # Explicit: fixed-checkpoint support-only evidence is NOT C1 independence;
    # similarity/random superiority alone is NOT causal proof of Utility Teacher.
    primary={'C1':'support_vs_backbone','C2':'learned_vs_same_pool_random','C3':'hidden_injection'}
    verdict={key:{'status':'INCONCLUSIVE','fixed_checkpoint_signal':_signal(stats[value]),
                  'observed_gain':stats[value],
                  'reason':'Fixed-weight interventions are diagnostic only. Independent capacity-matched retraining, multiple seeds and networks remain missing.'}
             for key,value in primary.items()}
    rows=[{'mode':name,'val_MAE':r.get('overall_MAE'),'val_RMSE':r.get('overall_RMSE'),
           'queries':len(subset)} for name,r in results.items()]
    write_csv(output/'mode_comparison.csv',rows)
    dump_json(output/'candidate_utility.json',candidate or {'status':'not_run_for_this_stage'})
    dump_json(output/'paired_statistics.json',stats)
    dump_json(output/'innovation_verdict.json',verdict)
    frame=_sample_frame(subset)
    # Model behavior by horizon, severity and failure family. These are
    # descriptive slices of the *same* Validation audit sample, never Test.
    selected_modes=('full','support_only','audit_same_pool_random',
                    'audit_same_pool_similarity','no_risk_control',
                    'no_hidden_injection','audit_shared_horizon')
    horizon_rows=[];strata_rows=[]
    for mode in selected_modes:
        trace=traces[mode]
        pred=np.asarray(trace['prediction']);truth=np.asarray(trace['target']);valid=np.asarray(trace['valid'],bool)
        for horizon in range(pred.shape[1]):
            v=valid[:,horizon,:]
            delta=(pred[:,horizon,:]-truth[:,horizon,:])[v]
            horizon_rows.append({'mode':mode,'horizon':horizon+1,'MAE':float(np.abs(delta).mean()) if delta.size else None,
                                 'RMSE':float(np.sqrt(np.mean(delta**2))) if delta.size else None,'valid_points':int(v.sum())})
        if frame is not None:
            families=frame['family'].fillna('unknown').astype(str).to_numpy() if 'family' in frame else np.full(len(frame),'unknown')
            severities=frame['severity'].fillna('unknown').astype(str).to_numpy() if 'severity' in frame else np.full(len(frame),'unknown')
            categories=sorted(set(zip(severities,families)))
            for severity,family in categories:
                ix=np.where((severities==severity)&(families==family))[0]
                v=valid[ix]; d=(pred[ix]-truth[ix])[v]
                strata_rows.append({'mode':mode,'severity':severity,'failure_family':family,
                    'queries':len(ix),'valid_points':int(v.sum()),
                    'MAE':float(np.abs(d).mean()) if d.size else None,
                    'RMSE':float(np.sqrt(np.mean(d**2))) if d.size else None})
    write_csv(output/'horizon_metrics.csv',horizon_rows)
    if strata_rows:write_csv(output/'severity_family_metrics.csv',strata_rows)
    coverage={}
    if frame is not None:
        for name in ('severity','family'):
            if name in frame:
                coverage[name]={str(k):int(v) for k,v in frame[name].fillna('unknown').astype(str).value_counts().items()}
        if 'window_start' in frame:
            coverage['days']=int((frame.window_start.astype('int64')//288).nunique())
    manifest={'split':'validation_only','stage':stage,'seed':int(ac.get('seed',10)),
              'queries':len(subset),'candidate_queries':int(candidate['tested_queries']) if candidate else 0,
              'coverage':coverage,'memory_provenance':memory.metadata.get('residual_provenance'),
              'single_checkpoint_intervention':True,
              'intervention_definitions':{
                  'audit_same_pool_random':'same shortlist and available evidence, deterministic random selection score; normal learned gate remains',
                  'audit_same_pool_similarity':'same shortlist and evidence, shortlist coarse similarity selection score',
                  'audit_static_prior':'OOF historical candidate residual coverage / absolute OOF error only; excludes query-conditioned transport',
                  'audit_shared_horizon':'one candidate/branch/NULL weight distribution shared across future horizons (only all-horizon valid branches retained)',
                  'candidate_curve':'forced shortlist slot with alpha 0/0.1/0.25/0.5/1 or learned gate; limited simultaneous spatial evidence intervention',
                  'null_oracle':'max(0, best examined gain), diagnostics only'},
              'missing':['Independent C1 failure-aware vs mask-only capacity-matched retraining',
                         'Independent C2 no-counterfactual teacher retraining',
                         'Independent C3 horizon/no-gate retraining','Multiple seeds','Multiple networks',
                         'Validation severity stress suite (if only training s40 is available)'],
              'test_set_used_for_tuning':False}
    dump_json(output/'evaluation_manifest.json',manifest)
    lines=['# FARE v6.0.1 Validation Innovation Audit','',
           'This is a **fixed-checkpoint** mechanism study; it does not establish independent TRC innovations.','',
           '| Component | MAE gain (control − comparison) | 95% day-block CI | signal |',
           '|---|---:|---|---|']
    for key,val in stats.items():
        lines.append(f"| {key} | {val['gain']} | {val['ci95']} | {_signal(val)} |")
    lines += ['', '## Preliminary contribution verdicts', '']
    for key in ('C1','C2','C3'):
        lines.append(f"- {key}: **INCONCLUSIVE**; {_signal(stats[primary[key]])} fixed-checkpoint signal.")
    if candidate is not None:
        lines += ['', '## C2 candidate alpha/gate curves (normalized gain; higher is better)', '',
                  '| injection | learned | similarity | random | null oracle | positive candidate fraction |',
                  '|---|---:|---:|---:|---:|---:|']
        for name,vals in candidate['curves'].items():
            lines.append(f"| {name} | {vals['learned']} | {vals['similarity']} | {vals['matched_random']} | {vals['oracle_with_null']} | {vals['positive_candidate_fraction']} |")
        lines += ['', 'The oracle only covers the examined shortlist and includes the no-evidence action.']
    lines += ['', '## C2 candidate diagnostic', '',
              'See `candidate_utility.json` (alpha–gain curves) and `candidate_node_details.csv` (per-node decisions).',
              'Positive is a reduction in normalized MAE relative to support-only. Forced α=1 is NOT normal inference.',
              'An examined-candidate oracle always includes NULL (gain 0).',
              '', '## Caveats', '',
              '- Candidate slot perturbations are simultaneous across probed nodes; network interactions are possible.',
              '- The strict shared-horizon intervention requires a stream valid for every horizon; this also changes availability.',
              '- Support-only improvement does not independently validate Failure-aware Support; that requires matched retraining.',
              '- Positive fixed-checkpoint signals do not become SUPPORTED without the missing controls in the manifest.']
    (output/'innovation_report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('FARE6_INNOVATION_AUDIT '+json.dumps({'stage':stage,'queries':len(subset),
          'status':{k:v['status'] for k,v in verdict.items()},
          'candidate_nodes':candidate and candidate['evaluated_node_queries']}),flush=True)
    return verdict
