from __future__ import annotations
from collections import deque
import copy, time
import numpy as np
import pandas as pd
import torch

from .metrics import MetricAccumulator, improvement_pct, query_masked_mae
from .model_io import predict_raw, predict_raw_and_key, predict_raw_from_x_batch
from .memory import MatureEvidenceMemory, MatureEvidenceCase
from .retriever import retrieve, prefix_retrieval
from .updater import GuidedUpdater
from .evidence import build_completion, build_patchwise_completion, fuse_predictions
from .oracles import perfect_completion_item, retrieved_completion_oracles, retrieved_entry_oracles_by_k
from .utility import (OnlineUtilityScorer, prepare_candidate_pool, build_patch_candidate_tables,
                      select_patch_supports, sample_probe_specs, build_probe_augmented_histories,
                      FEATURE_NAMES)
from .utils import log, write_json, sha256_file


def _safe_gain(a,b):
    return improvement_pct(a,b) if np.isfinite(a) and np.isfinite(b) else float('nan')


def _metric_keys():
    return ['overall_mae','masked_mae','support_mae','overall_rmse','masked_rmse','support_rmse']


def _mean_or_nan(x):
    x=np.asarray(x,dtype=np.float64); x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


def _retrieval_summary(retrieved):
    if retrieved is None:
        return {'retrieved_k':0,'mean_latent_sim':float('nan'),'mean_observed_match':float('nan'),
                'mean_complementarity':float('nan'),'mean_age':float('nan')}
    return {'retrieved_k':len(retrieved['case_index']),'mean_latent_sim':_mean_or_nan(retrieved['latent_sim']),
            'mean_observed_match':_mean_or_nan(retrieved['observed_match']),
            'mean_complementarity':_mean_or_nan(retrieved['complementarity']),'mean_age':_mean_or_nan(retrieved['age'])}


def _label_name(y):
    return ['harmful','neutral','helpful'][int(y)]


def run_stream(cfg, ds, base_model, ck_path, ck, source, variant, out_dir, device, max_queries=None):
    acfg=cfg['adaptation']; mcfg=cfg['evidence_memory']; rcfg=cfg['retrieval']; compcfg=cfg.get('compositional',{})
    ucfg=cfg.get('utility',{}); ccfg=dict(cfg['completion']); ccfg['score_temperature']=rcfg['score_temperature']
    fcfg=cfg['fusion']; dcfg=cfg.get('diagnostics',{})
    mean,std=float(ck['mean']),float(ck['std'])
    oracle_ks=sorted({int(k) for k in dcfg.get('entry_oracle_ks',[8,16,32,64]) if int(k)>0})
    oracle_pool_k=max(oracle_ks) if oracle_ks else int(rcfg['top_k'])

    frozen=copy.deepcopy(base_model).to(device).eval()
    slow=copy.deepcopy(base_model).to(device).eval()
    slow_updater=GuidedUpdater(slow,ds,device,acfg,seed=cfg['experiment']['seed'])
    memory=MatureEvidenceMemory(int(mcfg['capacity']))
    utility_scorer=OnlineUtilityScorer(ucfg,device,seed=cfg['experiment']['seed']) if variant=='rag_utility_compositional' else None
    pending=deque(); rng=np.random.default_rng(int(cfg['experiment']['seed'])+911)

    base_acc=MetricAccumulator(cfg['experiment']['horizon'])
    slow_acc=MetricAccumulator(cfg['experiment']['horizon'])
    completion_acc=MetricAccumulator(cfg['experiment']['horizon'])
    final_acc=MetricAccumulator(cfg['experiment']['horizon'])
    perfect_oracle_acc=MetricAccumulator(cfg['experiment']['horizon'])
    case_oracle_acc=MetricAccumulator(cfg['experiment']['horizon'])
    entry_oracle_acc={k:MetricAccumulator(cfg['experiment']['horizon']) for k in oracle_ks}
    probe_slow_acc=MetricAccumulator(cfg['experiment']['horizon'])
    sampled_patch_utility_oracle_acc=MetricAccumulator(cfg['experiment']['horizon'])

    stream_rows=[]; update_rows=[]; retrieval_rows=[]; patch_support_rows=[]; memory_rows=[]; evidence_rows=[]
    utility_label_rows=[]; utility_update_rows=[]
    recon_abs=0.0; recon_n=0; placeholder_abs=0.0; placeholder_n=0; total_target=0; total_filled=0
    case_recon=[]; oracle_recon={k:[] for k in oracle_ks}; final_help=[]; perfect_query_gains=[]; oracle_query_gains={k:[] for k in oracle_ks}
    utility_best_probe_gains=[]; utility_ready_flags=[]; utility_selected_help=[]; utility_selected_harm=[]
    n=len(ds) if max_queries is None else min(len(ds),int(max_queries))
    t0=time.time(); update_every=max(1,int(acfg['update_every_queries']))

    evidence_variants={'rag_similarity_global','rag_similarity_compositional','rag_utility_compositional'}
    patch_variants={'rag_similarity_compositional','rag_utility_compositional'}

    for pos in range(n):
        idx=ds.cache_index(pos); item=ds.item(idx); origin=int(item['origin']); matured_now=0; labels_matured_now=0

        # Strict causal maturation: labels for utility probes become available only here.
        while pending and int(pending[0]['future_end']) <= origin:
            oldp=pending.popleft(); old_idx=int(oldp['index']); old=ds.item(old_idx)
            memory.add(MatureEvidenceCase(
                index=old_idx,origin=int(oldp['origin']),key=oldp['key'].astype(np.float32),tod=int(oldp['tod']),dow=int(oldp['dow']),
                x_obs=old['x_obs'].numpy().astype(np.float32),history_observed=old['history_observed'].numpy().astype(bool),
                y_raw=old['y_raw'].numpy().astype(np.float32),future_valid=old['future_valid'].numpy().astype(bool),
            ))
            memory_rows.append({'stream_pos':pos,'case_index':old_idx,'case_origin':int(oldp['origin']),
                                'future_end':int(oldp['future_end']),'memory_size':len(memory)})
            matured_now+=1

            probe=oldp.get('utility_probe')
            if utility_scorer is not None and probe is not None and len(probe.get('predictions',[])):
                oy=old['y_raw'].numpy(); ov=old['future_valid'].numpy(); od=old['direct_mask'].numpy(); os=old['support_mask'].numpy()
                ref=np.asarray(probe['slow_pred'],np.float32); ref_mae=query_masked_mae(ref,oy,ov,od)
                preds=np.asarray(probe['predictions'],np.float32); feats=np.asarray(probe['features'],np.float32)
                utils=[]; maes=[]
                for j,pred in enumerate(preds):
                    cm=query_masked_mae(pred,oy,ov,od); maes.append(cm); u=_safe_gain(ref_mae,cm); utils.append(u)
                    lab=utility_scorer.utility_to_label(u)
                    meta=probe['meta'][j]
                    utility_label_rows.append({'mature_stream_pos':pos,'query_index':old_idx,'query_origin':int(oldp['origin']),
                        'node':int(meta['node']),'patch_id':int(meta['patch_id']),'patch_start':int(meta['start']),'patch_end':int(meta['end']),
                        'case_index':int(meta['case_index']),'pool_rank':int(meta['pool_rank']),'utility_pct':float(u),'label':int(lab),'label_name':_label_name(lab)})
                utility_scorer.add(feats,np.asarray(utils,np.float32)); labels_matured_now+=len(utils)
                train_loss=utility_scorer.maybe_train()
                if train_loss is not None:
                    utility_update_rows.append({'stream_pos':pos,'examples':len(utility_scorer),'class_counts':str(utility_scorer.class_counts()),
                        'train_loss':float(train_loss),'train_updates':utility_scorer.train_updates,'train_steps':utility_scorer.train_steps})
                # Honest sampled single-patch downstream oracle over probes generated at query time.
                probe_slow_acc.update(ref,oy,ov,od,os)
                best_pred=ref; best_mae=ref_mae
                for j,cm in enumerate(maes):
                    if np.isfinite(cm) and cm<best_mae: best_mae=cm; best_pred=preds[j]
                sampled_patch_utility_oracle_acc.update(best_pred,oy,ov,od,os); utility_best_probe_gains.append(_safe_gain(ref_mae,best_mae))

        # Frozen reference key/prediction.
        base_pred,qkey=predict_raw_and_key(frozen,item,device,mean,std)

        # Slow persistent online learning is unchanged.
        slow_info={'updated':False,'loss':float('nan'),'steps':0,'param_delta_l2':0.0,'batch_indices':[]}
        if variant!='frozen' and pos%update_every==0 and len(memory)>=int(acfg['min_memory_cases']):
            ids=memory.recent_indices(int(acfg['replay_memory_size']))
            slow_info=slow_updater.update(ids,None,mode='random_replay')
            if slow_info['updated']:
                update_rows.append({'stream_pos':pos,'current_index':idx,'origin':origin,'memory_size':len(memory),
                    'candidate_pool_size':len(ids),'update_loss_norm':slow_info['loss'],'steps':slow_info['steps'],
                    'param_delta_l2':slow_info['param_delta_l2'],'batch_indices':' '.join(map(str,slow_info['batch_indices']))})
        slow_pred=base_pred if variant=='frozen' else predict_raw(slow,item,device,mean,std)

        # ---- Real evidence path: visible current input + matured memory only. ----
        pool=None; prepared=None; tables=None; completion=None; completion_pred=slow_pred.copy(); final_pred=slow_pred.copy(); probe_payload=None
        gc=np.zeros_like(slow_pred,np.float32); scorer_ready=False; abstained_patches=0; selected_patches=0
        if variant in evidence_variants and len(memory)>=int(mcfg['min_cases']):
            qx=item['x_obs'].numpy(); qobs=item['history_observed'].numpy(); qmissing=item['artificial_missing'].numpy(); rows=memory.rows()
            pool=retrieve(memory,qkey,origin,int(item['tod_hist'][-1]),int(item['dow_hist'][-1]),qx,qobs,qmissing,rcfg,
                          top_k=max(int(compcfg.get('candidate_pool_k',oracle_pool_k)),oracle_pool_k))
            if pool is not None and len(pool['case_index']):
                if variant=='rag_similarity_global':
                    completion=build_completion(item,rows,prefix_retrieval(pool,int(rcfg['top_k'])),ccfg)
                else:
                    prepared=prepare_candidate_pool(rows,pool,qx,qobs,qmissing,ccfg)
                    tables=build_patch_candidate_tables(prepared,ucfg,rcfg)
                    mode='utility' if variant=='rag_utility_compositional' else 'similarity'
                    selected=select_patch_supports(prepared,tables,utility_scorer,ucfg,ccfg,mode=mode)
                    scorer_ready=bool(selected.get('scorer_ready',False)); selected_patches=len(selected.get('patches',{})); abstained_patches=max(len(tables)-selected_patches,0)
                    completion=build_patchwise_completion(item,prepared,selected,ccfg)
                    max_log=int(dcfg.get('patch_support_log_max_per_query',64))
                    for meta in completion.get('support_meta',[])[:max_log]:
                        patch_support_rows.append({'stream_pos':pos,'query_index':idx,'origin':origin,'variant':variant,'scorer_ready':scorer_ready,**meta})

                    # Generate strictly causal utility probes now; labels are attached only when this query matures.
                    if variant=='rag_utility_compositional' and tables:
                        specs=sample_probe_specs(tables,ucfg,rng)
                        pb=build_probe_augmented_histories(item,prepared,tables,specs)
                        if pb is not None:
                            pp=predict_raw_from_x_batch(slow,item,pb['x_batch'],device,mean,std)
                            probe_payload={'features':pb['features'],'predictions':pp,'meta':pb['meta'],'slow_pred':slow_pred.copy()}

                if completion is not None:
                    item_aug=dict(item); item_aug['x_obs']=torch.from_numpy(completion['x_aug'])
                    if np.any(completion['fill_mask']): completion_pred=predict_raw(slow,item_aug,device,mean,std)
                    final_pred,gc,_=fuse_predictions(slow_pred,completion_pred,None,completion,item,ccfg,{'max_gate':0.0},fcfg,
                                                     use_completion=True,use_transition=False,completion_multiplier=1.0)

        # ---- Hidden current truth/future only below this line. ----
        y=item['y_raw'].numpy(); valid=item['future_valid'].numpy(); direct=item['direct_mask'].numpy(); support=item['support_mask'].numpy()
        base_acc.update(base_pred,y,valid,direct,support); slow_acc.update(slow_pred,y,valid,direct,support)
        completion_acc.update(completion_pred,y,valid,direct,support); final_acc.update(final_pred,y,valid,direct,support)
        bm=query_masked_mae(base_pred,y,valid,direct); sm=query_masked_mae(slow_pred,y,valid,direct); fm=query_masked_mae(final_pred,y,valid,direct)
        if completion is not None: final_help.append(1.0 if np.isfinite(sm) and np.isfinite(fm) and fm<sm else 0.0)

        target=item['artificial_missing'].numpy().astype(bool)&np.isfinite(item['x_full'].numpy())
        total_target+=int(target.sum()); fill_count=0; comp_cov=0.0; comp_recon=float('nan'); placeholder=float('nan')
        if completion is not None:
            fill=np.asarray(completion['fill_mask'],bool)&target; fill_count=int(fill.sum()); total_filled+=fill_count
            comp_cov=float(completion['coverage']); xf=item['x_full'].numpy(); xo=item['x_obs'].numpy(); xa=completion['x_aug']
            if fill_count:
                ae=np.abs(xa[fill]-xf[fill]); recon_abs+=float(ae.sum()); recon_n+=fill_count; comp_recon=float(ae.mean())
            if np.any(target):
                pae=np.abs(xo[target]-xf[target]); placeholder_abs+=float(pae.sum()); placeholder_n+=int(target.sum()); placeholder=float(pae.mean())

        perfect_pred=slow_pred.copy(); perfect_mae=sm
        if bool(dcfg.get('perfect_completion_oracle',True)) and np.any(target):
            pitem,_=perfect_completion_item(item); perfect_pred=predict_raw(slow,pitem,device,mean,std); perfect_mae=query_masked_mae(perfect_pred,y,valid,direct)
        perfect_oracle_acc.update(perfect_pred,y,valid,direct,support); perfect_query_gains.append(_safe_gain(sm,perfect_mae))

        case_pred=slow_pred.copy(); case_mae=sm; case_recon_mae=float('nan')
        oracle_preds={k:slow_pred.copy() for k in oracle_ks}; oracle_maes={k:sm for k in oracle_ks}; oracle_recons={k:float('nan') for k in oracle_ks}
        if bool(dcfg.get('retrieved_completion_oracle',True)) and pool is not None and len(pool['case_index']):
            pool_completion=build_completion(item,memory.rows(),pool,ccfg)
            ro_case=retrieved_completion_oracles(item,build_completion(item,memory.rows(),prefix_retrieval(pool,int(rcfg['top_k'])),ccfg))
            if ro_case is not None:
                case_recon_mae=ro_case['case_recon_mae_norm']; case_recon.append(case_recon_mae)
                if ro_case['case_fill_count']>0:
                    ci=dict(item); ci['x_obs']=torch.from_numpy(ro_case['case_x_aug']); case_pred=predict_raw(slow,ci,device,mean,std); case_mae=query_masked_mae(case_pred,y,valid,direct)
            byk=retrieved_entry_oracles_by_k(item,pool_completion,oracle_ks)
            for k in oracle_ks:
                od=byk.get(k)
                if od is not None:
                    oracle_recons[k]=od['recon_mae_norm']; oracle_recon[k].append(od['recon_mae_norm'])
                    if od['fill_count']>0:
                        oi=dict(item); oi['x_obs']=torch.from_numpy(od['x_aug']); op=predict_raw(slow,oi,device,mean,std)
                        oracle_preds[k]=op; oracle_maes[k]=query_masked_mae(op,y,valid,direct)
        case_oracle_acc.update(case_pred,y,valid,direct,support)
        for k in oracle_ks:
            entry_oracle_acc[k].update(oracle_preds[k],y,valid,direct,support); oracle_query_gains[k].append(_safe_gain(sm,oracle_maes[k]))

        rsummary=_retrieval_summary(prefix_retrieval(pool,int(rcfg['top_k'])) if pool is not None else None)
        if pool is not None:
            logk=min(len(pool['case_index']),int(dcfg.get('retrieval_log_top_k',8)))
            for rank in range(logk):
                retrieval_rows.append({'stream_pos':pos,'query_index':idx,'rank':rank+1,'case_index':int(pool['case_index'][rank]),
                    'score':float(pool['score'][rank]),'latent_sim':float(pool['latent_sim'][rank]),'observed_match':float(pool['observed_match'][rank]),
                    'overlap_fraction':float(pool['overlap_fraction'][rank]),'complementarity':float(pool['complementarity'][rank]),
                    'recency':float(pool['recency'][rank]),'context':float(pool['context'][rank]),'age':float(pool['age'][rank])})

        utility_examples=len(utility_scorer) if utility_scorer is not None else 0
        utility_counts=utility_scorer.class_counts() if utility_scorer is not None else [0,0,0]
        utility_ready_flags.append(float(scorer_ready)) if variant=='rag_utility_compositional' else None
        if completion is not None and variant=='rag_utility_compositional':
            ph=[x for x in completion.get('support_meta',[]) if np.isfinite(x.get('p_helpful',np.nan))]
            if ph:
                utility_selected_help.extend([float(x['p_helpful']) for x in ph]); utility_selected_harm.extend([float(x['p_harmful']) for x in ph])

        ev={'stream_pos':pos,'query_index':idx,'variant':variant,'memory_size':len(memory),'missing_target_entries':int(target.sum()),
            'filled_entries':fill_count,'completion_coverage':comp_cov,'completion_recon_mae_norm':comp_recon,'placeholder_mae_norm':placeholder,
            'mean_completion_gate':float(np.mean(gc)),'max_completion_gate':float(np.max(gc)),
            'slow_masked_mae':sm,'final_masked_mae':fm,'rag_increment_vs_slow_pct':_safe_gain(sm,fm),
            'perfect_completion_oracle_masked_mae':perfect_mae,'perfect_completion_oracle_gain_pct':_safe_gain(sm,perfect_mae),
            'retrieved_case_oracle_masked_mae':case_mae,'retrieved_case_oracle_gain_pct':_safe_gain(sm,case_mae),
            'retrieved_case_oracle_recon_mae_norm':case_recon_mae,'patch_units_available':0 if tables is None else int(len(tables)),
            'patch_units_selected':int(selected_patches),'patch_units_abstained':int(abstained_patches),'utility_scorer_ready':bool(scorer_ready),
            'utility_examples':int(utility_examples),'utility_harmful_examples':int(utility_counts[0]),'utility_neutral_examples':int(utility_counts[1]),
            'utility_helpful_examples':int(utility_counts[2]),'utility_labels_matured_now':int(labels_matured_now),**rsummary}
        for k in oracle_ks:
            ev[f'retrieved_entry_oracle_at{k}_masked_mae']=oracle_maes[k]; ev[f'retrieved_entry_oracle_at{k}_gain_pct']=_safe_gain(sm,oracle_maes[k]); ev[f'retrieved_entry_oracle_at{k}_recon_mae_norm']=oracle_recons[k]
        evidence_rows.append(ev)

        sr={'stream_pos':pos,'cache_index':idx,'window_start':int(item['start']),'origin':origin,'future_end':int(item['future_end']),
            'matured_now':matured_now,'memory_size':len(memory),'slow_did_update':bool(slow_info['updated']),
            'base_masked_mae':bm,'slow_masked_mae':sm,'completion_masked_mae':query_masked_mae(completion_pred,y,valid,direct),
            'final_masked_mae':fm,'rag_delta_vs_slow':fm-sm,'perfect_oracle_masked_mae':perfect_mae,
            'retrieved_case_oracle_masked_mae':case_mae,'utility_scorer_ready':bool(scorer_ready),'utility_examples':int(utility_examples),**rsummary}
        for k in oracle_ks: sr[f'retrieved_entry_oracle_at{k}_masked_mae']=oracle_maes[k]
        stream_rows.append(sr)

        pending.append({'future_end':int(item['future_end']),'index':idx,'origin':origin,'key':qkey,
                        'tod':int(item['tod_hist'][-1]),'dow':int(item['dow_hist'][-1]),'utility_probe':probe_payload})

        if (pos+1)%100==0 or pos==n-1:
            bs=base_acc.summary()['masked_mae']; ss=slow_acc.summary()['masked_mae']; fs=final_acc.summary()['masked_mae']; ps=perfect_oracle_acc.summary()['masked_mae']
            oracle_msg=' '.join([f"entry@{k}={_safe_gain(ss,entry_oracle_acc[k].summary()['masked_mae']):+.3f}%" for k in oracle_ks])
            util_msg=''
            if utility_scorer is not None:
                util_msg=f" util_ready={utility_scorer.ready} util_n={len(utility_scorer)} util_classes={utility_scorer.class_counts()} util_loss={utility_scorer.last_loss:.4f}"
            log(f"STREAM {pos+1}/{n} variant={variant} memory={len(memory)} base={bs:.6f} slow={ss:.6f} final={fs:.6f} "
                f"slow_gain={_safe_gain(bs,ss):+.3f}% final_gain={_safe_gain(bs,fs):+.3f}% rag_vs_slow={_safe_gain(ss,fs):+.3f}% "
                f"perfect_oracle={_safe_gain(ss,ps):+.3f}% {oracle_msg}{util_msg}")

    sdf=pd.DataFrame(stream_rows); udf=pd.DataFrame(update_rows); rdf=pd.DataFrame(retrieval_rows); psdf=pd.DataFrame(patch_support_rows); mdf=pd.DataFrame(memory_rows); edf=pd.DataFrame(evidence_rows)
    uldf=pd.DataFrame(utility_label_rows); uudf=pd.DataFrame(utility_update_rows)
    sdf.to_csv(out_dir/'stream_metrics.csv',index=False); udf.to_csv(out_dir/'slow_updates.csv',index=False); rdf.to_csv(out_dir/'retrievals.csv',index=False)
    psdf.to_csv(out_dir/'patch_supports.csv',index=False); mdf.to_csv(out_dir/'memory_events.csv',index=False); edf.to_csv(out_dir/'evidence_metrics.csv',index=False)
    uldf.to_csv(out_dir/'utility_labels.csv',index=False); uudf.to_csv(out_dir/'utility_updates.csv',index=False)

    base=base_acc.summary(); slow_sum=slow_acc.summary(); comp_sum=completion_acc.summary(); final=final_acc.summary(); perfect=perfect_oracle_acc.summary(); caseo=case_oracle_acc.summary()
    entry_summaries={str(k):entry_oracle_acc[k].summary() for k in oracle_ks}
    slow_gains={k:_safe_gain(base[k],slow_sum[k]) for k in _metric_keys()}; final_gains={k:_safe_gain(base[k],final[k]) for k in _metric_keys()}
    rag_inc={k:_safe_gain(slow_sum[k],final[k]) for k in _metric_keys()}; perfect_inc={k:_safe_gain(slow_sum[k],perfect[k]) for k in _metric_keys()}
    case_inc={k:_safe_gain(slow_sum[k],caseo[k]) for k in _metric_keys()}
    entry_inc={str(kk):{k:_safe_gain(slow_sum[k],entry_summaries[str(kk)][k]) for k in _metric_keys()} for kk in oracle_ks}
    probe_slow=probe_slow_acc.summary(); probe_oracle=sampled_patch_utility_oracle_acc.summary()
    probe_oracle_inc={k:_safe_gain(probe_slow[k],probe_oracle[k]) for k in _metric_keys()} if probe_slow.get('masked_count',0)>0 else {k:float('nan') for k in _metric_keys()}

    segments=[]
    if len(sdf):
        cuts=np.linspace(0,len(sdf),6,dtype=int)
        for j in range(5):
            z=sdf.iloc[cuts[j]:cuts[j+1]]; b=float(z.base_masked_mae.mean()); s=float(z.slow_masked_mae.mean()); f=float(z.final_masked_mae.mean()); p=float(z.perfect_oracle_masked_mae.mean())
            row={'segment':j+1,'start_pos':int(cuts[j]),'end_pos':int(cuts[j+1]),'base_query_masked_mae':b,'slow_query_masked_mae':s,
                 'final_query_masked_mae':f,'slow_gain_pct':_safe_gain(b,s),'final_gain_pct':_safe_gain(b,f),'rag_increment_vs_slow_pct':_safe_gain(s,f),
                 'perfect_completion_oracle_vs_slow_pct':_safe_gain(s,p)}
            for k in oracle_ks:
                e=float(z[f'retrieved_entry_oracle_at{k}_masked_mae'].mean()); row[f'retrieved_entry_oracle_at{k}_vs_slow_pct']=_safe_gain(s,e)
            segments.append(row)

    utility_summary={
        'enabled':utility_scorer is not None,'feature_names':FEATURE_NAMES,
        'examples':len(utility_scorer) if utility_scorer is not None else 0,
        'class_counts':utility_scorer.class_counts() if utility_scorer is not None else [0,0,0],
        'train_updates':utility_scorer.train_updates if utility_scorer is not None else 0,
        'train_steps':utility_scorer.train_steps if utility_scorer is not None else 0,
        'last_train_loss':utility_scorer.last_loss if utility_scorer is not None else float('nan'),
        'ready_fraction':_mean_or_nan(utility_ready_flags),'mean_selected_p_helpful':_mean_or_nan(utility_selected_help),
        'mean_selected_p_harmful':_mean_or_nan(utility_selected_harm),'mean_best_sampled_single_patch_utility_pct':_mean_or_nan(utility_best_probe_gains),
        'sampled_single_patch_oracle_vs_probe_slow_pct':probe_oracle_inc,'probe_queries':int(probe_slow.get('masked_count',0)>0 and len(utility_best_probe_gains) or 0),
    }

    summary={
        'version':cfg['version'],'definition':'Online PreSuRe v3.3: slow online PatchSTG + node×temporal-patch complementary candidate discovery + causal online predictive-utility scoring + abstaining selective completion',
        'variant':variant,'stage':ds.stage,'split':ds.split,'queries':n,'history':ds.history,'horizon':ds.horizon,
        'causal_rule':'Current-query prediction uses only x_obs/masks/context, frozen retrieval keys, mature memory, and utility labels from already-matured queries. Probe forecasts are generated before labels exist; their downstream utility is attached only after the complete future matures.',
        'base_checkpoint':str(ck_path),'base_checkpoint_sha256':sha256_file(ck_path),'base_checkpoint_source':source,
        'retrieval_key_backbone':'frozen PatchSTG pre-head pooled key','adaptation_config':acfg,'evidence_memory_config':mcfg,'retrieval_config':rcfg,
        'compositional_config':compcfg,'utility_config':ucfg,'completion_config':cfg['completion'],'fusion_config':fcfg,'diagnostics_config':dcfg,
        'slow_updates':slow_updater.update_count,'slow_optimizer_steps':slow_updater.step_count,'final_memory_size':len(memory),
        'base':base,'slow_online':slow_sum,'completion_forecast':comp_sum,'online_presure_v33':final,
        'perfect_completion_oracle':perfect,'retrieved_case_completion_oracle_top8':caseo,'retrieved_entry_completion_oracles':entry_summaries,
        'slow_gains_pct':slow_gains,'final_gains_pct':final_gains,'rag_increment_vs_slow_pct':rag_inc,
        'perfect_completion_oracle_vs_slow_pct':perfect_inc,'retrieved_case_oracle_vs_slow_pct':case_inc,
        'retrieved_entry_oracle_by_k_vs_slow_pct':entry_inc,'utility':utility_summary,
        'missing_completion':{'target_entries':int(total_target),'filled_entries':int(total_filled),'fill_coverage':float(total_filled/max(total_target,1)),
            'filled_reconstruction_mae_norm':float(recon_abs/max(recon_n,1)) if recon_n else float('nan'),
            'missing_placeholder_mae_norm':float(placeholder_abs/max(placeholder_n,1)) if placeholder_n else float('nan'),
            'retrieved_case_oracle_recon_mae_norm':_mean_or_nan(case_recon),
            'retrieved_entry_oracle_recon_mae_norm_by_k':{str(k):_mean_or_nan(oracle_recon[k]) for k in oracle_ks}},
        'query_level':{'final_helpful_fraction_vs_slow':_mean_or_nan(final_help),'mean_perfect_oracle_gain_pct':_mean_or_nan(perfect_query_gains),
            'mean_retrieved_entry_oracle_gain_pct_by_k':{str(k):_mean_or_nan(oracle_query_gains[k]) for k in oracle_ks}},
        'segments':segments,'elapsed_seconds':time.time()-t0,
    }
    if bool(acfg.get('save_final_checkpoint',True)) and variant!='frozen':
        torch.save({'format':'online_presure_v33_slow_state','version':cfg['version'],'variant':variant,'state_dict':slow.state_dict(),
            'base_checkpoint':str(ck_path),'base_checkpoint_sha256':sha256_file(ck_path),'stream_queries':n,'slow_updates':slow_updater.update_count,
            'adaptation_config':acfg},out_dir/'final_slow_online.pt')
    if utility_scorer is not None:
        torch.save({'format':'online_presure_v33_utility_scorer','version':cfg['version'],'feature_names':FEATURE_NAMES,
                    'state_dict':utility_scorer.model.state_dict(),'examples':len(utility_scorer),'class_counts':utility_scorer.class_counts(),
                    'config':ucfg},out_dir/'final_utility_scorer.pt')
    write_json(out_dir/'summary.json',summary)
    return summary
