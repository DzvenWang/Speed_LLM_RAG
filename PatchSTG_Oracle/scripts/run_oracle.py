#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from oracle.config import load_config
from oracle.utils import write_json,seed_all,deterministic_split,sha256_file,log
from oracle.contract import assert_cache_ready,cache_dir
from oracle.backbone import load_patchstg_adapter
from oracle.data import OracleValDataset,audit_canonical_source
from oracle.deep import make_inputs,forward_with_deep_states,decode_prehead
from oracle.metrics import MetricAccumulator,improvement_pct,per_query_masked_mae,latent_query_stats


def part_flags(indices,dev_set,final_set):
    ii=np.asarray(indices,np.int64)
    return {'all':np.ones(len(ii),bool),'dev':np.asarray([int(x) in dev_set for x in ii],bool),'final':np.asarray([int(x) in final_set for x in ii],bool)}


def group_flags(ratios,cfg):
    r=np.asarray(ratios,np.float64); out={}
    for name,bounds in cfg['oracle']['missing_groups'].items():
        lo,hi=float(bounds[0]),float(bounds[1])
        if name=='none': out[name]=r==0
        else: out[name]=(r>lo)&(r<=hi)
    return out


def corr(a,b,rank=False):
    x=pd.Series(np.asarray(a,np.float64)); y=pd.Series(np.asarray(b,np.float64)); m=x.notna()&y.notna()
    if m.sum()<3: return float('nan')
    if rank: x=x.rank(); y=y.rank()
    return float(x[m].corr(y[m]))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--run-tag',required=True); ap.add_argument('--checkpoint',default=None); a=ap.parse_args()
    cfg=load_config(ROOT); seed_all(cfg['experiment']['seed']); ident=assert_cache_ready(cfg,a.stage); cache=cache_dir(cfg,a.stage)
    out=Path(cfg['project']['exp_root'])/cfg['experiment']['subgraph']/a.stage/a.run_tag; out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); adapter,ckpath,ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint)
    ds=OracleValDataset(cfg,cache,float(ck['mean']),float(ck['std'])); n=len(ds)
    dev_ids,final_ids=deterministic_split(n,cfg['experiment']['dev_queries'],cfg['experiment']['final_val_queries'],cfg['experiment']['seed']); dev_set=set(dev_ids.tolist()); final_set=set(final_ids.tolist())
    audit=audit_canonical_source(cfg,cache,float(ck['mean']),float(ck['std']))
    log(f'COMPLETE_HISTORY_SOURCE {json.dumps(audit,sort_keys=True)}')
    rt=cfg['runtime']; kw={'batch_size':int(rt['batch_size']),'shuffle':False,'drop_last':False,'num_workers':int(rt['num_workers']),'pin_memory':bool(rt['pin_memory']) and torch.cuda.is_available()}
    if kw['num_workers']>0: kw.update(prefetch_factor=int(rt['prefetch_factor']),persistent_workers=bool(rt['persistent_workers']))
    dl=DataLoader(ds,**kw)
    fractions=[float(x) for x in cfg['oracle']['input_restore_fractions']]; variant_names=['incomplete']+[f'input_oracle_{int(round(x*100)):03d}' for x in fractions]
    parts=['all','dev','final']; metrics={v:{p:MetricAccumulator(cfg['experiment']['horizon']) for p in parts} for v in variant_names}
    group_metrics={v:{g:MetricAccumulator(cfg['experiment']['horizon']) for g in cfg['oracle']['missing_groups']} for v in ['incomplete','input_oracle_100']}
    qrows=[]; first=True; t0=time.time()
    for bi,b in enumerate(dl,1):
        idx=b['index'].numpy(); flags=part_flags(idx,dev_set,final_set); gf=group_flags(b['missing_ratio'].numpy(),cfg)
        x0=b['x_obs'].to(device,non_blocking=True).float(); x1=b['x_complete'].to(device,non_blocking=True).float(); tod=b['tod_hist']; dow=b['dow_hist']
        tx0,te=make_inputs(x0,tod,dow,device); tx1,_=make_inputs(x1,tod,dow,device)
        with torch.no_grad():
            pn0,s0=forward_with_deep_states(adapter.model,tx0,te,cfg['oracle']['capture_layers'])
            pn1,s1=forward_with_deep_states(adapter.model,tx1,te,cfg['oracle']['capture_layers'])
            if first:
                official=adapter.eval_forward({'x':b['x_obs'],'tod_hist':tod,'dow_hist':dow})
                d1=float(torch.max(torch.abs(official-pn0)).item()); recon=decode_prehead(adapter.model,s0['prehead_4d']); d2=float(torch.max(torch.abs(recon-pn0)).item())
                atol=float(cfg['canonical_history'].get('deep_equivalence_atol',1e-5)) if 'deep_equivalence_atol' in cfg['canonical_history'] else 1e-5
                log(f'DEEP_EQUIVALENCE official_vs_interface_max_abs_diff={d1:.9g} prehead_reconstruction_max_abs_diff={d2:.9g}')
                if d1>atol or d2>atol: raise RuntimeError('PatchSTG deep interface equivalence failed')
                log('PATCHSTG DEEP INTERFACE EQUIVALENCE PASS'); first=False
            preds={'incomplete':pn0*float(ck['std'])+float(ck['mean'])}
            for frac in fractions:
                name=f'input_oracle_{int(round(frac*100)):03d}'
                if abs(frac-1.0)<1e-12: pn=pn1
                else:
                    xa=x0+frac*(x1-x0); txa,_=make_inputs(xa,tod,dow,device); pn=adapter.model(txa,te)
                    if pn.ndim==4 and pn.shape[-1]==1: pn=pn.squeeze(-1)
                preds[name]=pn*float(ck['std'])+float(ck['mean'])
        y=b['y_raw'].numpy(); valid=b['future_valid'].numpy(); direct=b['direct_mask'].numpy(); support=b['support_mask'].numpy()
        pnp={k:v.detach().cpu().numpy().astype(np.float32) for k,v in preds.items()}
        for name,pred in pnp.items():
            for part,rm in flags.items(): metrics[name][part].update(pred,y,valid,direct,support,rm)
        for name in ['incomplete','input_oracle_100']:
            for g,rm in gf.items(): group_metrics[name][g].update(pnp[name],y,valid,direct,support,rm)
        q0=per_query_masked_mae(pnp['incomplete'],y,valid,direct); q1=per_query_masked_mae(pnp['input_oracle_100'],y,valid,direct)
        l3=latent_query_stats(s0['layer_3_node_4d'].detach().cpu().numpy(),s1['layer_3_node_4d'].detach().cpu().numpy(),direct,support)
        l5=latent_query_stats(s0['prehead_4d'].detach().cpu().numpy(),s1['prehead_4d'].detach().cpu().numpy(),direct,support)
        mr=b['missing_ratio'].numpy()
        for j,qid in enumerate(idx):
            qrows.append({'index':int(qid),'partition':'dev' if int(qid) in dev_set else ('final' if int(qid) in final_set else 'other'),'missing_ratio':float(mr[j]),
                'base_masked_mae':float(q0[j]),'complete_masked_mae':float(q1[j]),'oracle_abs_gain':float(q0[j]-q1[j]) if np.isfinite(q0[j]) and np.isfinite(q1[j]) else np.nan,
                'l3_rel_l2':float(l3['rel_l2'][j]),'l3_cosine':float(l3['cosine'][j]),'l3_direct_rel_l2':float(l3['direct_rel_l2'][j]),'l3_support_rel_l2':float(l3['support_rel_l2'][j]),
                'l5_rel_l2':float(l5['rel_l2'][j]),'l5_cosine':float(l5['cosine'][j]),'l5_direct_rel_l2':float(l5['direct_rel_l2'][j]),'l5_support_rel_l2':float(l5['support_rel_l2'][j])})
        if bi%20==0 or bi==len(dl): log(f'ORACLE INFER batch={bi}/{len(dl)} elapsed={time.time()-t0:.1f}s')
    sm={v:{p:metrics[v][p].summary() for p in parts} for v in variant_names}
    gains={}
    for v in variant_names[1:]:
        gains[v]={}
        for p in parts:
            gains[v][p]={k:improvement_pct(sm['incomplete'][p][k],sm[v][p][k]) for k in ['overall_mae','masked_mae','support_mae','overall_rmse','masked_rmse','support_rmse']}
    gm={v:{g:group_metrics[v][g].summary() for g in group_metrics[v]} for v in group_metrics}
    gg={}
    for g in cfg['oracle']['missing_groups']:
        gg[g]={'masked_mae_gain_pct':improvement_pct(gm['incomplete'][g]['masked_mae'],gm['input_oracle_100'][g]['masked_mae']),
               'masked_rmse_gain_pct':improvement_pct(gm['incomplete'][g]['masked_rmse'],gm['input_oracle_100'][g]['masked_rmse'])}
    qdf=pd.DataFrame(qrows); qdf.to_csv(out/'per_query_oracle.csv',index=False)
    corr_stats={}
    for part in ['all','dev','final']:
        z=qdf if part=='all' else qdf[qdf.partition==part]
        corr_stats[part]={
            'missing_ratio_vs_oracle_gain_pearson':corr(z.missing_ratio,z.oracle_abs_gain),
            'missing_ratio_vs_oracle_gain_spearman':corr(z.missing_ratio,z.oracle_abs_gain,True),
            'l3_gap_vs_oracle_gain_pearson':corr(z.l3_rel_l2,z.oracle_abs_gain),
            'l3_gap_vs_oracle_gain_spearman':corr(z.l3_rel_l2,z.oracle_abs_gain,True),
            'l5_gap_vs_oracle_gain_pearson':corr(z.l5_rel_l2,z.oracle_abs_gain),
            'l5_gap_vs_oracle_gain_spearman':corr(z.l5_rel_l2,z.oracle_abs_gain,True),
            'mean_l3_rel_l2':float(z.l3_rel_l2.mean()),'mean_l5_rel_l2':float(z.l5_rel_l2.mean()),
            'mean_l3_cosine':float(z.l3_cosine.mean()),'mean_l5_cosine':float(z.l5_cosine.mean()),
            'oracle_beneficial_query_rate':float((z.oracle_abs_gain>0).mean()),'oracle_harmful_query_rate':float((z.oracle_abs_gain<0).mean()),
        }
    final_gain=gains['input_oracle_100']['final']['masked_mae']; gate=cfg['headroom_gate']
    if final_gain>=float(gate['strong_gain_pct']): decision='STRONG_INFORMATION_HEADROOM'
    elif final_gain>=float(gate['meaningful_gain_pct']): decision='MEANINGFUL_INFORMATION_HEADROOM'
    elif final_gain>=float(gate['limited_gain_pct']): decision='LIMITED_INFORMATION_HEADROOM'
    else: decision='WEAK_INFORMATION_HEADROOM'
    summary={'version':cfg['version'],'stage':a.stage,'run_tag':a.run_tag,'scope':'Frozen-PatchSTG inference-time recoverable-history information ceiling; not Bayes/retrained-architecture ceiling',
        'cache_identity':ident,'checkpoint':str(ckpath),'checkpoint_sha256':sha256_file(ckpath),'checkpoint_source':source,'canonical_history_audit':audit,
        'split':{'val_total':n,'dev_queries':len(dev_ids),'final_queries':len(final_ids),'seed':cfg['experiment']['seed']},'metrics':sm,'gains_pct':gains,'missing_group_metrics':gm,'missing_group_gains':gg,'latent_gap_and_correlation':corr_stats,
        'decision':decision,'final_complete_masked_mae_gain_pct':final_gain}
    write_json(out/'summary.json',summary)
    log('================================================================================')
    for v in variant_names:
        m=sm[v]['final']; g=0.0 if v=='incomplete' else gains[v]['final']['masked_mae']
        log(f'FINAL {v}: masked_MAE={m["masked_mae"]:.6f} masked_RMSE={m["masked_rmse"]:.6f} gain={g:+.3f}%')
    log(f'FINAL COMPLETE OBSERVATION: {sm["incomplete"]["final"]["masked_mae"]:.6f} -> {sm["input_oracle_100"]["final"]["masked_mae"]:.6f} ({final_gain:+.3f}%)')
    log(f'LATENT GAP FINAL: L3_relL2={corr_stats["final"]["mean_l3_rel_l2"]:.6f} L5_relL2={corr_stats["final"]["mean_l5_rel_l2"]:.6f} L3_cos={corr_stats["final"]["mean_l3_cosine"]:.6f} L5_cos={corr_stats["final"]["mean_l5_cosine"]:.6f}')
    log(f'DECISION={decision}')
    log(f'SUMMARY={out/"summary.json"}')
    log('================================================================================')
if __name__=='__main__': main()
