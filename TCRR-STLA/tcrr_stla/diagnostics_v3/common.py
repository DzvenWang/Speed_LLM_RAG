"""Self-contained inference-only diagnostics. No train or optimizer operations.

All labels enter only post-forward metrics and hindsight probes. Audit labels must
never influence model features, candidate generation, logits, or action weights.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

SCHEMA = 'oneshot-v3-inference-audit-1'


def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False,default=str),encoding='utf-8')
    os.replace(tmp,path)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


class ProtocolDigest:
    """Exactly ordered semantic sample evidence; independent of CUDA batch size."""
    FIELDS=('window_start','x','mask','tod','dow','y_raw','future_valid','direct_mask')
    def __init__(self):
        self.parts={k:hashlib.sha256() for k in self.FIELDS};self.n=0
    def update(self,b):
        for k in self.FIELDS:
            if k not in b:raise KeyError('Protocol field missing: '+k)
            a=b[k].detach().cpu().contiguous().numpy()
            # float32 / integer representations deterministic across either project
            if a.dtype.kind in 'f': a=np.asarray(a,dtype='<f4')
            elif a.dtype.kind=='b': a=np.asarray(a,dtype=np.uint8)
            else:a=np.asarray(a,dtype='<i8')
            self.parts[k].update(a.tobytes(order='C'))
        self.n+=int(b['x'].shape[0])
    def result(self):
        return {'query_count':self.n,'field_sha256':{k:v.hexdigest() for k,v in self.parts.items()}}


def np_cpu(x):
    return x.detach().float().cpu().numpy() if hasattr(x,'detach') else np.asarray(x)


def _query_stats(y,valid,direct,base,pred):
    """Original-unit forecast arrays [B,P,N]; masked official element denominators."""
    y=np.asarray(y,dtype=np.float32); valid=np.asarray(valid,dtype=bool)
    direct=np.asarray(direct,dtype=bool)
    base=np.asarray(base,dtype=np.float32);pred=np.asarray(pred,dtype=np.float32)
    if not(y.shape==valid.shape==base.shape==pred.shape):raise ValueError('Forecast shape mismatch')
    if direct.shape != (y.shape[0],y.shape[2]):raise ValueError('Direct node mask shape mismatch')
    ok=valid & np.isfinite(y) & np.isfinite(base) & np.isfinite(pred)
    e0=np.abs(base-y);e1=np.abs(pred-y);g=e0-e1
    masked=ok & direct[:,None,:]
    counts=ok.sum((1,2));mask_counts=masked.sum((1,2))
    def sum_mask(v,m):return np.where(m,v,0).sum((1,2),dtype=np.float64)
    return {'count':counts,'base_l1':sum_mask(e0,ok),'full_l1':sum_mask(e1,ok),
            'base_sq':sum_mask((base-y)**2,ok),'full_sq':sum_mask((pred-y)**2,ok),
            'direct_count':mask_counts,'direct_base_l1':sum_mask(e0,masked),
            'direct_full_l1':sum_mask(e1,masked),
            'gain_positive_sum':sum_mask(np.maximum(g,0),ok),
            'harm_sum':sum_mask(np.maximum(-g,0),ok),
            'beneficial_count':sum_mask((g>1e-8).astype(np.float32),ok),
            'harmful_count':sum_mask((g<-1e-8).astype(np.float32),ok),
            'corrected_count':sum_mask((np.abs(pred-base)>1e-8).astype(np.float32),ok)}


def add_mode(rows,b,base,y,valid,direct,pred,mode,sev,extra=None):
    stats=_query_stats(y,valid,direct,base,pred)
    extra=extra or {}
    bsz=len(stats['count'])
    starts=np_cpu(b['window_start']).astype(np.int64).reshape(-1)
    # Family comes from original SupportSplit metadata at the caller, NOT labels.
    for i in range(bsz):
        r={'severity':int(sev),'mode':str(mode),'query_idx':int(b['index'][i]),
           'window_start':int(starts[i]),'family':extra.get('family',['unknown']*bsz)[i]}
        r.update({k:float(v[i]) for k,v in stats.items()})
        rows.append(r)


def add_horizon(rows,b,base,y,valid,direct,pred,sev,mode='full',family=None):
    starts=np_cpu(b['window_start']).astype(np.int64).reshape(-1)
    fam=family or ['unknown']*len(starts)
    for h in range(y.shape[1]):
        st=_query_stats(y[:,h:h+1],valid[:,h:h+1],direct,base[:,h:h+1],pred[:,h:h+1])
        for i in range(len(starts)):
            rows.append({'severity':int(sev),'mode':mode,'horizon':h+1,
                         'window_start':int(starts[i]),'family':fam[i],
                         **{k:float(v[i]) for k,v in st.items()}})


def summary(rows,keys):
    if not rows:return pd.DataFrame()
    df=pd.DataFrame(rows)
    agg_cols=('count','base_l1','full_l1','base_sq','full_sq','direct_count',
              'direct_base_l1','direct_full_l1','gain_positive_sum','harm_sum',
              'beneficial_count','harmful_count','corrected_count')
    grouped=df.groupby(keys,dropna=False,sort=True)[list(agg_cols)].sum().reset_index()
    n=grouped['count'].to_numpy(dtype=float);md=grouped['direct_count'].to_numpy(dtype=float)
    def div(c,den=n):return np.divide(grouped[c],den,out=np.full(len(n),np.nan),where=den>0)
    grouped['MAE']=div('full_l1');grouped['base_MAE']=div('base_l1')
    grouped['RMSE']=np.sqrt(div('full_sq'));grouped['base_RMSE']=np.sqrt(div('base_sq'))
    grouped['MAE_gain']=grouped['base_MAE']-grouped['MAE']
    grouped['direct_MAE']=div('direct_full_l1',md)
    grouped['direct_base_MAE']=div('direct_base_l1',md)
    grouped['direct_gain']=grouped['direct_base_MAE']-grouped['direct_MAE']
    grouped['positive_gain_per_element']=div('gain_positive_sum')
    grouped['harm_per_element']=div('harm_sum')
    grouped['beneficial_rate']=div('beneficial_count')
    grouped['harmful_rate']=div('harmful_count')
    grouped['corrected_rate']=div('corrected_count')
    return grouped


def bootstrap(rows,boots=300,block_steps=288,seed=10):
    """Paired time-block bootstrap. Does NOT claim independent run significance."""
    df=pd.DataFrame([r for r in rows if r['mode']=='full'])
    if df.empty:return pd.DataFrame()
    out=[];rng=np.random.default_rng(seed)
    for (sev,fam),g in df.groupby(['severity','family'],sort=True):
        g=g.copy();g['block']=np.floor_divide(g.window_start.astype(np.int64),int(block_steps))
        blocks=g.groupby('block')[['count','base_l1','full_l1']].sum().to_numpy(dtype=np.float64)
        if not len(blocks):continue
        res=[]
        for _ in range(boots):
            ids=rng.integers(0,len(blocks),size=len(blocks));v=blocks[ids].sum(0)
            if v[0]>0:res.append((v[1]-v[2])/v[0])
        out.append({'severity':sev,'family':fam,'n_blocks':len(blocks),
                    'gain':float((blocks[:,1].sum()-blocks[:,2].sum())/blocks[:,0].sum()),
                    'ci_lower':float(np.quantile(res,.025)),'ci_upper':float(np.quantile(res,.975))})
    return pd.DataFrame(out)


def save_severity(outdir,rows,horizon_rows,mechanism_rows,manifest):
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/'query_modes.csv',index=False)
    pd.DataFrame(horizon_rows).to_csv(outdir/'query_horizon.csv',index=False)
    pd.DataFrame(mechanism_rows).to_csv(outdir/'mechanism_queries.csv',index=False)
    summary(rows,['severity','mode']).to_csv(outdir/'mode_summary.csv',index=False)
    summary(rows,['severity','family','mode']).to_csv(outdir/'family_summary.csv',index=False)
    summary(horizon_rows,['severity','family','horizon','mode']).to_csv(outdir/'family_horizon.csv',index=False)
    atomic_json(outdir/'manifest.json',manifest)


def consolidate(root,severities):
    root=Path(root)
    queries=[];horizon=[];mechanisms=[]
    for sev in severities:
        sub=root/f's{sev}'
        if not (sub/'manifest.json').is_file(): raise FileNotFoundError(sub/'manifest.json')
        queries.append(pd.read_csv(sub/'query_modes.csv'))
        horizon.append(pd.read_csv(sub/'query_horizon.csv'))
        mechanisms.append(pd.read_csv(sub/'mechanism_queries.csv'))
    q=pd.concat(queries,ignore_index=True);h=pd.concat(horizon,ignore_index=True)
    m=pd.concat(mechanisms,ignore_index=True)
    summary(q.to_dict('records'),['severity','mode']).to_csv(root/'mode_severity.csv',index=False)
    summary(q.to_dict('records'),['severity','family','mode']).to_csv(root/'mode_family.csv',index=False)
    summary(h.to_dict('records'),['severity','family','horizon','mode']).to_csv(root/'mode_family_horizon.csv',index=False)
    base=summary(q.to_dict('records'),['severity','mode'])
    avg=base.groupby('mode')[['MAE','RMSE','MAE_gain','base_MAE','base_RMSE']].mean().reset_index()
    avg.to_csv(root/'mode_average.csv',index=False)
    m.to_csv(root/'mechanism_queries.csv',index=False)
    bootstrap(q.to_dict('records')).to_csv(root/'paired_block_bootstrap.csv',index=False)
    f=q[q['mode']=='full'].copy()
    f['query_harm_per_element']=np.divide(f['harm_sum'],f['count'],out=np.zeros(len(f)),where=f['count']>0)
    quant=(f.groupby(['severity','family'])['query_harm_per_element']
             .quantile([.50,.90,.95,.99]).unstack().reset_index())
    quant=quant.rename(columns={.50:'harm_query_p50',.90:'harm_query_p90',.95:'harm_query_p95',.99:'harm_query_p99'})
    quant.to_csv(root/'harm_tail_summary.csv',index=False)
    numeric=m.select_dtypes(include='number').columns.difference(['window_start','severity','episode_count'])
    if len(numeric):m.groupby(['severity','family'])[list(numeric)].mean().reset_index().to_csv(root/'mechanism_group_summary.csv',index=False)
    return {'n_query_mode_rows':len(q),'n_query_horizon_rows':len(h),'n_mechanism_rows':len(m),
            'modes':sorted(q['mode'].unique().tolist()),'averages':avg.to_dict('records')}


def reproduce_official(run_dir, severity, rows, tolerance_fraction=0.003):
    """Compare a frozen M3 re-evaluation to the already completed formal Full.
    Does not silently rewrite official model evaluation artifacts.
    """
    p=Path(run_dir)/'evaluation'/f's{severity}'/'full'/'summary.json'
    if not p.is_file():return {'status':'REFERENCE_NOT_FOUND','path':str(p)}
    reference=json.loads(p.read_text())
    metrics=reference.get('metrics',{}).get('ALL',{})
    if 'MAE' not in metrics or 'RMSE' not in metrics:return {'status':'INVALID_REFERENCE','path':str(p)}
    table=summary([r for r in rows if r['mode']=='full'],['severity','mode'])
    if len(table)!=1:raise ValueError('Missing unique full-mode result')
    got=table.iloc[0]
    diffs={k:{'saved':float(metrics[k]),'recomputed':float(got[k]),
              'absolute_difference':abs(float(got[k])-float(metrics[k]))}
           for k in ('MAE','RMSE')}
    passed=all(v['absolute_difference'] <= max(0.002,abs(v['saved'])*tolerance_fraction)
               for v in diffs.values())
    return {'status':'PASS' if passed else 'MISMATCH','path':str(p),'metrics':diffs,
            'tolerance_fraction':tolerance_fraction,'notes':'same checkpoint, masks and full test; batch-level FP differences allowed'}


def protocol_query_rows(batch, raw_base, severity):
    """Per-query hashes locate the first *actual* protocol disagreement, not only aggregates."""
    raw_base=np.asarray(raw_base,dtype='<f4')
    fields=('x','mask','tod','dow','y_raw','future_valid','direct_mask')
    cpu={k:np.asarray(batch[k].detach().cpu().numpy()) for k in fields}
    starts=np.asarray(batch['window_start'].detach().cpu().numpy(),dtype=np.int64)
    inds=np.asarray(batch['index'].detach().cpu().numpy(),dtype=np.int64)
    result=[]
    for i in range(len(starts)):
        row={'severity':int(severity),'query_idx':int(inds[i]),'window_start':int(starts[i]),
             'base_mean':float(raw_base[i].mean()),'base_abs_mean':float(np.abs(raw_base[i]).mean())}
        for key,arr in [*[(k,cpu[k]) for k in fields],('base_prediction',raw_base)]:
            x=arr[i]
            if x.dtype.kind=='f': x=np.asarray(x,dtype='<f4')
            elif x.dtype.kind=='b':x=np.asarray(x,dtype=np.uint8)
            else:x=np.asarray(x,dtype='<i8')
            row[key+'_sha256']=hashlib.sha256(x.tobytes(order='C')).hexdigest()
        result.append(row)
    return result
