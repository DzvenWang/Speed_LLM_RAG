from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def load_protocol_views(cfg, stage: str):
    from shared_protocol.consumer import load_frozen_stage
    manifest, views = load_frozen_stage(
        cfg['project']['protocol_artifact_root'],
        cfg['experiment']['subgraph'],
        cfg['protocol']['shared_protocol_name'],
        stage,
    )
    return manifest, views


def _time_features(bench):
    T=int(bench.traffic.shape[0]); freq=int(bench.cfg['forecast']['freq_minutes']); spd=int(round(24*60/freq))
    p=bench.sg_dir/'core'/'time_index.csv'
    if p.exists():
        df=pd.read_csv(p)
        for c in df.columns:
            s=pd.to_datetime(df[c],errors='coerce')
            if s.notna().mean()>0.95 and len(s)>=T:
                tod=((s.dt.hour*60+s.dt.minute)//freq).to_numpy(np.int16)[:T]; dow=s.dt.dayofweek.to_numpy(np.int8)[:T]
                return tod,dow,'core/time_index.csv'
    idx=np.arange(T); return (idx%spd).astype(np.int16),((idx//spd)%7).astype(np.int8),'index_modulo_fallback'


def materialize(bench,df,outdir,split):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True); H,P,N=bench.H,bench.P,bench.N; M=len(df)
    shapes={
      'x_obs_norm':((M,H,N),np.float32), 'x_full_norm':((M,H,N),np.float32),
      'history_observed':((M,H,N),np.uint8), 'artificial_missing':((M,H,N),np.uint8),
      'y_raw':((M,P,N),np.float32), 'y_norm':((M,P,N),np.float32), 'future_valid':((M,P,N),np.uint8),
      'direct_mask':((M,N),np.uint8), 'support_mask':((M,N),np.uint8),
      'tod_hist':((M,H),np.int16), 'dow_hist':((M,H),np.int8), 'tod_future':((M,P),np.int16), 'dow_future':((M,P),np.int8)
    }
    arr={k:np.lib.format.open_memmap(outdir/f'{split}_{k}.npy',mode='w+',dtype=dt,shape=sh) for k,(sh,dt) in shapes.items()}
    speed=bench.manifest['primary_channel']; mean=float(bench.stats['channels'][speed]['mean']); std=max(float(bench.stats['channels'][speed]['std']),1e-6); tod,dow,tsource=_time_features(bench)
    meta=[]
    for i,r in enumerate(df.itertuples(index=False)):
        s=bench.sample(str(r.view_id)); full=s.history_full[...,bench.speed_idx].astype(np.float32); fulln=(full-mean)/std
        obs=s.observation_mask[...,bench.speed_idx].astype(bool); art=s.artificial_mask.astype(bool); x=np.where(obs,fulln,0.0).astype(np.float32)
        arr['x_obs_norm'][i]=x; arr['x_full_norm'][i]=fulln; arr['history_observed'][i]=obs; arr['artificial_missing'][i]=art
        arr['y_raw'][i]=s.future_speed; arr['y_norm'][i]=(s.future_speed-mean)/std; arr['future_valid'][i]=s.raw_valid_future
        arr['direct_mask'][i]=art.any(axis=0); arr['support_mask'][i]=bench.predictive_context_metrics(s)['support_dependent_target_mask']
        st=int(s.window_start); arr['tod_hist'][i]=tod[st:st+H]; arr['dow_hist'][i]=dow[st:st+H]; arr['tod_future'][i]=tod[st+H:st+H+P]; arr['dow_future'][i]=dow[st+H:st+H+P]
        meta.append({'protocol_view_id':getattr(r,'protocol_view_id',''),'view_id':s.view_id,'view_group_id':s.view_group_id,'window_start':st,'family':str(s.view_spec['family']),'severity':float(s.view_spec['severity']),'distribution':str(s.view_spec.get('distribution','id'))})
        if (i+1)%500==0: print(f'materialize {split}: {i+1}/{M}',flush=True)
    for a in arr.values(): a.flush()
    pd.DataFrame(meta).to_csv(outdir/f'{split}_manifest.csv',index=False)
    np.save(outdir/'adjacency.npy',np.asarray(bench.adj,dtype=np.float32))
    return {'rows':M,'mean':mean,'std':std,'time_feature_source':tsource}
