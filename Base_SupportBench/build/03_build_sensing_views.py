#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from supportbench.config import load_config
from supportbench.sensing import enrich_view_spec, make_view_spec
from supportbench.utils import ensure_dir, log, read_json



def progress(label: str, i: int, total: int, rows_count: int) -> None:
    if total <= 0:
        return
    step=max(1,total//10)
    done=i+1
    if done==1 or done==total or done % step==0:
        log(f'{label}: windows={done:,}/{total:,} ({100.0*done/total:.1f}%); accumulated_views={rows_count:,}')

def add_spec(rows, sg_dir: Path, split: str, start: int, family: str, severity: float, instance: int, cfg: dict, H: int, N: int, distribution: str='id'):
    spec=make_view_spec(sg_dir.name,split,int(start),family,float(severity),int(instance),cfg,distribution)
    rows.append(enrich_view_spec(spec,H,N,sg_dir))


def build_one(sg_dir: Path, cfg: dict):
    man=read_json(sg_dir/'manifest.json'); H=int(man['history_len']); N=int(man['shape'][1])
    scfg=cfg['sensing']
    general=list(scfg['general_families'])
    severities=list(map(float,scfg['severities']))
    # Train severities are intentionally separable from the full evaluation grid.
    # This preserves the historical 40%-training protocol/checkpoint while allowing
    # new 90% robustness views in Validation/Test.
    train_sev=list(map(float,scfg.get('train_severities',severities)))
    main=float(scfg['main_severity']); rows=[]

    # Train: deterministic balanced pool across the explicitly configured Train grid.
    tr=np.load(sg_dir/'windows'/'train_full.npy')
    train_choices=[(f,s) for f in general for s in train_sev]
    vper=int(scfg['train_views_per_window'])
    for idx,s in enumerate(tr):
        add_spec(rows,sg_dir,'train',s,'complete',0.0,0,cfg,H,N)
        for v in range(vper):
            fam,sev=train_choices[(idx*vper+v)%len(train_choices)]
            add_spec(rows,sg_dir,'train',s,fam,sev,v,cfg,H,N,'id')
        progress(f'{sg_dir.name} train views',idx,len(tr),len(rows))

    # Validation/Test: exact original General-track generation for 20/40/60/90.
    for split in ['val','test']:
        starts=np.load(sg_dir/'windows'/f'{split}_full.npy')
        inst_base=int(scfg[f'{split}_instances_per_family_severity'])
        for split_idx,s in enumerate(starts):
            add_spec(rows,sg_dir,split,s,'complete',0.0,0,cfg,H,N)
            for fam in general:
                for sev in severities:
                    ninst=inst_base + (int(scfg['paired_main_instances'])-1 if abs(sev-main)<1e-9 else 0)
                    for i in range(max(1,ninst)):
                        add_spec(rows,sg_dir,split,s,fam,sev,i,cfg,H,N,'id')
            progress(f'{sg_dir.name} {split} views',split_idx,len(starts),len(rows))

    log(f'{sg_dir.name}: converting {len(rows):,} view records to DataFrame')
    df=pd.DataFrame(rows)
    out=ensure_dir(sg_dir/'views'); df.to_csv(out/'view_manifest.csv',index=False)
    log(f'{sg_dir.name}: {len(df):,} deterministic sensing views written')
    for split in ['train','val','test']:
        d=df[df['split']==split]
        log(f"{sg_dir.name} {split}: views={len(d):,}; families={sorted(d.family.unique())}")


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=None); ap.add_argument('--subgraph',action='append',default=[]); args=ap.parse_args(); cfg=load_config(args.config)
    root=Path(cfg['project']['benchmark_root']); items=sorted(root.glob('TraffiDent_Net_*'))
    if args.subgraph: items=[x for x in items if x.name in set(args.subgraph)]
    for x in items: build_one(x,cfg)
if __name__=='__main__': main()
