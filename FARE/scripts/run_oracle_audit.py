#!/usr/bin/env python3
"""FARE v2.1.2 candidate-level Oracle Audit for actual v2.1.1 checkpoints.

Safe baseline: the frozen validation-selected checkpoint. No weight updates.
Outputs per-candidate realized MAE for each sampled query-region in Val/Test.
"""
from __future__ import annotations
import argparse,hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch
import _bootstrap
from fare.config import config_from_checkpoint,cache_path,project_root,sha256
from fare.runtime import setup
from fare.trainer import load_ckpt
from fare.data import SupportSplit,to_device
from fare.candidate_oracle import evaluate_query_regions
from fare.retrieval_diagnostics import clean,summarize


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--expect-target',choices=('speed','flow'))
    parser.add_argument('--expect-dataset')
    parser.add_argument('--memory')
    parser.add_argument('--split',choices=('val','test','both'),default='both')
    parser.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    parser.add_argument('--max-queries',type=int,default=12,help='Per split/severity; 0=entire split (expensive)')
    parser.add_argument('--max-regions',type=int,default=2,help='Per query; 0=all regions')
    parser.add_argument('--max-candidates',type=int,default=64,help='Top legal coarse candidates; 0=all')
    parser.add_argument('--seed',type=int,default=10)
    parser.add_argument('--output')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--no-strict-pool-match',action='store_true',help='Debug only: disables replay consistency')
    args=parser.parse_args()
    if min(args.max_queries,args.max_regions,args.max_candidates)<0:parser.error('Counts must be >= 0')
    if not args.severities or any(s not in (20,40,60,90) for s in args.severities):parser.error('Invalid severities')
    if len(set(args.severities))!=len(args.severities):parser.error('Duplicate severities')
    if args.smoke:
        args.max_queries=2;args.max_regions=1;args.max_candidates=4;args.severities=[40]
    return args


def sample_indices(total,max_rows):
    if max_rows==0 or max_rows>=total:return list(range(total))
    return sorted(set(int(i) for i in np.linspace(0,total-1,max_rows,dtype=np.int64)))


def sampled_regions(count,max_regions,seed,window_start):
    if max_regions==0 or max_regions>=count:return list(range(count))
    # The same physical window is assigned the same region sample across severities.
    rng=np.random.default_rng((int(seed)*7919+int(window_start)*131) % (2**32))
    return sorted(rng.choice(count,size=max_regions,replace=False).tolist())


def append_row(file,row):
    with file.open('a',encoding='utf-8') as f:
        f.write(json.dumps(clean(row),ensure_ascii=False,allow_nan=False)+'\n')
        f.flush()


def manifest(checkpoint,cfg,args,memory_path):
    return {'audit_version':'2.1.2','checkpoint':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),
            'memory':str(memory_path or ''),
            'memory_sha256':sha256(memory_path) if memory_path and Path(memory_path).is_file() else None,
            'dataset':cfg['experiment']['dataset'],
            'target':cfg['experiment']['target'],'seed':args.seed,
            'split':args.split,'severities':args.severities if args.split!='val' else [],
            'max_queries':args.max_queries,'max_regions':args.max_regions,
            'max_candidates':args.max_candidates,'strict_pool_match':not args.no_strict_pool_match,
            'oracle_scope':'region_singleton_fixed_other_regions'}


def main():
    args=parse_args()
    checkpoint=Path(args.checkpoint).expanduser().resolve()
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('model')!='FARE':raise RuntimeError('Checkpoint is not FARE')
    cfg=config_from_checkpoint(ck)
    target=cfg['experiment']['target'];dataset=cfg['experiment']['dataset']
    if args.expect_target and target != args.expect_target:
        raise RuntimeError(f'PBS TARGET={args.expect_target} differs from checkpoint target={target}')
    if args.expect_dataset and dataset != args.expect_dataset:
        raise RuntimeError(f'PBS DATASET={args.expect_dataset} differs from checkpoint dataset={dataset}')
    if cfg['experiment']['seed']!=args.seed:raise RuntimeError('Checkpoint seed differs from requested audit seed')
    memory_path=args.memory or ck.get('memory_path')
    default=project_root()/'FARE_exps'/target/dataset/'oracle_audit'/f'{checkpoint.parent.name}_{time.strftime("%Y%m%d_%H%M%S")}'
    folder=Path(args.output or default).expanduser().resolve()
    folder.mkdir(parents=True,exist_ok=True)
    wanted=manifest(checkpoint,cfg,args,memory_path)
    mf=folder/'audit_manifest.json'
    if mf.exists():
        if not args.resume:raise FileExistsError(f'Audit output already exists: {mf} (use --resume)')
        prior=json.loads(mf.read_text(encoding='utf-8'))
        if prior!=wanted:raise RuntimeError('Audit resume manifest mismatch; not mixing different checkpoints/protocols')
    else:
        if args.resume and (folder/'oracle_records.jsonl').exists():
            raise RuntimeError('Cannot resume records with missing manifest')
        mf.write_text(json.dumps(wanted,indent=2),encoding='utf-8')
    records=folder/'oracle_records.jsonl'
    all_rows=[];done=set()
    if args.resume and records.exists():
        for line in records.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row=json.loads(line)
                key=(row['split'],row.get('severity'),row['row_id'],row['region_id'])
                if key not in done:done.add(key);all_rows.append(row)
    print('FARE_ORACLE_START '+json.dumps({**wanted,'output':str(folder)}),flush=True)
    model,memory,train,val,device,meta=setup(cfg,ck['metadata']['backbone_checkpoint'],memory_path)
    if meta['backbone_sha256']!=ck['metadata']['backbone_sha256']:
        raise RuntimeError('Frozen backbone SHA mismatch')
    load_ckpt(checkpoint,model,device)
    model.eval()
    with torch.no_grad():model.refresh(memory)
    splits=['val','test'] if args.split=='both' else [args.split]
    for split in splits:
        sevlist=[None] if split=='val' else args.severities
        for sev in sevlist:
            ds=val if sev is None else SupportSplit(cache_path(cfg,'evaluation',sev),'test')
            chosen=sample_indices(len(ds),args.max_queries)
            start=time.monotonic()
            for step,idx in enumerate(chosen,1):
                original=ds[idx]
                ws=int(original['window_start'])
                regions=sampled_regions(model.regions.g,args.max_regions,args.seed,ws)
                regions=[g for g in regions if (split,sev,int(idx),g) not in done]
                if not regions:continue
                batch=to_device({k:(v.unsqueeze(0) if isinstance(v,torch.Tensor) and v.ndim>0 else v.reshape(1))
                                 for k,v in original.items()},device)
                with torch.no_grad():
                    measured=evaluate_query_regions(model,memory,batch,std=meta['std'],
                              max_candidates=args.max_candidates,region_ids=regions,
                              strict_pool_match=not args.no_strict_pool_match)
                for row in measured:
                    row.update({'split':split,'severity':sev,'row_id':int(idx),
                        'window_start':ws,'target':target,'dataset':dataset})
                    append_row(records,row)
                    all_rows.append(row)
                    done.add((split,sev,int(idx),row['region_id']))
                print('FARE_ORACLE_PROGRESS '+json.dumps({'split':split,'severity':sev,
                     'done_queries':step,'planned_queries':len(chosen),'region_records':len(all_rows),
                     'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
            summarize(all_rows,folder)
            if torch.cuda.is_available():torch.cuda.empty_cache()
    summary=summarize(all_rows,folder)
    print('FARE_ORACLE_COMPLETE '+json.dumps({'output':str(folder),'region_records':len(all_rows),
                                             'summary_groups':len(summary)}),flush=True)


if __name__=='__main__':main()
