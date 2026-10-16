#!/usr/bin/env python3
"""P0 hindsight candidate/utility/gate calibration from existing FULL best.pt.

Never writes model parameters or trains; future target is accessed ONLY after
retrieval/candidate proposals have been computed for offline evaluation.
"""
from __future__ import annotations
import argparse,json,os,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
METHOD='PCFT-STLA'
from pcft_stla.config import cache_path,relocate_saved_config,file_sha256
from pcft_stla.p0 import candidate_probe_batch,aggregate_calibration,sha256,support_stratum
from pcft_stla.trainer import build_model,make_loader,move
from pcft_stla.data import SupportSplit,FullTrainRepository
from pcft_stla.memory import LatentMemoryIndex


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True,help='trained v1.0 Full checkpoint')
    ap.add_argument('--backbone-checkpoint',help='override model STLA original checkpoint path')
    ap.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    ap.add_argument('--max-queries',type=int,default=0,help='0: all queries, positive: first N diagnostic queries')
    ap.add_argument('--batch-size',type=int,help='override probe batch size without affecting evaluated checkpoint')
    ap.add_argument('--output',help='default <checkpoint parent>/p0_retrieval')
    args=ap.parse_args()
    cp=Path(args.checkpoint).resolve();ck=torch.load(cp,map_location='cpu',weights_only=False)
    if ck.get('version')!='1.0.0' or ck['config']['training'].get('ablation','full')!='full':
        raise ValueError('P0 candidate oracle must use a v1.0 FULL checkpoint')
    cfg=relocate_saved_config(ck['config']);os.environ['TRAFFIDENT_TARGET']=cfg['experiment']['target']
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone=Path(args.backbone_checkpoint or os.environ.get('BACKBONE_CHECKPOINT') or ck['backbone_checkpoint'])
    if not backbone.is_file():raise FileNotFoundError(backbone)
    if ck.get('backbone_sha256') and file_sha256(backbone)!=ck['backbone_sha256']:
        raise ValueError('Backbone SHA256 mismatch with full checkpoint')
    model,_=build_model(cfg,backbone,dev)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    index_file=cp.parent/'best_memory_index.npz'
    if not index_file.is_file():raise FileNotFoundError('Missing FULL best memory index: '+str(index_file))
    index=LatentMemoryIndex.load(index_file).to(dev)
    repo=FullTrainRepository(cache_path(cfg,'formal'))
    if cfg['runtime'].get('preload_repository_gpu',True):repo.preload(dev)
    if len(repo)!=len(index.starts_np) or not np.array_equal(repo.starts,index.starts_np):
        raise RuntimeError('Train repository and memory index window starts do not match')
    result=[];calibration=[]
    output=Path(args.output).resolve() if args.output else cp.parent/'p0_retrieval'
    output.mkdir(parents=True,exist_ok=True)
    for sev in args.severities:
        ds=SupportSplit(cache_path(cfg,'evaluation',sev),'test',limit=args.max_queries or None)
        dl=make_loader(ds,int(args.batch_size or cfg['training']['batch_size']),False,cfg)
        batch_count=0
        for bb in dl:
            b=move(bb,dev)
            with torch.no_grad():
                pass
                with torch.autocast(device_type=dev.type,dtype=torch.float16,
                                    enabled=bool(cfg['runtime'].get('amp',False)) and dev.type=='cuda'):
                    out=model(b,index,repo,mode='full')
                batch_rows,bins=candidate_probe_batch(out,b,max_delta=model.max_delta,
                                      margin=float(cfg['training'].get('utility_margin',0.02)))
            indexes=b['index'].cpu().numpy()
            coverage=b['mask'].float().mean((1,2)).cpu().numpy()
            for name in ('base_mae_norm','full_mae_norm',
                         'best_single_candidate_or_none_mae_norm','best_single_candidate_gain_norm'):
                batch_rows[name.replace('_norm','_raw')] = batch_rows[name]*float(ck['std'])
            for j,ind in enumerate(indexes):
                datum={k:(int(v[j]) if k=='n_valid' else float(v[j])) for k,v in batch_rows.items()}
                datum.update(query_index=int(ind),severity=int(sev),
                             family=str(ds.meta.iloc[int(ind)].get('family','NA')),
                             window_start=int(ds.meta.iloc[int(ind)].get('window_start',int(ind))),
                             coverage=float(coverage[j]),support_bin=support_stratum(coverage[j]),
                             method=METHOD)
                result.append(datum)
            for item in bins:
                calibration.append(dict(severity=int(sev),**item))
            batch_count+=1
            if batch_count%50==0:
                print(f'P0_PROBE_PROGRESS method={METHOD} s{sev} samples={len(indexes)*batch_count}',flush=True)
        print(f'P0_PROBE_SEVERITY_PASS method={METHOD} s{sev} queries={len(ds)}',flush=True)
    frame=pd.DataFrame(result)
    if frame.empty:raise RuntimeError('P0 probe collected no queries')
    frame.to_csv(output/'p0_candidate_queries.csv',index=False)
    summaries=[]
    keys=[['severity'],['severity','family'],['severity','support_bin']]
    for groupkeys in keys:
        for value,part in frame.groupby(groupkeys,dropna=False):
            if not isinstance(value,tuple):value=(value,)
            obj={'grouping':'+'.join(groupkeys),'queries':len(part)}
            obj.update(zip(groupkeys,value))
            for name in batch_rows:
                obj[name]=float(part[name].mean())
            summaries.append(obj)
    pd.DataFrame(summaries).to_csv(output/'p0_candidate_summary.csv',index=False)
    tab=aggregate_calibration(calibration)
    tab.to_csv(output/'p0_risk_calibration.csv',index=False)
    ece=[]
    for sev,grp in tab.groupby('severity'):
        if grp['count'].sum()>0:
            value=float((grp['count']*grp['absolute_calibration_error'].fillna(0)).sum()/grp['count'].sum())
            ece.append({'severity':int(sev),'risk_gate_ece':value,'n_elements':int(grp['count'].sum())})
    pd.DataFrame(ece).to_csv(output/'p0_risk_ece.csv',index=False)
    manifest={'schema':'p0-v1','method':METHOD,'full_checkpoint':str(cp),
              'checkpoint_sha256':sha256(cp),'query_count':len(frame),
              'mean_std_units':'Oracle residual and MAE are in original normalized STLA units; raw MAE multiplies by checkpoint std',
              'oracle_definition':'Hindsight best single candidate with bounded tanh correction, or base; label selected PER FUTURE ELEMENT. It is NOT deployable and NOT a strict upper bound on combinations.',
              'candidate_universe':'Only the model-retrieved eligible historical candidates, not the full Train Memory.',
              'calibration_label':'Whether ungated correction improves normalized absolute error beyond training utility_margin.',
              'severities':args.severities,'max_queries':args.max_queries}
    (output/'p0_candidate_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('P0_CANDIDATE_PASS',output,flush=True)

if __name__=='__main__':main()
