#!/usr/bin/env python3
"""Prepare externally trained chronological OOS predictions; does not train Backbone.

Manifest example:
{
 "folds": [
  {"checkpoint":"/path/checkpoint_fold1.pt", "trained_until":15000,
   "evaluate_from":15001,"evaluate_until":30000},
  {"checkpoint":"/path/checkpoint_fold2.pt", "trained_until":30000,
   "evaluate_from":30001,"evaluate_until":45000}
 ]
}

Proof responsibility: trained_until must be externally auditable (fold training logs).
"""
import _bootstrap,argparse,json
from pathlib import Path
import numpy as np,torch
from fare.config import read_config,cache_dir,hash_file,dump_json
from fare.protocol import SupportDataset
from fare.backbone import load_stla

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='FARE/configs/formal.yaml');p.add_argument('--target');p.add_argument('--dataset')
    p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args();cfg=read_config(a.config,a.dataset,a.target)
    ds=SupportDataset(cache_dir(cfg),'train')
    starts=ds.meta.window_start.to_numpy(np.int64)
    _,indices=np.unique(starts,return_index=True)
    manifest=json.loads(Path(a.manifest).read_text())
    if not manifest.get('folds'):raise RuntimeError('No chronological OOS folds')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results={};provenance=[]
    for fold in manifest['folds']:
        cutoff=int(fold['trained_until']);lo=int(fold['evaluate_from']);hi=int(fold['evaluate_until'])
        if cutoff>=lo:raise RuntimeError('Fold leakage: producer training end must precede evaluation start')
        relevant=[int(i) for i in indices if lo<=starts[i]<=hi and starts[i]>cutoff]
        if not relevant:continue
        checkpoint=Path(fold['checkpoint']).expanduser().resolve()
        model=load_stla(checkpoint,device)
        if model.n!=ds.n:raise ValueError('Fold checkpoint nodes mismatch')
        with torch.no_grad():
            for low in range(0,len(relevant),a.batch_size):
                ix=np.array(relevant[low:low+a.batch_size],dtype=np.int64)
                x=torch.from_numpy(np.asarray(ds.arr['x_obs_norm'][ix],dtype=np.float32)).to(device)
                m=torch.from_numpy(np.asarray(ds.arr['history_observed'][ix],dtype=bool)).to(device)
                tod=torch.from_numpy(np.asarray(ds.arr['tod_hist'][ix])).to(device)
                dow=torch.from_numpy(np.asarray(ds.arr['dow_hist'][ix])).to(device)
                pred=model(x,m,tod,dow)['prediction'].cpu().numpy()
                for j,i in enumerate(ix):
                    key=int(starts[i])
                    if key in results:raise RuntimeError(f'Overlapping OOS folds for window {key}')
                    results[key]=(pred[j],cutoff)
        provenance.append({'checkpoint':str(checkpoint),'sha256':hash_file(checkpoint),
                           'trained_until':cutoff,'evaluate_from':lo,'evaluate_until':hi,'rows':len(relevant)})
        del model
        if torch.cuda.is_available():torch.cuda.empty_cache()
    if not results:raise RuntimeError('No OOS forecast windows produced')
    ordered=sorted(results)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,window_start=np.asarray(ordered,dtype=np.int64),
                        trained_until=np.asarray([results[x][1] for x in ordered],dtype=np.int64),
                        prediction=np.stack([results[x][0] for x in ordered]))
    dump_json(path.with_suffix('.json'),{'folds':provenance,'rows':len(ordered),
                                        'sha256':hash_file(path),'source_manifest':str(a.manifest)})
    print(f'FARE_OOS_COMPLETE rows={len(ordered)} file={path}',flush=True)
if __name__=='__main__':main()
