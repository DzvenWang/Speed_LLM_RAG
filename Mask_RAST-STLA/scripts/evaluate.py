#!/usr/bin/env python3
from pathlib import Path
import argparse,json,os,sys
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_rast_stla.config import cache_path,relocate_saved_config,file_sha256
from mask_rast_stla.data import SupportSplit,FullTrainRepository
from mask_rast_stla.trainer import build_model,make_loader,predict
from mask_rast_stla.memory import LatentMemoryIndex
from mask_rast_stla.evaluation import summarize_by_family,save_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--target',choices=['speed','flow'])
    p.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    p.add_argument('--backbone-checkpoint');p.add_argument('--evidence-mode',choices=['full','no_retrieval','no_future','no_regional','global_only','no_none','shuffled','random_future','cross_history','no_innovation','no_risk']);a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if str(ck.get('version',''))!='2.3.0':raise RuntimeError('v2.3 evaluator requires a v2.3 checkpoint')
    cfg=relocate_saved_config(ck['config'])
    if a.target and a.target!=cfg['experiment']['target']:raise RuntimeError('Target does not match saved checkpoint')
    os.environ['TRAFFIDENT_TARGET']=cfg['experiment']['target']
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_path=Path(a.backbone_checkpoint or os.environ.get('BACKBONE_CHECKPOINT') or ck['backbone_checkpoint'])
    if not backbone_path.is_file():raise FileNotFoundError(f'STLA checkpoint unavailable: {backbone_path}; pass --backbone-checkpoint')
    if ck.get('backbone_sha256') and file_sha256(backbone_path)!=ck['backbone_sha256']:
        raise RuntimeError('Incorrect STLA Backbone checkpoint: SHA256 differs from training')
    model,_=build_model(cfg,backbone_path,dev)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    mode=a.evidence_mode or cfg['training'].get('ablation','full')
    store_path=Path(a.checkpoint).resolve().parent/'best_memory_index.npz'
    if mode!='no_retrieval':
        if not store_path.is_file():raise FileNotFoundError(f'Index must accompany checkpoint: {store_path}')
        index=LatentMemoryIndex.load(store_path).to(dev)
    else:index=None
    repo=FullTrainRepository(cache_path(cfg,'formal'))
    if cfg['runtime'].get('preload_repository_gpu',True):repo.preload(dev)
    if index is not None and (len(repo)!=len(index.starts_np) or not np.array_equal(repo.starts,index.starts_np)):
        raise RuntimeError('Index and Train-only repository fingerprints differ')
    mean=float(ck['mean']);std=float(ck['std'])
    rows=[]
    for sev in a.severities:
        ds=SupportSplit(cache_path(cfg,'evaluation',sev),'test')
        dl=make_loader(ds,int(cfg['training']['batch_size']),False,cfg)
        pr=predict(model,dl,index,repo,dev,mean,std,mode,amp=bool(cfg['runtime'].get('amp',False)))
        fm=summarize_by_family(pr['prediction'],pr['true'],pr['valid'],pr['direct'],ds.meta)
        bm=summarize_by_family(pr['base'],pr['true'],pr['valid'],pr['direct'],ds.meta)
        out=Path(a.checkpoint).resolve().parent/'evaluation'/f's{sev}'/mode;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'test_predictions.npz',prediction=pr['prediction'].astype(np.float32),
                            base_prediction=pr['base'].astype(np.float32),correction=pr['correction'].astype(np.float32))
        save_summary(out/'summary.json',{'severity':sev,'mode':mode,'checkpoint_sha256':file_sha256(a.checkpoint),'metrics':fm,'backbone':bm,'gate_mean':pr['gate_mean'],'useful_candidate_recall':pr['useful_candidate_recall'],'weighted_candidate_utility_normalized':pr['weighted_candidate_utility_normalized']})
        rows.append({'severity':sev,'mode':mode,**fm['ALL'],**{'base_'+k:v for k,v in bm['ALL'].items()}})
        print('EVAL '+json.dumps(rows[-1]),flush=True)
    edir=Path(a.checkpoint).resolve().parent/'evaluation';edir.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(edir/f'severity_summary.{mode}.csv',index=False)
    if mode==cfg['training'].get('ablation','full'):
        pd.DataFrame(rows).to_csv(edir/'severity_summary.csv',index=False)
        sfile=Path(a.checkpoint).resolve().parent/'summary.json'
        data=json.loads(sfile.read_text()) if sfile.is_file() else {}
        data.update(test_evaluated=True,evaluation_summary=rows);save_summary(sfile,data)
    print('EVALUATION_PASS',edir/f'severity_summary.{mode}.csv',flush=True)
if __name__=='__main__':main()
