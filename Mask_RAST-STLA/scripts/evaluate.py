#!/usr/bin/env python3
from pathlib import Path
import argparse,json,os,sys
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_rast_stla.config import load_config,cache_path
from mask_rast_stla.data import SupportSplit,FullTrainRepository
from mask_rast_stla.trainer import build_model,move,predict
from mask_rast_stla.memory import LatentMemoryIndex,build_memory_index
from mask_rast_stla.evaluation import summarize_by_family,save_summary

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--target',choices=['speed','flow']);ap.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90]);ap.add_argument('--config',default=None);a=ap.parse_args()
    if a.target:os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);model,_=build_model(cfg,Path(ck['backbone_checkpoint']),device);model.load_state_dict(ck['state_dict'],strict=True);model.to(device).eval();formal=cache_path(cfg,'formal');repo=FullTrainRepository(formal);store=Path(a.checkpoint).resolve().parent/'best_memory_index.npz';index=LatentMemoryIndex.load(store) if store.is_file() else build_memory_index(model.memory_encoder,repo,device,int(cfg['retrieval']['store_batch_size']));index.to(device);mean=float(ck['mean']);std=float(ck['std']);rows=[]
    for sev in a.severities:
        cache=cache_path(cfg,'evaluation',sev);ds=SupportSplit(cache,'test');dl=DataLoader(ds,batch_size=int(cfg['training']['batch_size']),shuffle=False,num_workers=0,pin_memory=torch.cuda.is_available());p,y,v,d,fg,lg=predict(model,dl,index,repo,device,mean,std);fam=summarize_by_family(p,y,v,d,ds.meta);out=Path(a.checkpoint).resolve().parent/'evaluation'/f's{sev}';out.mkdir(parents=True,exist_ok=True);np.savez_compressed(out/'test_predictions.npz',prediction=p.astype(np.float32));save_summary(out/'summary.json',{'severity':sev,'metrics':fam,'forecast_gate_mean':fg,'latent_gate_mean':lg});rows.append({'severity':sev,**fam['ALL']});print(f'EVAL s{sev} '+json.dumps(fam['ALL']),flush=True)
    pd.DataFrame(rows).to_csv(Path(a.checkpoint).resolve().parent/'evaluation'/'severity_summary.csv',index=False)
if __name__=='__main__':main()
