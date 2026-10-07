#!/usr/bin/env python3
from pathlib import Path
import argparse,os,sys,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_rast_stla.config import load_config,cache_path,resolve_stla_checkpoint
from mask_rast_stla.data import FullTrainRepository
from mask_rast_stla.trainer import build_model
from mask_rast_stla.memory import build_memory_index

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',choices=['speed','flow']);ap.add_argument('--checkpoint',help='Mask_RAST-STLA checkpoint; if omitted builds from initialized encoders');ap.add_argument('--backbone-checkpoint');ap.add_argument('--output');ap.add_argument('--smoke',action='store_true');ap.add_argument('--config',default=None);a=ap.parse_args()
    if a.target:os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config);device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if a.checkpoint:
        ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);backbone=Path(ck['backbone_checkpoint']);model,_=build_model(cfg,backbone,device);model.load_state_dict(ck['state_dict'],strict=True)
    else:
        backbone=resolve_stla_checkpoint(cfg,a.backbone_checkpoint);model,_=build_model(cfg,backbone,device)
    repo=FullTrainRepository(cache_path(cfg,'formal'),256 if a.smoke else None);idx=build_memory_index(model.memory_encoder,repo,device,16 if a.smoke else int(cfg['retrieval']['store_batch_size']));out=Path(a.output) if a.output else Path(cfg['project']['store_root'])/cfg['experiment']['dataset']/'memory_index.npz';idx.save(out,{'memory_rows':len(repo),'source_checkpoint':a.checkpoint or 'initialized','backbone':str(backbone)});print(f'STORE_BUILD_PASS {out}',flush=True)
if __name__=='__main__':main()
