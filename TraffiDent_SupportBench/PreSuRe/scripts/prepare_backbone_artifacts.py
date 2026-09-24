#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json,numpy as np,torch
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml,write_json,sha256_file,log
from presure.contract import assert_cache_ready,cache_dir
from presure.backbone import load_patchstg_adapter,infer_split,resolve_checkpoint
from presure.data import CacheSplit
from presure.pipeline import art_dir

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--checkpoint',default=None); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
 cfg=load_yaml(ROOT/'configs/presure.yaml'); ident=assert_cache_ready(cfg,a.stage); out=art_dir(cfg,a.stage); out.mkdir(parents=True,exist_ok=True)
 if (out/'backbone_manifest.json').exists() and not a.force:
  old=json.loads((out/'backbone_manifest.json').read_text()); curp,curck,cursrc=resolve_checkpoint(cfg,a.stage,a.checkpoint); cursha=sha256_file(curp)
  if old.get('checkpoint_sha256')==cursha and old.get('cache_identity',{}).get('cache_fingerprint')==ident.get('cache_fingerprint'):
   old['version']=cfg['version']; old['implementation_revision']=cfg.get('implementation_revision'); write_json(out/'backbone_manifest.json',old)
   print('Backbone artifacts already ready and current; manifest refreshed for',cfg['version'],':',out); return
  print('STALE backbone artifacts detected; rebuilding for current checkpoint/cache identity.',flush=True)
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); adapter,ckpath,ck,src=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint); cache=cache_dir(cfg,a.stage)
 bs=int(cfg['backbone_artifacts']['batch_size']); dtype=np.float16 if cfg['backbone_artifacts']['storage_dtype']=='float16' else np.float32
 for split in ['train','val']:
  pred=infer_split(adapter,cache,split,bs,device); np.save(out/f'{split}_base_pred.npy',pred.astype(dtype)); log(f'wrote {split}_base_pred {pred.shape} {dtype}')
  if split=='train':
   ds=CacheSplit(cache,'train'); res=np.asarray(ds.y,np.float32)-pred; np.save(out/'train_residual.npy',res.astype(dtype)); log(f'wrote train_residual {res.shape}')
 write_json(out/'backbone_manifest.json',{'artifact_type':'presure_backbone_artifacts','version':cfg['version'],'implementation_revision':cfg.get('implementation_revision'),'stage':a.stage,'cache_identity':ident,'checkpoint':str(ckpath),'checkpoint_sha256':sha256_file(ckpath),'checkpoint_source':src,'checkpoint_seed':int(ck['seed']),'storage_dtype':str(dtype),'test_instantiated':False})
 print('PREPARE BACKBONE ARTIFACTS PASS')
if __name__=='__main__': main()
