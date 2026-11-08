#!/usr/bin/env python3
"""Evaluate the exact saved v2 config; only Train Memory is consulted."""
from pathlib import Path
import argparse,json,os,sys
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tcrr_stla.config import cache_path,relocate_saved_config,file_sha256
from tcrr_stla.data import SupportSplit,RetrievalCache
from tcrr_stla.evidence import FutureMemory
from tcrr_stla.normal_teacher import prepare_normal_teacher_bank
from tcrr_stla.trainer import build_model,_make_loader,predict
from tcrr_stla.evaluation import summarize_by_family,save_summary
from scripts.build_retrieval_cache import ensure_cache


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',required=True)
    ap.add_argument('--target',choices=['speed','flow'])
    ap.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    ap.add_argument('--backbone-checkpoint')
    ap.add_argument('--batch-size',type=int,default=8,help='Separate safe evaluation batch size')
    ap.add_argument('--num-workers',type=int,default=0)
    ap.add_argument('--evidence-mode',choices=['full','no_evidence','shuffled','random_future','cross_history','no_risk','standard_future','no_mask_replay','no_rerank','no_retrieval','legacy_v23','no_traffic_regime','no_traffic_retrieval','no_failure_transfer','no_utility','no_traffic_control','no_mask_matched_retrieval','no_failure_encoder','no_normal_residual','no_failure_residual','no_candidate_utility','no_mixture_utility','shuffled_cf_pairing','no_support_conditioning','no_gain_harm'])
    a=ap.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if str(ck.get('version','')) not in ('2.0.0','3.0.0'):
        raise RuntimeError('Only v2/v3 checkpoints can be evaluated')
    cfg=relocate_saved_config(ck['config'])
    if a.evidence_mode:
        cfg['training']['ablation']=a.evidence_mode
    cfg['training']['batch_size']=int(a.batch_size)
    cfg['runtime']['num_workers']=int(a.num_workers)
    cfg['runtime']['persistent_workers']=False
    cfg['runtime']['preload_retrieval_memory_gpu']=False
    cfg['runtime']['replay_bank_on_gpu']=False
    cfg['runtime']['normal_teacher_on_gpu']=False
    if a.target and a.target!=cfg['experiment']['target']:raise RuntimeError('Target must match saved checkpoint')
    os.environ['TRAFFIDENT_TARGET']=cfg['experiment']['target']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone_path=Path(a.backbone_checkpoint or os.environ.get('BACKBONE_CHECKPOINT') or ck['backbone_checkpoint'])
    if not backbone_path.is_file():raise FileNotFoundError(f'STLA checkpoint unavailable: {backbone_path}; pass --backbone-checkpoint')
    if ck.get('backbone_sha256') and file_sha256(backbone_path)!=ck['backbone_sha256']:
        raise RuntimeError('Incorrect STLA Backbone checkpoint: SHA256 differs from training')
    model,_=build_model(cfg,backbone_path,device)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    # Train used an immutable reference teacher for cacheable mask-replay. Do
    # not accidentally use the subsequently fine-tuned query backbone here.
    teacher=None
    if cfg['runtime'].get('frozen_replay_teacher',False) and (a.evidence_mode or cfg['training'].get('ablation','full'))!='no_retrieval':
        initial,_=build_model(cfg,backbone_path,device)
        teacher=initial.stla.eval()
        for p in teacher.parameters():p.requires_grad_(False)
        print('EVAL_REPLAY_TEACHER fixed_pretrained_backbone',flush=True)
    mean=float(ck['mean']);std=float(ck['std'])
    formal=cache_path(cfg,'formal');store=FutureMemory(formal)
    if cfg['runtime'].get('preload_retrieval_memory_gpu',False):
        print(f'EVAL_RETRIEVAL_MEMORY_GPU loaded={store.preload(device)}',flush=True)
    mode=a.evidence_mode or cfg['training'].get('ablation','full')
    normal_bank=prepare_normal_teacher_bank(cfg,store,teacher,backbone_path,device) if teacher is not None and mode!='no_retrieval' else None
    os.environ['BACKBONE_CHECKPOINT']=str(backbone_path)
    rows=[]
    for sev in a.severities:
        c=cache_path(cfg,'evaluation',sev)
        r=None if mode=='no_retrieval' else ensure_cache(cfg,c,'test',severity=sev)
        ds=SupportSplit(c,'test')
        actual_batch=int(cfg['training']['batch_size'])
        while True:
            try:
                dl=_make_loader(ds,actual_batch,False,cfg)
                p=predict(model,dl,RetrievalCache(r) if r else None,store,device,mean,std,cfg,mode,
                          teacher=teacher,normal_bank=normal_bank)
                break
            except torch.cuda.OutOfMemoryError:
                if actual_batch<=1:raise
                actual_batch=max(1,actual_batch//2)
                if device.type=='cuda':torch.cuda.empty_cache()
                print(f'EVALUATION_OOM_RETRY severity={sev} batch={actual_batch}',flush=True)
        fam=summarize_by_family(p['prediction'],p['true'],p['valid'],p['direct'],ds.meta)
        base=summarize_by_family(p['base'],p['true'],p['valid'],p['direct'],ds.meta)
        out=Path(a.checkpoint).resolve().parent/'evaluation'/f's{sev}'/mode;out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/'test_predictions.npz',prediction=p['prediction'].astype(np.float32),
             base_prediction=p['base'].astype(np.float32),correction=p['correction'].astype(np.float32))
        save_summary(out/'summary.json',{'severity':sev,'mode':mode,'checkpoint_sha256':file_sha256(a.checkpoint),'metrics':fam,'backbone':base,'gate_mean':p['gate_mean'],'useful_candidate_recall':p['useful_candidate_recall'],'weighted_candidate_utility_normalized':p['weighted_candidate_utility_normalized']})
        rows.append({'severity':sev,'mode':mode,**fam['ALL'],**{'base_'+k:v for k,v in base['ALL'].items()}})
        print('EVAL '+json.dumps(rows[-1]),flush=True)
    out=Path(a.checkpoint).resolve().parent/'evaluation';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/f'severity_summary.{mode}.csv',index=False)
    if mode==ck['config']['training'].get('ablation','full'):
        # Only the ORIGINAL checkpoint training mode updates the official
        # summary. Inference interventions must not overwrite Full results.
        pd.DataFrame(rows).to_csv(out/'severity_summary.csv',index=False)
        root=Path(a.checkpoint).resolve().parent/'summary.json'
        payload=json.loads(root.read_text()) if root.is_file() else {}
        payload.update(test_evaluated=True,evaluation_summary=rows)
        save_summary(root,payload)
    print(f'EVALUATION_PASS mode={mode} summary={out / f"severity_summary.{mode}.csv"}',flush=True)
if __name__=='__main__':main()
