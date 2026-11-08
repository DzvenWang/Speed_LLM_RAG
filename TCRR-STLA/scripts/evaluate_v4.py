#!/usr/bin/env python3
"""Fixed-checkpoint, train-memory-only four-severity v4 evaluation."""
from __future__ import annotations
import argparse,os,sys,json
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tcrr_stla.config import cache_path,relocate_saved_config,file_sha256
from tcrr_stla.data import SupportSplit,RetrievalCache
from tcrr_stla.evidence import FutureMemory
from tcrr_stla.normal_teacher import prepare_normal_teacher_bank
from tcrr_stla.v4.trainer import build_v4_model,_make_loader,predict
from tcrr_stla.v4.cache import digest_model
from tcrr_stla.evaluation import summarize_by_family,save_summary
from scripts.build_retrieval_cache import ensure_cache

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--backbone-checkpoint')
    p.add_argument('--adapter-checkpoint')
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--severities',nargs='+',type=int,default=[20,40,60,90])
    p.add_argument('--evidence-mode',default='full',choices=('full','no_retrieval','no_evidence','no_normal_residual',
        'no_failure_residual','no_support_conditioning','shuffled_cf_pairing','no_pair_interaction',
        'no_utility_ranking','similarity_only','no_gain_harm','no_risk'))
    a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    if not str(ck.get('version','')).startswith('4.'):
        raise RuntimeError('evaluate_v4 requires a v4 checkpoint')
    cfg=relocate_saved_config(ck['config'])
    cfg['training']['batch_size']=int(a.batch_size)
    cfg['runtime']['num_workers']=0
    cfg['runtime']['replay_bank_on_gpu']=False
    cfg['runtime']['normal_teacher_on_gpu']=False
    mode=a.evidence_mode
    os.environ['TRAFFIDENT_TARGET']=cfg['experiment']['target']
    backbone=Path(a.backbone_checkpoint or ck['backbone_checkpoint'])
    if not backbone.is_file():raise FileNotFoundError(backbone)
    if file_sha256(backbone)!=ck['backbone_sha256']:raise RuntimeError('Backbone SHA mismatch')
    adapter=Path(a.adapter_checkpoint or Path(a.checkpoint).resolve().parent/'s0_shared_adapter.pt')
    if not adapter.is_file():raise FileNotFoundError(adapter)
    cfg['runtime']['teacher_adapter_checkpoint']=str(adapter)
    if ck.get('adapter_sha256') and file_sha256(adapter)!=ck['adapter_sha256']:
        raise RuntimeError('Adapter SHA mismatch')
    cfg['runtime']['teacher_fingerprint']=file_sha256(adapter)
    os.environ['BACKBONE_CHECKPOINT']=str(backbone)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model,bc=build_v4_model(cfg,backbone,device,adapter)
    model.load_state_dict(ck['state_dict'],strict=True)
    model.set_stage('joint')
    model.v4.reject_threshold=ck.get('calibration',{}).get('reject_threshold',None)
    model.eval()
    teacher=model.stla.eval()
    cfg['runtime']['teacher_fingerprint']=digest_model(teacher)
    normal=None
    memory=FutureMemory(cache_path(cfg,'formal'))
    if memory._source_masks is None:raise RuntimeError('Original valid support is required')
    if mode!='no_retrieval':
        normal=prepare_normal_teacher_bank(cfg,memory,teacher,backbone,device)
    mean=float(ck['mean']);std=float(ck['std']);rows=[]
    for severity in a.severities:
        cfg['runtime']['replay_context']=f's{severity}'
        source=cache_path(cfg,'evaluation',severity)
        retr=None if mode=='no_retrieval' else ensure_cache(cfg,source,'test',severity=severity)
        ds=SupportSplit(source,'test')
        batch=int(a.batch_size)
        while True:
            try:
                loader=_make_loader(ds,batch,False,cfg)
                p=predict(model,loader,RetrievalCache(retr) if retr else None,memory,device,mean,std,cfg,mode,
                          teacher=teacher if mode!='no_retrieval' else None,normal_bank=normal)
                break
            except torch.cuda.OutOfMemoryError:
                if batch==1:raise
                batch=max(1,batch//2)
                torch.cuda.empty_cache()
                print(f'V4_EVAL_OOM_RETRY severity={severity} batch={batch}',flush=True)
        perf=summarize_by_family(p['prediction'],p['true'],p['valid'],p['direct'],ds.meta)
        base=summarize_by_family(p['base'],p['true'],p['valid'],p['direct'],ds.meta)
        target=Path(a.checkpoint).resolve().parent/'evaluation'/f's{severity}'/mode
        target.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target/'predictions.npz',prediction=p['prediction'].astype('float32'),
                            base=p['base'].astype('float32'),truth=p['true'].astype('float32'),valid=p['valid'],direct=p['direct'])
        save_summary(target/'summary.json',dict(severity=severity,mode=mode,metrics=perf,base=base,
                     gate_mean=p['gate_mean'],useful_candidate_recall=p['useful_candidate_recall'],
                     weighted_candidate_utility_normalized=p['weighted_candidate_utility_normalized'],
                     oracle_candidate_gain_normalized=p['oracle_candidate_gain_normalized'],
                     realized_gain_normalized=p['realized_gain_normalized'],
                     harm_rate=p['harm_rate'],correction_coverage=p['correction_coverage']))
        row={'severity':severity,'mode':mode,**perf['ALL'],
             **{'base_'+k:v for k,v in base['ALL'].items()},
             'oracle_gain_normalized':p['oracle_candidate_gain_normalized'],
             'realized_gain_normalized':p['realized_gain_normalized'],
             'harm_rate':p['harm_rate'],'coverage':p['correction_coverage']}
        rows.append(row)
        print('V4_EVAL '+json.dumps(row),flush=True)
    root=Path(a.checkpoint).resolve().parent/'evaluation'
    pd.DataFrame(rows).to_csv(root/f'severity_summary.{mode}.csv',index=False)
    if mode=='full' and ck['config']['training']['ablation']=='full':
        pd.DataFrame(rows).to_csv(root/'severity_summary.csv',index=False)
        summary_path=Path(a.checkpoint).resolve().parent/'summary.json'
        existing=json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        existing.update(test_evaluated=True,evaluation_summary=rows)
        save_summary(summary_path,existing)
    print(f'V4_EVALUATION_PASS mode={mode} folder={root}',flush=True)
if __name__=='__main__':main()
