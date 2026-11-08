#!/usr/bin/env python3
"""Four-severity blind evaluation for a frozen v9.2 checkpoint.

No TEST label is used in retrieval, replay, alignment, utility or selection.
Labels only appear after predictions, for reporting. Refuses to overwrite TEST.
"""
from __future__ import annotations
import argparse,os,sys,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from torch.utils.data import DataLoader
from tcrr_stla.config import load_config,cache_path,file_sha256
from tcrr_stla.v5.shared_adapter import initial_model
from tcrr_stla.v92.prepare import prepare_view,V9EvidenceDataset
from tcrr_stla.v92.training import to_device,forward_batch
from tcrr_stla.v92.diagnostics import load_checkpoint_model
from tcrr_stla.v92 import VERSION
from scripts.build_retrieval_cache import ensure_cache

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True);p.add_argument('--run-identity',required=True)
    p.add_argument('--severity',required=True,type=int,choices=(20,40,60,90))
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--prep-batch',type=int,default=4)
    p.add_argument('--smoke',action='store_true');p.add_argument('--force-replay',action='store_true')
    a=p.parse_args()
    info=json.loads(Path(a.run_identity).read_text())
    if info['version']!=VERSION:raise RuntimeError('v9.2 run identity required')
    os.environ['TRAFFIDENT_TARGET']=info['target']
    cfg=load_config(ROOT/'configs/tcrr_stla_v5.yaml')
    cfg['experiment'].update(target=info['target'],dataset=info['dataset'],seed=info['seed'])
    cfg['project']['retrieval_root']=str(Path(cfg['project']['retrieval_root'])/'v9_2_isolated')
    formal=cache_path(cfg,'formal');evaluation=cache_path(cfg,'evaluation',a.severity)
    ckfile=Path(info['backbone_checkpoint']);adapter=Path(info['adapter_checkpoint'])
    os.environ['BACKBONE_CHECKPOINT']=str(ckfile)
    if (file_sha256(ckfile)!=info['backbone_sha256'] or file_sha256(adapter)!=info['adapter_sha256']):
        raise RuntimeError('Frozen teacher / adapter has changed')
    if bool(a.smoke)!=bool(info['smoke']):raise ValueError('Smoke/full evaluation identity mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not a.smoke:raise RuntimeError('Formal severity test requires CUDA')
    model,checkpoint=load_checkpoint_model(a.checkpoint,device)
    if checkpoint['seed']!=info['seed']:raise RuntimeError('Checkpoint seed mismatch')
    teacher,bc=initial_model(cfg,ckfile,device,adapter)
    teacher.eval();[p.requires_grad_(False) for p in teacher.parameters()]
    evidence=Path(info['evidence_root']);evidence.mkdir(parents=True,exist_ok=True)
    split_key=f'test_s{a.severity}';folder=evidence/f'{split_key}_view0'
    if a.force_replay and folder.exists():
        import shutil;shutil.rmtree(folder)
    shortlist=ensure_cache(cfg,evaluation,'test',severity=a.severity,smoke=a.smoke)
    prepare_view(evaluation,'test',0,teacher,device,evidence,shortlist,k=12,chunk=a.prep_batch,
        seed=info['seed'],teacher_sha=info['teacher_fingerprint'],history_cache=formal,
        split_key=split_key,limit=24 if a.smoke else None)
    del teacher
    if device.type=='cuda':torch.cuda.empty_cache()
    ds=V9EvidenceDataset(evaluation,'test',(0,),root=evidence,split_key=split_key)
    dl=DataLoader(ds,batch_size=a.batch_size,num_workers=0,pin_memory=device.type=='cuda')
    totals={name:{'mae':0.,'squared':0.,'masked':0.,'h12':0.,'count':0.,'masked_count':0.,'h12_count':0.}
            for name in ('base','normal','full')}
    harm=active=valid_total=0
    ss=float(bc['std']);mm=float(bc['mean'])
    with torch.inference_mode():
        for raw in dl:
            b=to_device(raw,device)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                out=forward_batch(model,b,enable_cf=checkpoint['stage']!='normal',
                  alignment_mode=checkpoint['cfg']['alignment_mode'],
                  graph_mode=checkpoint['cfg']['graph_mode'],
                  temporal_mode=checkpoint['cfg']['temporal_mode'],
                  null_mode=checkpoint['cfg']['null_mode'])
            y=b['y_raw'].float();fv=b['future_valid'].bool();direct=b['direct_mask'].bool()
            if direct.ndim==2:direct=direct[:,None,:].expand_as(fv)
            if direct.shape!=fv.shape:raise RuntimeError('Direct mask shape mismatch')
            direct=direct&fv
            errors={name:pred.float()*ss+mm-y for name,pred in
                    (('base',out['base']),('normal',out['normal_prediction']),('full',out['prediction']))}
            for name,err in errors.items():
                st=totals[name];sq=torch.where(fv,err.square(),0)
                st['mae']+=float(torch.where(fv,err.abs(),0).sum())
                st['squared']+=float(sq.sum())
                st['masked']+=float(torch.where(direct,err.abs(),0).sum())
                st['h12']+=float(torch.where(fv[:,-1:],err[:,-1:].abs(),0).sum())
                st['count']+=int(fv.sum());st['masked_count']+=int(direct.sum())
                st['h12_count']+=int(fv[:,-1:].sum())
            en=errors['normal'].abs();ef=errors['full'].abs()
            harm+=int(torch.where(fv,ef>en+1e-6,False).sum());valid_total+=int(fv.sum())
            active+=int(torch.where(fv,out['cf_increment'].abs()>1e-6,False).sum())
    result={}
    for name,v in totals.items():
        cnt=max(1,v['count'])
        result[name]={'mae':v['mae']/cnt,'rmse':(v['squared']/cnt)**.5,
              'directly_masked_mae':v['masked']/max(1,v['masked_count']),
              'H12_MAE':v['h12']/max(1,v['h12_count'])}
    result.update(version=VERSION,target=info['target'],dataset=info['dataset'],severity=a.severity,
          seed=info['seed'],stage=checkpoint['stage'],checkpoint=str(Path(a.checkpoint).resolve()),
          cf_gain_raw=result['normal']['mae']-result['full']['mae'],
          cf_harm_rate=harm/max(1,valid_total),cf_activation_rate=active/max(1,valid_total),
          improvement_vs_base_pct=100*(result['base']['mae']-result['full']['mae'])/max(1e-8,result['base']['mae']))
    target=Path(a.run_identity).parent/f'test_s{a.severity}.json'
    if target.exists():raise FileExistsError('TEST results are write-once; refusing to overwrite '+str(target))
    target.write_text(json.dumps(result,indent=2))
    print('V92_TEST '+json.dumps(result),flush=True)
if __name__=='__main__':main()
