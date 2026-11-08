#!/usr/bin/env python3
"""TCRR v3 frozen-checkpoint, same-candidate C1/C2/C3 One-Shot.
Full and CF/support/gate interventions share exactly one Query STLA forward,
retrieval list and matched Teacher replay per batch. This is inference-only.
"""
from __future__ import annotations
import argparse,copy,json,os,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tcrr_stla.config import cache_path,relocate_saved_config,file_sha256
from tcrr_stla.data import SupportSplit,RetrievalCache
from tcrr_stla.evidence import FutureMemory
from tcrr_stla.normal_teacher import prepare_normal_teacher_bank
from tcrr_stla.trainer import build_model,_make_loader,_batch_to_device,evidence_for_batch
from scripts.build_retrieval_cache import ensure_cache
from tcrr_stla.diagnostics_v3.common import (SCHEMA,ProtocolDigest,np_cpu,add_mode,add_horizon,save_severity,consolidate,atomic_json,sha256,reproduce_official,protocol_query_rows)

SEV=(20,40,60,90)
CF_MODES=('no_normal_residual','no_failure_residual','shuffled_cf_pairing','no_support_conditioning','no_gain_harm','no_risk','no_evidence')

def checkpoint_run(root,target):
    pattern=f'TCRR-STLA_exps/{target}/incomplete/TraffiDent_Net_06/formal/*_seed10_full_M3/best.pt'
    matches=sorted(root.glob(pattern),key=lambda p:p.stat().st_mtime,reverse=True)
    if not matches:raise FileNotFoundError(f'No TCRR M3 checkpoint for {target}: {pattern}')
    if len(matches)>1:print('CHECKPOINT_CANDIDATES',*[str(x) for x in matches],sep='\n',flush=True)
    return matches[0]

def discover_family(ds,idx):
    return ds.meta['family'].astype(str).iloc[idx].tolist() if 'family' in ds.meta else ['unknown']*len(idx)

def train_only_cf_reference_probe(cfg,model,teacher,normal_bank,mem,backbone,device,args,target,output):
    """Optional supervised mechanism audit on TRAIN cases ONLY.

    The target is a paired frozen-Teacher full-vs-masked response. The complete
    query history is read only from the Train dataset. No Test-full history is
    ever accessed or exported. Numbers do not represent held-out Test accuracy.
    """
    if args.cf_train_queries<=0:return {'status':'SKIPPED_BY_CONFIG'}
    source=cache_path(cfg,'formal')
    ds=SupportSplit(source,'train',limit=args.cf_train_queries,cf_training=True)
    rc_path=ensure_cache(cfg,source,'train')
    rc=RetrievalCache(rc_path)
    dl=_make_loader(ds,args.batch_size,False,cfg)
    rows=[]
    with torch.inference_mode():
        for bi,bb in enumerate(dl,1):
            b=_batch_to_device(bb,device)
            ev=evidence_for_batch(model,b,rc,mem,device,cfg,'full',teacher=teacher,normal_bank=normal_bank)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(cfg['runtime'].get('amp',False)) and device.type=='cuda'):
                o=model(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long(),mode='full',**ev)
                teacher_full=teacher(b['x_full_training_only'].float(),torch.ones_like(b['mask']),b['tod'].long(),b['dow'].long())[0]
                teacher_missing=teacher(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long())[0]
            label=(teacher_full.float()-teacher_missing.float()).cpu().numpy()
            estimate=np_cpu(o['cf_response']);valid=b['future_valid'].bool().cpu().numpy()
            for i in range(len(label)):
                select=valid[i];
                if not select.any():continue
                rows.append({'query_idx':int(bb['index'][i]),'window_start':int(bb['window_start'][i]),
                    'support_coverage':float(b['mask'][i].float().mean()),
                    'cf_label_abs_mean':float(np.abs(label[i][select]).mean()),
                    'cf_estimate_abs_mean':float(np.abs(estimate[i][select]).mean()),
                    'cf_recovery_mae_normalized':float(np.abs(estimate[i][select]-label[i][select]).mean()),
                    'cf_zero_baseline_mae_normalized':float(np.abs(label[i][select]).mean())})
            if bi%10==0 or bi==len(dl):print(f'TCRR_CF_TRAIN_PROBE {target} {bi}/{len(dl)}',flush=True)
    out=Path(output)/'cf_reference_train_only.csv'
    pd.DataFrame(rows).to_csv(out,index=False)
    result={'status':'PASS','split':'train','note':'Teacher paired-response reference from complete TRAIN history; NOT Test, not independently held out',
            'queries':len(rows),'mean_cf_recovery_mae_normalized':float(pd.DataFrame(rows)['cf_recovery_mae_normalized'].mean()),
            'mean_cf_zero_baseline_mae_normalized':float(pd.DataFrame(rows)['cf_zero_baseline_mae_normalized'].mean())}
    atomic_json(Path(output)/'cf_reference_train_only.json',result)
    return result


def run_target(args,target,checkpoint):
    cp=Path(checkpoint).resolve();ck=torch.load(cp,map_location='cpu',weights_only=False)
    if ck.get('version')!='3.0.0' or ck['config'].get('v3',{}).get('contribution')!='M3':raise RuntimeError('Requires trained TCRR v3 M3 best.pt')
    cfg=relocate_saved_config(ck['config']);cfg['training']['batch_size']=args.batch_size
    cfg['runtime'].update(num_workers=0,persistent_workers=False,replay_bank_on_gpu=False,normal_teacher_on_gpu=False,preload_retrieval_memory_gpu=False)
    if cfg['experiment']['target']!=target:raise RuntimeError('Checkpoint target mismatch')
    os.environ['TRAFFIDENT_TARGET']=target
    backbone=Path(ck['backbone_checkpoint'])
    if not backbone.is_file():raise FileNotFoundError(backbone)
    if ck.get('backbone_sha256') and file_sha256(backbone)!=ck['backbone_sha256']:raise RuntimeError('Backbone SHA256 mismatch')
    os.environ['BACKBONE_CHECKPOINT']=str(backbone)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type!='cuda' and not args.allow_cpu:raise RuntimeError('GPU required; submit PBS (or use --allow-cpu intentionally)')
    model,_=build_model(cfg,backbone,device);model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    # Frozen teacher must be the PRETRAINED backbone, never the updated query model.
    teacher,_=build_model(cfg,backbone,device);teacher=teacher.stla.eval()
    for p in teacher.parameters():p.requires_grad_(False)
    mem=FutureMemory(cache_path(cfg,'formal'))
    bank=prepare_normal_teacher_bank(cfg,mem,teacher,backbone,device)
    root=Path(cfg['project']['root'])/'TCRR-STLA_exps'/'diagnostics_v3'/cfg['experiment']['dataset']/f'seed{cfg["experiment"]["seed"]}'/target
    root.mkdir(parents=True,exist_ok=True)
    hash_cp=sha256(cp)
    for sev in args.severities:
        outdir=root/f's{sev}'
        # Each severity writes protocol_queries.csv before save_severity() runs.
        # Creating only the target-level directory is insufficient on Gadi.
        outdir.mkdir(parents=True, exist_ok=True)
        done=outdir/'manifest.json'
        ds=SupportSplit(cache_path(cfg,'evaluation',sev),'test',limit=(args.max_queries or None))
        if done.is_file() and not args.force:
            old=json.loads(done.read_text())
            if (old.get('status')=='PASS' and old.get('checkpoint_sha256')==hash_cp and
                old.get('query_count')==len(ds) and old.get('modes')==['full',*CF_MODES,'similarity_only_frozen_gate']):
                print(f'TCRR_RESUME_SKIP target={target} severity={sev}',flush=True);continue
        rc_path=ensure_cache(cfg,ds.root,'test',severity=sev)
        rc=RetrievalCache(rc_path)
        loader=_make_loader(ds,args.batch_size,False,cfg)
        prot=ProtocolDigest();proto_rows=[];base_digest=__import__('hashlib').sha256();rows=[];hz=[];mechanisms=[];start=time.time()
        # Capture the real hidden feature tensor from the SINGLE query STLA forward.
        capture=[]
        def keep(module,input,output):capture.append(output[1])
        hook=model.stla.register_forward_hook(keep)
        with torch.inference_mode():
            try:
                for batch_i,bb in enumerate(loader,1):
                    prot.update(bb);b=_batch_to_device(bb,device)
                    ev=evidence_for_batch(model,b,rc,mem,device,cfg,'full',normal_bank=bank,teacher=teacher)
                    capture.clear()
                    with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=bool(cfg['runtime'].get('amp',False)) and device.type=='cuda'):
                        full=model(b['x'].float(),b['mask'].float(),b['tod'].long(),b['dow'].long(),mode='full',**ev)
                    if len(capture)!=1:raise RuntimeError(f'Expected one STLA call, got {len(capture)}')
                    z=model.latent_proj(capture[0]).float();capture.clear()
                    bn=np_cpu(full['base_prediction']);y=np_cpu(b['y_raw']);v=b['future_valid'].cpu().numpy();d=b['direct_mask'].cpu().numpy()
                    family=discover_family(ds,bb['index'].numpy().astype(int))
                    ext={'family':family}
                    pred=np_cpu(full['prediction'])*float(ck['std'])+float(ck['mean'])
                    base=bn*float(ck['std'])+float(ck['mean'])
                    base_digest.update(np.asarray(base,dtype='<f4').tobytes(order='C'))
                    proto_rows.extend(protocol_query_rows(bb,base,sev))
                    add_mode(rows,bb,base,y,v,d,pred,'full',sev,ext)
                    add_horizon(hz,bb,base,y,v,d,pred,sev,'full',family)
                    # C1/C2/C3 interventions recompute v3 transfer *including* downstream ranks/gates.
                    for mode in CF_MODES:
                        o=model.v3(b['x'],b['mask'],full['base_prediction'].float(),z,
                            ev['normal_residual'],ev['residual'],ev['candidate_y'],
                            ev['candidate_scores'],ev['candidate_coverage'],ev['candidate_valid'],
                            ev['candidate_histories'],model.traffic.neighbors,mode=mode)
                        p=np_cpu(o['prediction'])*float(ck['std'])+float(ck['mean'])
                        add_mode(rows,bb,base,y,v,d,p,mode,sev,ext)
                        if mode in ('no_normal_residual','no_failure_residual','no_support_conditioning','no_gain_harm'):
                            add_horizon(hz,bb,base,y,v,d,p,sev,mode,family)
                    # Similarity-only fixed-gate baseline: same candidates, None logits,
                    # final scale/gate from FULL. Diagnostic proxy, not a retrained C2 ablation.
                    K=full['candidate_residuals'].shape[1]
                    sim=ev['candidate_scores'][:,:,None,None].float().expand_as(full['candidate_residuals'])
                    sim=sim.masked_fill(~full['candidate_available'].bool(),-1e4)
                    null=full['candidate_logits'][:,-1:].float()
                    sw=torch.softmax(torch.cat([sim,null],1),1)[:,:K]
                    evidence=(sw*full['candidate_residuals'].float()).sum(1)
                    bound=float(full['residual_bound'])
                    sim_correction=full['gate'].float()*bound*torch.tanh(evidence/bound)
                    sim_pred=np_cpu(full['base_prediction']+sim_correction)*float(ck['std'])+float(ck['mean'])
                    add_mode(rows,bb,base,y,v,d,sim_pred,'similarity_only_frozen_gate',sev,ext)
                    add_horizon(hz,bb,base,y,v,d,sim_pred,sev,'similarity_only_frozen_gate',family)
                    # Per-query realized Gain/Harm and gated/bounded candidate oracle (hindsight only).
                    gate=np_cpu(full['gate']);w=np_cpu(full['candidate_weights']);norm=np_cpu(full['normal_residual']);failure=np_cpu(full['failure_residual'])
                    eg=np_cpu(full['estimated_gain']);eh=np_cpu(full['estimated_harm']);uc=np_cpu(full['estimated_uncertainty'])
                    cf=np_cpu(full['cf_response']);corr=np_cpu(full['correction'])
                    limit=float(full['residual_bound']);cands=np_cpu(full['candidate_residuals'])
                    avail=full['candidate_available'].cpu().numpy().astype(bool)
                    # Same bounded *single-candidate* application; labels used only after inference.
                    proposal=limit*np.tanh(cands/limit)*float(ck['std'])
                    gain=np.abs(base[:,None]-y[:,None])-np.abs(base[:,None]+proposal-y[:,None])
                    gain=np.where(avail & v[:,None],gain,-np.inf)
                    oracle=np.maximum(0,np.max(gain,axis=1))
                    best_idx=np.argmax(gain,axis=1)
                    selected=np.take_along_axis(gain,best_idx[:,None],axis=1)[:,0]
                    # Learned highest-weight candidate (unbounded by its actual chosen None weight).
                    learned_idx=np.argmax(w[:,:cands.shape[1]],axis=1)
                    learned_gain=np.take_along_axis(gain,learned_idx[:,None],axis=1)[:,0]
                    weighted_gain=np.where(np.isfinite(gain),gain,0)*w[:,:cands.shape[1]]
                    real=np.abs(base-y)-np.abs(pred-y)
                    for ii in range(len(family)):
                        ok=v[ii] & np.isfinite(real[ii]); c=ok.sum()
                        if not c:continue
                        def avg(a):return float(np.mean(a[ii][ok]))
                        mechanisms.append({'severity':sev,'family':family[ii],'window_start':int(bb['window_start'][ii]),
                            'support_coverage':float(np.mean(np_cpu(b['mask'][ii]))),
                            'gate_mean':avg(gate),'none_weight':avg(w[:,-1]),
                            'failure_abs':float(np.mean(np.abs(failure[ii]))),'normal_abs':float(np.mean(np.abs(norm[ii]))),
                            'cf_response_abs':avg(np.abs(cf)),'correction_abs':avg(np.abs(corr)),
                            'estimated_gain':avg(eg),'estimated_harm':avg(eh),'estimated_uncertainty':avg(uc),
                            'realized_gain':avg(real),'oracle_single_candidate_gain':avg(oracle),
                            'best_candidate_mean_gain':avg(np.maximum(selected,0)),
                            'top_model_candidate_gain':float(np.mean(np.where(np.isfinite(learned_gain[ii][ok]),learned_gain[ii][ok],0))),
                            'top_model_candidate_positive_rate':float(np.mean(learned_gain[ii][ok]>0)),
                            'weighted_candidate_gain':avg(weighted_gain.sum(axis=1)),
                            'candidate_available_rate':float(np.mean(avail[ii].any(axis=0)[ok])),
                            'hindsight_note':'labels used for offline oracle only; similarity proxy reuses full gate'})
                    if batch_i%args.print_every==0 or batch_i==len(loader):
                        print(f'TCRR_DIAG target={target} s{sev} batches={batch_i}/{len(loader)} elapsed={time.time()-start:.1f}s',flush=True)
            finally:hook.remove()
        reproduction=reproduce_official(cp.parent,sev,rows)
        if reproduction['status']=='MISMATCH':
            print(f'TCRR_OFFICIAL_REPRO_MISMATCH target={target} s{sev} details={reproduction}',flush=True)
        manifest={'status':('REPRO_MISMATCH' if reproduction['status']=='MISMATCH' else 'PASS'),'schema':SCHEMA,'method':'TCRR','target':target,'severity':sev,
                  'query_count':len(ds),'official_reproduction':reproduction,'checkpoint_sha256':hash_cp,'backbone_sha256':sha256(backbone),
                  'protocol':prot.result(),'base_raw_sha256':base_digest.hexdigest(),'modes':['full',*CF_MODES,'similarity_only_frozen_gate'],
                  'c1_cf_test_target':'NOT_MEASURED: full Test history forbidden',
                  'independent_c123_ablation':'NOT_MEASURED: M0/M1/M2 require independent training',
                  'oracle':'hindsight per forecast element; never used in model forward',
                  'elapsed_seconds':round(time.time()-start,1)}
        pd.DataFrame(proto_rows).to_csv(outdir/'protocol_queries.csv',index=False)
        save_severity(outdir,rows,hz,mechanisms,manifest)
        print(f'TCRR_SEVERITY_PASS {target} s{sev} {len(ds)}',flush=True)
    cf_status=train_only_cf_reference_probe(cfg,model,teacher,bank,mem,backbone,device,args,target,root)
    report=consolidate(root,args.severities)
    atomic_json(root/'summary.json',{'schema':SCHEMA,'method':'TCRR','target':target,'checkpoint':str(cp),'report':report,'cf_train_reference':cf_status})
    bad=[int(sev) for sev in args.severities if json.loads((root/f's{sev}'/'manifest.json').read_text()).get('status')!='PASS']
    if bad:raise RuntimeError('Full M3 evaluation reproduction mismatch in severities: '+str(bad)+'; inspect manifests before comparing models')
    print(f'TCRR_TARGET_PASS target={target} output={root}',flush=True)
    return root

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--target',required=True,choices=('speed','flow'))
    ap.add_argument('--checkpoint');ap.add_argument('--batch-size',type=int,default=16)
    ap.add_argument('--severities',type=int,nargs='+',default=list(SEV));ap.add_argument('--max-queries',type=int,default=0)
    ap.add_argument('--cf-train-queries',type=int,default=256)
    ap.add_argument('--print-every',type=int,default=40);ap.add_argument('--force',action='store_true');ap.add_argument('--allow-cpu',action='store_true')
    a=ap.parse_args();os.environ['TRAFFIDENT_TARGET']=a.target
    root=Path(os.environ.get('SPEED_LLM_RAG_ROOT',ROOT.parent)).resolve()
    cp=Path(a.checkpoint) if a.checkpoint else checkpoint_run(root,a.target)
    run_target(a,a.target,cp)
if __name__=='__main__':main()
