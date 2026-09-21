#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,json
import numpy as np
import torch

ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml,write_json,sha256_file,log
from presure.contract import assert_cache_ready,cache_dir
from presure.backbone import load_patchstg_adapter,infer_split_with_explicit_hidden
from presure.data import CacheSplit,cache_stats
from presure.paths import artifact_dir
from presure.retrieval import search_ip,filter_candidates
from presure.features import row_normalize_adjacency,build_neutral_coarse_descriptor,build_evidence_descriptor,build_backbone_descriptor,fit_standardizer,apply_standardizer,build_node_dual_features_batch


def cosine_keys(tr,va):
    mu=tr.mean(0,keepdims=True); sd=tr.std(0,keepdims=True)+1e-5; a=(tr-mu)/sd; b=(va-mu)/sd; a=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-8); b=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-8); return a.astype(np.float32),b.astype(np.float32)


def standardized_pair(tr,va,clip):
    mu,sd,clip=fit_standardizer(tr,clip); return apply_standardizer(tr,mu,sd,clip),apply_standardizer(va,mu,sd,clip)


def write_node_dual(out_prefix,ds,base,hidden,adj,mean,std,batch=64):
    state_mm=ev_mm=None; sdim=edim=None
    for a in range(0,len(ds),int(batch)):
        z=slice(a,min(len(ds),a+int(batch))); hh=None if hidden is None else np.asarray(hidden[z],np.float32)
        state,ev=build_node_dual_features_batch(np.asarray(ds.x[z],np.float32),np.asarray(ds.obs[z],bool),np.asarray(base[z],np.float32),hh,np.asarray(ds.tod[z]),np.asarray(ds.dow[z]),adj,mean,std)
        if state_mm is None:
            sdim=state.shape[-1]; edim=ev.shape[-1]
            state_mm=np.lib.format.open_memmap(str(out_prefix)+'_state.npy',mode='w+',dtype=np.float16,shape=(len(ds),state.shape[1],sdim))
            ev_mm=np.lib.format.open_memmap(str(out_prefix)+'_evidence.npy',mode='w+',dtype=np.float16,shape=(len(ds),ev.shape[1],edim))
        state_mm[z]=state.astype(np.float16); ev_mm[z]=ev.astype(np.float16)
        if a==0 or a%(int(batch)*50)==0: log(f'NODE FEATURES {Path(str(out_prefix)).name} {a}/{len(ds)} state={state.shape} evidence={ev.shape}')
    state_mm.flush(); ev_mm.flush(); return int(sdim),int(edim)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--checkpoint',default=None); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
    cfg=load_yaml(ROOT/'configs/presure_v12.yaml'); ident=assert_cache_ready(cfg,a.stage); out=artifact_dir(cfg,a.stage); out.mkdir(parents=True,exist_ok=True); ready=out/'READY.json'
    if ready.exists() and not a.force:
        old=json.loads(ready.read_text())
        required=[out/'train_base_pred.npy',out/'val_base_pred.npy',out/'train_candidates.npz',out/'val_candidates.npz',out/'train_desc_dual.npy',out/'val_desc_dual.npy',out/'train_node_state.npy',out/'val_node_state.npy',out/'train_node_evidence.npy',out/'val_node_evidence.npy']
        # v1.2.1 changes only forecaster attention execution; v1.2.0 artifacts are byte-contract compatible.
        if old.get('version') in {cfg['version'],'1.2.0'} and all(x.exists() for x in required):
            log(f'V1.2 artifacts already ready: {out} artifact_version={old.get("version")} compatible_with={cfg["version"]}')
            return
    cache=cache_dir(cfg,a.stage); tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    adapter,ckpath,ck,source=load_patchstg_adapter(cfg,a.stage,device,a.checkpoint); ac=cfg['artifacts']; bs=int(ac['backbone_batch_size']); mod=str(ac['explicit_hidden_module'])
    if device.type=='cuda':
        p=torch.cuda.get_device_properties(device); log(f'GPU {p.name} total_memory={p.total_memory/1024**3:.2f}GB backbone_batch={bs}')
    tb,th,treason=infer_split_with_explicit_hidden(adapter,cache,'train',bs,device,mod,cfg.get('runtime',{})); vb,vh,vreason=infer_split_with_explicit_hidden(adapter,cache,'val',bs,device,mod,cfg.get('runtime',{})); hidden_available=th is not None and vh is not None
    if not hidden_available and bool(ac.get('hidden_required',False)): raise RuntimeError(f'Explicit hidden required but unavailable train={treason} val={vreason}')
    dtype=np.float16 if str(ac.get('storage_dtype','float16'))=='float16' else np.float32; np.save(out/'train_base_pred.npy',tb.astype(dtype)); np.save(out/'val_base_pred.npy',vb.astype(dtype))
    if hidden_available: np.save(out/'train_backbone_hidden.npy',th.astype(dtype)); np.save(out/'val_backbone_hidden.npy',vh.astype(dtype)); log(f'EXPLICIT HIDDEN {mod} train={th.shape} val={vh.shape}')
    else: log(f'WARN hidden unavailable; dual query uses PatchSTG forecast as state representation. train={treason} val={vreason}')
    sroot=Path(cfg['project']['supportbench_root']); sys.path.insert(0,str(sroot)) if str(sroot) not in sys.path else None
    from supportbench.dataset import SupportBench
    sb=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root']); adj=row_normalize_adjacency(sb.adj); np.save(out/'graph_adj.npy',adj.astype(np.float32)); mean,std=cache_stats(cache)
    pc=cfg['candidate_pool']; batch=int(ac['descriptor_batch_size']); seed=int(pc['feature_seed'])
    c_tr=build_neutral_coarse_descriptor(tr.x,tr.obs,tr.tod,tr.dow,int(pc['coarse_node_proj_dim']),int(pc['coarse_mask_proj_dim']),seed,batch); c_va=build_neutral_coarse_descriptor(va.x,va.obs,va.tod,va.dow,int(pc['coarse_node_proj_dim']),int(pc['coarse_mask_proj_dim']),seed,batch); kt,kv=cosine_keys(c_tr,c_va)
    k=int(pc['candidate_k']); sk=min(len(tr),k+int(pc['search_overfetch'])); ti0,ts0,backend=search_ip(kt,kt,sk); vi0,vs0,_=search_ip(kt,kv,sk); ti,ts=filter_candidates(ti0,ts0,tr.meta,tr.meta,k,int(pc['exclude_interval_steps']),True); vi,vs=filter_candidates(vi0,vs0,va.meta,tr.meta,k,int(pc['exclude_interval_steps']),False)
    np.savez_compressed(out/'train_candidates.npz',indices=ti,scores=ts); np.savez_compressed(out/'val_candidates.npz',indices=vi,scores=vs); log(f'CANDIDATE POOL K0={k} backend={backend} train={ti.shape} val={vi.shape}')
    rc=cfg['representations']; clip=float(rc['descriptor_clip'])
    ev_tr=build_evidence_descriptor(tr.x,tr.obs,tr.tod,tr.dow,adj,int(rc['evidence_node_proj_dim']),int(rc['evidence_mask_proj_dim']),seed+30,batch); ev_va=build_evidence_descriptor(va.x,va.obs,va.tod,va.dow,adj,int(rc['evidence_node_proj_dim']),int(rc['evidence_mask_proj_dim']),seed+30,batch); ev_tr,ev_va=standardized_pair(ev_tr,ev_va,clip)
    bb_tr=build_backbone_descriptor(tb,th if hidden_available else None,mean,std,int(rc['backbone_node_proj_dim']),int(rc['hidden_node_proj_dim']),seed+40,batch); bb_va=build_backbone_descriptor(vb,vh if hidden_available else None,mean,std,int(rc['backbone_node_proj_dim']),int(rc['hidden_node_proj_dim']),seed+40,batch); bb_tr,bb_va=standardized_pair(bb_tr,bb_va,clip)
    dual_tr=np.concatenate([bb_tr,ev_tr],1); dual_va=np.concatenate([bb_va,ev_va],1); np.save(out/'train_desc_dual.npy',dual_tr.astype(np.float16)); np.save(out/'val_desc_dual.npy',dual_va.astype(np.float16)); log(f'DUAL QUERY DESCRIPTOR train={dual_tr.shape} val={dual_va.shape}')
    nfb=int(ac['node_feature_batch_size']); sdim,edim=write_node_dual(out/'train_node',tr,tb,th if hidden_available else None,adj,mean,std,nfb); vsdim,vedim=write_node_dual(out/'val_node',va,vb,vh if hidden_available else None,adj,mean,std,nfb)
    if (sdim,edim)!=(vsdim,vedim): raise RuntimeError('Train/Val node feature dimension mismatch')
    write_json(ready,{'version':cfg['version'],'stage':a.stage,'cache_identity':ident,'checkpoint':str(ckpath),'checkpoint_sha256':sha256_file(ckpath),'checkpoint_source':source,'checkpoint_seed':int(ck['seed']),'candidate_k':k,'top_k':int(pc['top_k']),'retrieval_backend':backend,'train_rows':len(tr),'val_rows':len(va),'nodes':int(sb.N),'speed_mean':mean,'speed_std':std,'hidden_module':mod,'hidden_available':hidden_available,'hidden_train_reason':treason,'hidden_val_reason':vreason,'dual_descriptor_dim':int(dual_tr.shape[1]),'state_node_dim':sdim,'evidence_node_dim':edim,'query_future_used_in_artifacts':False})
    log('PREPARE PRESURE V1.2 ARTIFACTS PASS')
if __name__=='__main__': main()
