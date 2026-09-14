#!/usr/bin/env python3
from pathlib import Path
import argparse,sys,numpy as np
ROOT=Path('/scratch/mh20/zw1165/Speed_LLM_RAG/PreSuRe'); ROOT=ROOT if ROOT.exists() else Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from presure.utils import load_yaml,write_json
from presure.contract import assert_cache_ready,cache_dir
from presure.data import CacheSplit
from presure.features import build_features,make_retrieval_keys,apply_key_norm
from presure.retrieval import search_ip,filter_candidates
from presure.pipeline import art_dir
from presure.regions import spectral_graph_regions, region_manifest

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--stage',default='development',choices=['development','formal']); ap.add_argument('--force',action='store_true'); a=ap.parse_args()
 cfg=load_yaml(ROOT/'configs/presure.yaml'); ident=assert_cache_ready(cfg,a.stage); out=art_dir(cfg,a.stage); cache=cache_dir(cfg,a.stage)
 out.mkdir(parents=True,exist_ok=True)
 if not (out/'backbone_manifest.json').exists(): raise RuntimeError('Run prepare_backbone_artifacts first')
 rc=cfg['retrieval']; tr=CacheSplit(cache,'train'); va=CacheSplit(cache,'val'); tb=np.load(out/'train_base_pred.npy',mmap_mode='r'); vb=np.load(out/'val_base_pred.npy',mmap_mode='r')
 # Graph-only routing regions: fixed from SupportBench adjacency, never from Val/Test targets.
 if str(Path(cfg['project']['supportbench_root'])) not in sys.path: sys.path.insert(0,str(Path(cfg['project']['supportbench_root'])))
 from supportbench.dataset import SupportBench
 sb=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
 rlabels,rmethod=spectral_graph_regions(sb.adj,int(cfg['local_routing']['n_regions']),int(cfg['local_routing']['region_seed']))
 np.save(out/'node_regions.npy',rlabels.astype(np.int32)); write_json(out/'region_manifest.json',region_manifest(rlabels,rmethod,sb.adj))
 print(f'GRAPH REGIONS PASS method={rmethod} n_regions={int(rlabels.max())+1} sizes={[int((rlabels==i).sum()) for i in range(int(rlabels.max())+1)]}',flush=True)
 kwargs=dict(node_proj_dim=int(rc['feature_node_proj_dim']),mask_proj_dim=int(rc['mask_node_proj_dim']),seed=int(rc['feature_seed']))
 tf=build_features(tr.x,tr.obs,tb,tr.tod,tr.dow,**kwargs); vf=build_features(va.x,va.obs,vb,va.tod,va.dow,**kwargs); np.save(out/'train_features.npy',tf); np.save(out/'val_features.npy',vf)
 mu=tf.mean(0,keepdims=True).astype(np.float32); sd=(tf.std(0,keepdims=True)+1e-5).astype(np.float32); np.savez(out/'feature_norm.npz',mean=mu,std=sd)
 H=int(cfg['experiment']['history']); P=int(cfg['experiment']['horizon']); mk=float(rc['mask_feature_weight']); keys,kmu,ksd=make_retrieval_keys(tf,H,P,kwargs['node_proj_dim'],kwargs['mask_proj_dim'],mk); vkeys=apply_key_norm(vf,kmu,ksd,H,P,kwargs['node_proj_dim'],kwargs['mask_proj_dim'],mk); np.savez(out/'retrieval_key_norm.npz',mean=kmu,std=ksd); np.save(out/'train_retrieval_keys.npy',keys); np.save(out/'val_retrieval_keys.npy',vkeys)
 k=int(rc['coarse_k']); search_k=min(len(tr),k+int(rc['search_overfetch']))
 oracle_k=min(int(rc.get('oracle_expanded_k',k)),len(tr)); oracle_search_k=min(len(tr),oracle_k+int(rc.get('oracle_search_overfetch',rc['search_overfetch'])))
 ti,ts,backend=search_ip(keys,keys,search_k)
 # One larger validation search serves both deployable coarse candidates and diagnostic Oracle-B/C.
 vi_full,vs_full,_=search_ip(keys,vkeys,max(search_k,oracle_search_k))
 ti,ts=filter_candidates(ti,ts,tr.meta,tr.meta,k,int(rc['exclude_interval_steps']),True)
 vi,vs=filter_candidates(vi_full,vs_full,va.meta,tr.meta,k,int(rc['exclude_interval_steps']),False)
 voi,vos=filter_candidates(vi_full,vs_full,va.meta,tr.meta,oracle_k,int(rc['exclude_interval_steps']),False)
 np.savez_compressed(out/'train_candidates.npz',indices=ti,scores=ts); np.savez_compressed(out/'val_candidates.npz',indices=vi,scores=vs); np.savez_compressed(out/'val_oracle_candidates.npz',indices=voi,scores=vos)
 write_json(out/'retrieval_manifest.json',{
  'stage':a.stage,'cache_identity':ident,'backend':backend,'coarse_k':k,'search_k':search_k,
  'oracle_expanded_k':oracle_k,'oracle_search_k':oracle_search_k,'train_rows':len(tr),'val_rows':len(va),
  'memory_split':'train','same_group_excluded':True,'interval_exclusion_steps':int(rc['exclude_interval_steps']),
  'query_future_used':False,'test_instantiated':False,
  'oracle_candidates_validation_only':True,
  'graph_regions_file':'node_regions.npy','region_manifest_file':'region_manifest.json',
 })
 print(f'RETRIEVAL CANDIDATES PASS coarse_k={k} oracle_expanded_k={oracle_k} backend={backend}')
if __name__=='__main__': main()
