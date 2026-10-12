#!/usr/bin/env python3
from pathlib import Path
import argparse,os,sys,json
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from mask_raft_stla.config import load_config,cache_path
from mask_raft_stla.retrieval import build_retrieval_cache
from mask_raft_stla.forecast_pool import prepare_forecast_pool
from mask_raft_stla.config import resolve_stla_checkpoint


def ensure_cache(cfg,source,split,severity=None,smoke=False,force=False):
    tag=f'eval_s{int(severity)}' if severity is not None else ('formal_smoke' if smoke else 'formal')
    out=Path(cfg['project']['retrieval_root'])/cfg['experiment']['dataset']/('v23_'+tag)/f'{split}.mask_raft_topk.npz'
    source=Path(source)
    formal=cache_path(cfg,'formal')
    def signature(p):
        st=Path(p).stat()
        return f'{st.st_size}:{st.st_mtime_ns}'
    expected_files=[source/f'{split}_x_obs_norm.npy',source/f'{split}_history_observed.npy',
                    source/f'{split}_manifest.csv',formal/'train_x_full_norm.npy',formal/'train_manifest.csv']
    expected={'file_signatures':{str(p):signature(p) for p in expected_files},
              'top_k':int(cfg['retrieval']['top_k']),'periods':[int(v) for v in cfg['retrieval']['periods']],
              'split':split,'severity':severity,'version':'2.3.0','dataset':cfg['experiment']['dataset']}
    meta=out.with_suffix('.json')
    if out.is_file() and meta.is_file() and not force:
        try:
            info=json.loads(meta.read_text());ok=all(info.get(k)==v for k,v in expected.items())
            if ok:
                with np.load(out) as z:
                    nq=len(np.load(Path(source)/f'{split}_x_obs_norm.npy',mmap_mode='r'))
                    if z['indices'].shape[0] == (min(nq,24) if smoke else nq):
                        return prepare_forecast_pool(cfg, formal, source, split, out, resolve_stla_checkpoint(cfg,None), torch.device('cuda' if torch.cuda.is_available() else 'cpu'), smoke=smoke)
        except (OSError,ValueError,KeyError):pass
    memory=cache_path(cfg,'formal');dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    build_retrieval_cache(source,split,out,cfg['retrieval']['periods'],int(cfg['retrieval']['top_k']),
                          float(cfg['retrieval']['coverage_gamma']),
                          12 if smoke else int(cfg['retrieval']['query_batch_size']),
                          128 if smoke else int(cfg['retrieval']['memory_chunk_size']),
                          int(cfg['retrieval']['min_valid_features']),dev,
                          limit=24 if smoke else None,memory_cache_dir=memory,memory_limit=256 if smoke else None)
    info=json.loads(meta.read_text());info.update(expected);meta.write_text(json.dumps(info,indent=2))
    return prepare_forecast_pool(cfg, formal, source, split, out, resolve_stla_checkpoint(cfg,None), dev, smoke=smoke)


def main():
    p=argparse.ArgumentParser();p.add_argument('--target',choices=['speed','flow']);p.add_argument('--split',choices=['train','val'],default='train')
    p.add_argument('--severity',type=int,choices=[20,40,60,90]);p.add_argument('--smoke',action='store_true');p.add_argument('--force',action='store_true');p.add_argument('--config')
    a=p.parse_args();
    if a.target:os.environ['TRAFFIDENT_TARGET']=a.target
    cfg=load_config(a.config);stage='evaluation' if a.severity is not None else 'formal';split='test' if a.severity is not None else a.split
    out=ensure_cache(cfg,cache_path(cfg,stage,a.severity),split,a.severity,a.smoke,a.force)
    print('RETRIEVAL_CACHE_PASS',out,flush=True)
if __name__=='__main__':main()
