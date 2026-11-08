"""Fixed-teacher mask-matched replay bank, split specific and Train-future-only.

Training masks are fixed in Base_SupportBench. A frozen STLA replay teacher makes
candidate residuals deterministic across epochs, so expensive teacher forward
passes can be amortized without reducing training queries or candidate counts.
This is an explicit method variant: cache is keyed to v2.3 and the precise forecast-conditioned shortlist.
"""
from __future__ import annotations
import json, os, time, hashlib
from pathlib import Path
import numpy as np
import torch
from .evidence import shortlist, counterfactual_residuals
from .config import file_sha256


def _file_key(path):
    path=Path(path)
    st=path.stat()
    return {'absolute_path':str(path.resolve()),'size':st.st_size,'mtime_ns':st.st_mtime_ns}

class ReplayBank:
    def __init__(self, path, meta, on_gpu=False,device='cpu'):
        self.path=Path(path)
        self.array=np.load(self.path,mmap_mode='r',allow_pickle=False)
        self.meta=meta
        self.gpu=None
        if on_gpu and torch.device(device).type=='cuda':
            try:
                d=torch.device(device)
                tensor=torch.empty(self.array.shape,dtype=torch.float16 if self.array.dtype==np.float16 else torch.float32,device=d)
                for start in range(0,len(self.array),256):
                    end=min(start+256,len(self.array))
                    tensor[start:end].copy_(torch.from_numpy(np.array(self.array[start:end],copy=True)).to(d))
                self.gpu=tensor
                print(f'REPLAY_BANK_GPU split={meta["split"]} GiB={tensor.numel()*tensor.element_size()/1024**3:.3f}',flush=True)
            except torch.cuda.OutOfMemoryError:
                self.gpu=None;torch.cuda.empty_cache()
                print(f'REPLAY_BANK_CPU_FALLBACK split={meta["split"]}',flush=True)
    def fetch(self,indices,device):
        if self.gpu is not None:
            return self.gpu[indices.long().to(self.gpu.device)].float()
        ids=indices.detach().cpu().numpy().astype(np.int64)
        return torch.from_numpy(np.array(self.array[ids],copy=True)).to(device,non_blocking=True).float()


def replay_bank_dir(cfg, smoke=False):
    """Version-isolated teacher evidence; smoke never shares the formal bank."""
    return (Path(cfg['project']['retrieval_root']) / cfg['experiment']['dataset']
            / ('v5' if str(cfg.get('version','')).startswith('5.') else ('v41' if str(cfg.get('version','')).startswith('4.1.') else ('v4' if str(cfg.get('version','')).startswith(('4.','5.')) else 'v3'))) / ('formal_smoke' if smoke else 'formal') / ('teacher_replay' if str(cfg.get('version','')).startswith('5.') else 'teacher_replay_v2'))


def prepare_bank(cfg,split,dataset,cache,memory,teacher,checkpoint_path,retrieval_path,device,
                 num_queries=None, smoke=False,cache_tag=None):
    """Build/use a bank of shape [num_queries, candidates, horizon, nodes]."""
    from .data import RetrievalCache
    dtype=str(cfg['runtime'].get('replay_bank_dtype','float16'))
    if dtype not in ('float16','float32'):raise ValueError('replay_bank_dtype must be float16 or float32')
    shortlist_per=int(cfg['retrieval']['shortlist_per_period'])
    k=int(cache.candidate_count)
    n=len(dataset) if num_queries is None else min(int(num_queries),len(dataset))
    h=int(cfg['experiment']['horizon']);nodes=int(memory.arrays['x'].shape[-1])
    if cache_tag is not None:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_]+',cache_tag):
            raise ValueError('Unsafe replay cache_tag')
    name=f'{cache_tag or split}.teacher_replay_k{k}_{dtype}.npy'
    folder=replay_bank_dir(cfg, smoke=smoke)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/name;meta_file=folder/(name+'.json')
    source={
        'schema':27,'cache_tag':cache_tag,'replay_protocol':'source_mask_intersection_v2','split':split,'target':cfg['experiment']['target'],
        'dataset':cfg['experiment']['dataset'],'num_queries':n,'num_candidates':k,
        'horizon':h,'nodes':nodes,'dtype':dtype,
        'pretrained_backbone_sha256':file_sha256(checkpoint_path),
        'retrieval_index_sha256':file_sha256(retrieval_path),
        'query_masks':_file_key(Path(dataset.root)/f'{split}_history_observed.npy'),
        'history_memory':_file_key(Path(memory.arrays['x'].filename)),
        'future_memory':_file_key(Path(memory.arrays['y'].filename)),
        'replay_type':'matched','pretrained_mask_projection':cfg.get('runtime',{}).get('teacher_fingerprint','zero_initialization'),
        'strict_source_mask':str(cfg.get('version','')).startswith(('4.','5.')),
        'source_observed_sha256':file_sha256(Path(memory.arrays['x'].filename).parent/'train_history_observed.npy')
            if str(cfg.get('version','')).startswith(('4.','5.')) else None,
        'source_valid_sha256':file_sha256(Path(memory.arrays['x'].filename).parent/'train_artificial_missing.npy')
            if str(cfg.get('version','')).startswith(('4.','5.')) else None}
    if path.exists() and meta_file.exists():
        try:
            existing=json.loads(meta_file.read_text())
            arr=np.load(path,mmap_mode='r',allow_pickle=False)
            if existing==source and arr.shape==(n,k,h,nodes):
                print(f'REPLAY_BANK_HIT split={split} path={path}',flush=True)
                return ReplayBank(path,source,cfg['runtime'].get('replay_bank_on_gpu',True) and not smoke,device)
        except (OSError,ValueError,KeyError):pass
    tmp=path.with_name(path.name+'.partial.npy')
    print(f'REPLAY_BANK_BUILD split={split} queries={n} candidates={k} dtype={dtype} path={path}',flush=True)
    start=time.time()
    writer=np.lib.format.open_memmap(tmp,'w+',dtype=dtype,shape=(n,k,h,nodes))
    qb=int(cfg['runtime'].get('replay_query_batch_size',12))
    chunk=int(cfg['retrieval']['replay_batch_size'])
    try:
        teacher.eval()
        with torch.inference_mode():
            for s in range(0,n,qb):
                e=min(n,s+qb)
                ids,_,_=shortlist(cache,np.arange(s,e),shortlist_per,device)
                candidate=memory.fetch(ids,device)
                qmask=torch.from_numpy(np.array(dataset.observed[s:e],np.float32,copy=True)).to(device)
                res=counterfactual_residuals(teacher,candidate,qmask,chunk_size=chunk,replay='matched',
                    strict_source_mask=str(cfg.get('version','')).startswith(('4.','5.')))
                writer[s:e]=res.detach().cpu().numpy().astype(dtype,copy=False)
                if smoke or e==n or ((s//qb)%25==0):
                    print(f'REPLAY_BANK_PROGRESS split={split} queries_done={e}/{n} elapsed={time.time()-start:.1f}s',flush=True)
        writer.flush();del writer
        os.replace(tmp,path)
        mtmp=meta_file.with_name(meta_file.name+'.partial')
        mtmp.write_text(json.dumps(source,indent=2),encoding='utf-8');os.replace(mtmp,meta_file)
    finally:
        if tmp.exists():tmp.unlink()
    print(f'REPLAY_BANK_PASS split={split} seconds={time.time()-start:.1f}',flush=True)
    return ReplayBank(path,source,cfg['runtime'].get('replay_bank_on_gpu',True) and not smoke,device)
