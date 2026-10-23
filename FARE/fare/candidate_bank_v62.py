"""One immutable, validation-label-free candidate set for all v6.2 Transfer variants.

A shared *frozen* support-model retrieval snapshot generates these candidate
identities. No target/future information from the evaluated query is accessed.
Use .npy memmaps so even Net-06 full-train banks remain RAM bounded.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from .protocol import make_loader,device_batch
from .config import hash_file,dump_json


def _ix_hash(ix):
    return hashlib.sha256(np.asarray(ix,dtype=np.int64).tobytes()).hexdigest()


class FrozenCandidateBank:
    def __init__(self,path,split,ds,manifest_expected=None):
        self.root=Path(path)
        meta=json.loads((self.root/'manifest.json').read_text())
        for name,val in (manifest_expected or {}).items():
            if meta.get(name)!=val:raise ValueError(f'CandidateBank {name} provenance mismatch')
        info=meta['splits'].get(split)
        if info is None:raise KeyError(f'CandidateBank missing split {split}')
        if info['index_sha256']!=_ix_hash(ds.ix) or info['rows']!=len(ds) or meta['nodes']!=ds.n:
            raise ValueError('CandidateBank dataset subset/order/nodes disagree')
        self.ids=np.load(self.root/f'{split}_ids.npy',mmap_mode='r',allow_pickle=False)
        self.valid=np.load(self.root/f'{split}_valid.npy',mmap_mode='r',allow_pickle=False)
        self.scores=np.load(self.root/f'{split}_scores.npy',mmap_mode='r',allow_pickle=False)
        for field in ('ids','valid','scores'):
            entry=info.get('files',{}).get(field)
            if entry and hash_file(self.root/f'{split}_{field}.npy')!=entry:
                raise RuntimeError(f'CandidateBank {split}_{field} hash mismatch')
        self.mapping={int(j):i for i,j in enumerate(np.asarray(ds.ix,dtype=np.int64))}
        if self.ids.shape!=(len(ds),ds.n,meta['slots']):raise RuntimeError('CandidateBank IDs shape mismatch')
        self.metadata=meta
    def fetch(self,row_ids,device):
        rows=row_ids.detach().cpu().numpy().reshape(-1)
        try:pos=np.array([self.mapping[int(j)] for j in rows],dtype=np.int64)
        except KeyError as e:raise ValueError(f'CandidateBank unknown row ID: {e}') from e
        ids=torch.as_tensor(np.array(self.ids[pos],copy=True),device=device,dtype=torch.long)
        valid=torch.as_tensor(np.array(self.valid[pos],copy=True),device=device,dtype=torch.bool)
        scores=torch.as_tensor(np.array(self.scores[pos],copy=True),device=device,dtype=torch.float32)
        return {'ids':ids,'valid':valid,'scores':scores}


@torch.no_grad()
def build_candidate_bank(model,memory,graph,train_ds,val_ds,device,output,config,support_sha,meta,batch_size=8,workers=0):
    """Freeze support/retriever; write selected candidates for train and val.

    The retrieval model is the randomly initialised retriever from the same
    fresh Support-only checkpoint. Its state *is not trained on future labels*.
    """
    output=Path(output)
    if output.exists() and (output/'manifest.json').exists():
        raise FileExistsError('CandidateBank is immutable; choose a new experiment directory')
    output.mkdir(parents=True,exist_ok=True)
    model.eval()
    slots=min(int(config['retrieval']['shortlist_k']),memory.m)
    manifest={'schema':1,'target':meta['target'],'dataset':meta['dataset'],
              'seed':meta['seed'],'backbone_sha256':meta['backbone_sha256'],
              'memory_sha256':meta['memory_sha256'],'support_checkpoint_sha256':support_sha,
              'selection':'frozen_support_retriever_no_query_targets',
              'purge_steps':int(config['memory']['purge_steps']),
              'nodes':train_ds.n,'slots':slots,'splits':{}}
    for split,ds in (('train',train_ds),('val',val_ds)):
        ids_mem=np.lib.format.open_memmap(output/f'{split}_ids.npy',mode='w+',dtype='int32',shape=(len(ds),ds.n,slots))
        val_mem=np.lib.format.open_memmap(output/f'{split}_valid.npy',mode='w+',dtype='bool',shape=(len(ds),ds.n,slots))
        scores_mem=np.lib.format.open_memmap(output/f'{split}_scores.npy',mode='w+',dtype='float32',shape=(len(ds),ds.n,slots))
        offset=0;valid_count=0
        for raw in make_loader(ds,batch_size,workers=workers):
            batch=device_batch(raw,device)
            base=(model.backbone.encode_base(batch['x'],batch['mask'],batch['tod'],batch['dow'])
                  if hasattr(model.backbone,'encode_base') else
                  model.backbone(batch['x'],batch['mask'],batch['tod'],batch['dow']))
            hidden=model.base_proj(base['hidden'])
            sup=model.support(batch['x'],batch['mask'],hidden,base['prediction'],model.graph)
            short=model.retriever(batch['x'],batch['mask'],sup,batch['origin'],memory,
                                  model.graph,purge=manifest['purge_steps'])
            v=short['valid']
            lawful=memory.valid_for(batch['origin'],device,manifest['purge_steps'])
            chosen_lawful=lawful.gather(1,short['ids'].reshape(len(v),-1)).reshape_as(v)
            if (v & ~chosen_lawful).any():raise RuntimeError('Causal retrieval check failed')
            count=batch['x'].shape[0];stop=offset+count
            ids_mem[offset:stop]=short['ids'].int().cpu().numpy()
            val_mem[offset:stop]=v.cpu().numpy()
            scores_mem[offset:stop]=short['scores'].float().cpu().numpy()
            valid_count+=int(v.sum().item());offset=stop
            if offset%max(batch_size,32*batch_size)==0 or offset==len(ds):
                print(f'FARE62_CANDIDATE_BANK_PROGRESS split={split} rows={offset}/{len(ds)}',flush=True)
        if offset!=len(ds):raise RuntimeError('CandidateBank row count mismatch')
        for arr in (ids_mem,val_mem,scores_mem):arr.flush()
        manifest['splits'][split]={'rows':len(ds),'index_sha256':_ix_hash(ds.ix),
                                    'valid_slots':valid_count,
                                    'files':{field:hash_file(output/f'{split}_{field}.npy')
                                             for field in ('ids','valid','scores')}}
    dump_json(output/'manifest.json',manifest)
    print('FARE62_CANDIDATE_BANK_COMPLETE '+json.dumps({'directory':str(output),
          'splits':manifest['splits']}),flush=True)
    return manifest
