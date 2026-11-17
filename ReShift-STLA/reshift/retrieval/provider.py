"""Evidence provider: builds/loads banks and serves per-batch neighbour evidence.

Bank keys: 'train_slot{s}' (Train mask slot s), 'val_s{sev}', 'test_s{sev}';
'<key>_wrong' is the legal wrong-case bank used only by the cross_history audit.
Audit modes (in-checkpoint interventions, NOT independently trained controls):
  full          true neighbours
  shuffled      neighbour contents rolled against their similarities (rank mismatch)
  cross_history legal non-overlapping wrong Train cases with the true similarities
  no_replay     neighbour P0 replaced by its full-evidence forecast (no failure replay)
"""
from __future__ import annotations
from pathlib import Path
import torch
from reshift.retrieval.bank import (BANK_SCHEMA,ReplayBank,build_bank,build_p_full,canonical,
                                   code_digest,digest)
from reshift.retrieval.index import MemoryIndex

AUDIT_MODES=('full','shuffled','cross_history','no_replay')


class EvidenceProvider:
    def __init__(self,memory,bank_root,stla,bank_cfg,device,identity,log=print):
        self.memory=memory;self.root=Path(bank_root);self.stla=stla;self.cfg=dict(bank_cfg)
        self.device=device;self.identity=dict(identity);self.log=log
        self.banks={};self.index=None
        self.top_k=int(self.cfg['top_k']);self.span=int(self.cfg.get('span',memory.history+memory.horizon))
        self.slots=int(self.cfg.get('mask_slots',2))
        if self.slots<1:raise ValueError('mask_slots must be >=1')
        self.code=code_digest()
        pf={'schema':BANK_SCHEMA,'kind':'p_full','memory_rows':len(memory),'code':self.code,**self.identity}
        folder=self.root/f'p_full-{digest(pf)}'
        memory.attach_p_full(build_p_full(folder,pf,memory,stla,device,int(self.cfg.get('pfull_batch',64)),log))
        if bool(self.cfg.get('memory_on_gpu',False)):
            ok=memory.preload(device);log(f'RESHIFT_MEMORY_GPU={ok}')

    # ------------------------------------------------------------------ keys
    def slot_for_epoch(self,epoch):
        return 1+(int(epoch)-1)%self.slots

    @staticmethod
    def train_key(slot):return f'train_slot{int(slot)}'

    def _index(self):
        if self.index is None:
            self.index=MemoryIndex(self.memory,self.cfg.get('periods',[3]),self.device,
                                   int(self.cfg.get('memory_chunk',4096)))
        return self.index

    def ensure(self,key,dataset,query_desc,train_query=False,wrong=False):
        """Build (or verify and load) the bank for `key` over `dataset`."""
        full_key=key+('_wrong' if wrong else '')
        if full_key in self.banks:return self.banks[full_key]
        embargo=int(self.cfg.get('train_embargo_steps',288)) if train_query else 0
        fp={'schema':BANK_SCHEMA,'kind':'replay','key':full_key,'queries':len(dataset),
            'top_k':self.top_k,'periods':list(self.cfg.get('periods',[3])),'span':self.span,
            'embargo':embargo,'min_valid':int(self.cfg.get('min_valid_features',8)),
            'wrong':bool(wrong),'memory_rows':len(self.memory),'code':self.code,
            'query':query_desc,**self.identity}
        folder=self.root/f'{full_key}-{digest(fp)}'
        if (folder/'COMPLETE').exists():
            bank=ReplayBank(folder,bool(self.cfg.get('bank_on_gpu',False)),self.device)
            if bank.fingerprint!=canonical(fp):
                raise RuntimeError(f'Bank fingerprint mismatch {folder}')
        else:
            self.log(f'RESHIFT_BANK_BUILD key={full_key} queries={len(dataset)} k={self.top_k} embargo={embargo}')
            build_bank(folder,fp,dataset,self.memory,self._index(),self.stla,self.device,self.top_k,
                       self.span,embargo,int(self.cfg.get('min_valid_features',8)),
                       int(self.cfg.get('query_batch',32)),int(self.cfg.get('replay_chunk',64)),
                       int(self.cfg.get('workers',0)),wrong=wrong,log=self.log)
            bank=ReplayBank(folder,bool(self.cfg.get('bank_on_gpu',False)),self.device)
        if len(bank)!=len(dataset):raise RuntimeError(f'bank {full_key} rows {len(bank)} != dataset {len(dataset)}')
        self.banks[full_key]=bank
        return bank

    # ----------------------------------------------------------------- batch
    def batch(self,key,rows,mode='full'):
        """Neighbour evidence for dataset rows [B]; tensors on self.device."""
        if mode not in AUDIT_MODES:raise ValueError(f'unknown evidence mode {mode}')
        name=key+('_wrong' if mode=='cross_history' else '')
        if name not in self.banks:raise RuntimeError(f'Bank {name} not prepared (call ensure first)')
        b=self.banks[name].fetch(rows,self.device)
        idx=b['idx'];sim=b['sim'].float();p0=b['p0'].float()
        if mode=='shuffled':
            # Keep similarities in rank order but roll the neighbour contents.
            idx=torch.roll(idx,1,dims=1);p0=torch.roll(p0,1,dims=1)
        mem=self.memory.fetch(idx,self.device,keys=('y','fv','p_full'))
        p_full=mem['p_full'].float()
        if mode=='no_replay':p0=p_full.clone()
        valid_k=idx>=0
        return {'sim':sim,'valid_k':valid_k,'p0':p0,'p_full':p_full,
                'y':mem['y'].float(),'fv':mem['fv'].bool()}
