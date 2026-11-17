"""Per-stage identities and the shared/own stage directory layout.

A stage key hashes ONLY what that stage (and its predecessors) consume.  The
retrieval configuration never enters the keys of s0/normal/anchor_cf, so every
retrieval ablation of the same seed reuses ReShift's own s0/normal/anchor_cf
checkpoints from the shared root; anchor_ret/anchor_control live in the
ablation's run directory.  Nothing outside ReShift is ever read.
"""
from __future__ import annotations
import copy,hashlib,json,os
from pathlib import Path

SHARED_STAGES=('s0','normal','anchor_cf')
RETRIEVAL_STAGES=('anchor_ret','anchor_control')
RETRIEVAL_TRAINING_KEYS=('lr_retrieval','arm_gain_weight','arm_gain_scale')


def _digest(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,allow_nan=False,default=str).encode()).hexdigest()


def retrieval_cfg(cfg):
    r=(cfg.get('model') or {}).get('retrieval') or {}
    return r if r.get('enabled',False) else None


def stage_keys(cfg,upstream,stages):
    exp=cfg['experiment']
    model=copy.deepcopy(cfg['model']);model.pop('retrieval',None)
    training=copy.deepcopy(cfg['training'])
    epochs=dict(training.pop('stage_epochs'))
    for k in RETRIEVAL_TRAINING_KEYS:training.pop(k,None)
    keys={}
    up_s0={k:upstream[k] for k in ('code_sha256','backbone_sha256','cache_path','split_manifest_sha256',
                                   'smoke','limits')}
    keys['s0']=_digest({'stage':'s0','up':up_s0,'experiment':exp,'s0':cfg.get('s0',{}),
                        'tail':cfg['model'].get('stla_tail_layers')})
    keys['bank']=_digest({'stage':'bank','s0':keys['s0']})
    keys['normal']=_digest({'stage':'normal','s0':keys['s0'],'up':upstream,'experiment':exp,
                            'model':model,'training':training,'epochs':epochs.get('normal')})
    keys['anchor_cf']=_digest({'stage':'anchor_cf','prev':keys['normal'],'epochs':epochs.get('anchor_cf')})
    ret={'retrieval':(cfg['model'].get('retrieval') or {}),'bank':cfg.get('retrieval_bank',{}),
         'training':{k:cfg['training'].get(k) for k in RETRIEVAL_TRAINING_KEYS}}
    prev=keys['anchor_cf']
    for st in stages:
        if st in RETRIEVAL_STAGES:
            keys[st]=_digest({'stage':st,'prev':prev,'ret':ret,'epochs':epochs.get(st)})
            prev=keys[st]
    return keys


class Layout:
    def __init__(self,run_dir,shared_root,keys):
        self.run=Path(run_dir);self.shared=Path(shared_root);self.keys=dict(keys)

    def is_shared(self,stage):return stage in SHARED_STAGES

    def key(self,stage):return self.keys[stage]

    def dir(self,stage):
        if self.is_shared(stage):return self.shared/f'{stage}-{self.keys[stage][:16]}'
        return self.run/stage

    def bank_root(self):return self.shared/f'banks-{self.keys["bank"][:16]}'

    def acquire(self,stage):
        """Exclusive lock while a SHARED stage is being trained."""
        if not self.is_shared(stage):return None
        d=self.dir(stage);d.mkdir(parents=True,exist_ok=True)
        lock=d/'.training.lock'
        try:
            fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
        except FileExistsError:
            raise RuntimeError(f'Shared stage {stage} is being trained by another job ({lock}). '
                               'Wait for it to finish (or delete a stale lock), then resubmit.')
        return lock
