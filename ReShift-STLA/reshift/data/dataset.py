"""Train-only privileged history and controlled mask interventions.
Val raw history may be used ONLY to build fixed masked validation scenarios;
it is never returned to the forecast model or paired teacher.
"""
from __future__ import annotations
import hashlib, re, os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from corsta.data import SupportSplit

SEVERITIES=(20,40,60,90)
KINDS=('entry_random','sensor','temporal','regional','directional')
# Keep the published v0.2.1 VAL scenario identities unchanged for fair ablations.
VALIDATION_KINDS=('sensor','regional','directional','temporal')
# Failure-family vocabulary of Base_SupportBench manifests (column `family`).
FAMILIES=('entry_random','sensor_outage','temporal_outage','regional_outage','directional_outage')
KIND_TO_FAMILY={'entry_random':'entry_random','sensor':'sensor_outage','temporal':'temporal_outage',
                'regional':'regional_outage','directional':'directional_outage'}


def family_id(name):
    name=str(name)
    name=KIND_TO_FAMILY.get(name,name)
    return FAMILIES.index(name) if name in FAMILIES else -1


def _rng(seed,epoch,index,severity,kind):
    key=f'{seed}/{epoch}/{index}/{severity}/{kind}'.encode()
    n=int.from_bytes(hashlib.sha256(key).digest()[:8],'little')
    return np.random.default_rng(n)


def _node_order(rng,n,kind,neighbors=None,directions=None):
    if kind=='sensor':return rng.permutation(n).tolist()
    if kind=='directional' and directions is not None:
        keys=np.asarray(directions)
        choices=np.unique(keys)
        if len(choices):
            selected=rng.choice(choices)
            group=rng.permutation(np.flatnonzero(keys==selected)).tolist()
            rest=rng.permutation(np.flatnonzero(keys!=selected)).tolist()
            return group+rest
    if kind=='regional' and neighbors is not None:
        start=int(rng.integers(n));order=[];seen={start};frontier=[start]
        while frontier:
            node=frontier.pop(0);order.append(node)
            adjacent=list(neighbors[node]);rng.shuffle(adjacent)
            for j in adjacent:
                j=int(j)
                if 0<=j<n and j not in seen:
                    seen.add(j);frontier.append(j)
        return order+[int(j) for j in rng.permutation(n) if int(j) not in seen]
    return rng.permutation(n).tolist()


def structured_mask(source_valid, severity,kind,rng,neighbors=None,directions=None):
    """Return mask subset of source_valid; deletion fraction is relative to valid elements.

    Pattern clusters dictate removal *order*; exact quotas preserve severity budget.
    """
    src=np.asarray(source_valid,dtype=bool);L,N=src.shape
    if severity<0 or severity>100:raise ValueError('severity must be percentage')
    n_drop=min(int(round(src.sum()*severity/100)),int(src.sum()))
    out=src.copy()
    if n_drop==0:return out
    if kind=='entry_random':
        valid_positions=np.flatnonzero(src.reshape(-1))
        chosen=rng.choice(valid_positions,size=n_drop,replace=False)
        out.reshape(-1)[chosen]=False
        return out
    order=_node_order(rng,N,kind,neighbors,directions)
    if kind=='temporal':
        start=int(rng.integers(L));times=list(range(start,L))+list(range(0,start))
        # Time-major contiguous slices; not node-major deletion.
        locations=[(t,int(i)) for t in times for i in order] 
    else:
        # sensor/regional/directional node-first structures; random time offset
        time_order=np.roll(np.arange(L),int(rng.integers(L)))
        locations=[(int(t),int(i)) for i in order for t in time_order]
    count=0
    for t,i in locations:
        if out[t,i]:
            out[t,i]=False;count+=1
            if count>=n_drop:break
    assert count==n_drop
    return out


class CoRSTASplit(Dataset):
    def __init__(self,cache,split,seed=10,severity=None,epoch=0,
                 augment=True,val_raw_scenarios=True,limit=None,neighbors=None,directions=None,contrast_views=False,num_views=2,
                 failure_kinds=None,legacy_multimask=False):
        self.raw=SupportSplit(cache,split,limit=limit,cf_training=(split=='train'))
        self.split=split;self.seed=int(seed);self.severity=severity;self.epoch=epoch
        self.augment=augment;self.neighbors=neighbors;self.directions=directions
        self.contrast_views=bool(contrast_views) and split=="train"
        self.kinds=tuple(failure_kinds) if failure_kinds is not None else KINDS
        if not self.kinds or any(k not in KINDS for k in self.kinds):
            raise ValueError('Unknown failure kind in requested mask curriculum')
        self.legacy_multimask=bool(legacy_multimask)
        self.num_views=int(num_views)
        if not 1<=self.num_views<=20:raise ValueError('num_views must be 1..20')
        self.full=None
        if split=='train':self.full=self.raw.x_full
        elif split=='val' and severity is not None and val_raw_scenarios:
            raw_file=Path(cache)/'val_x_full_norm.npy'
            if not raw_file.exists():
                raise FileNotFoundError('4-severity Validation needs val_x_full_norm.npy solely for constructing masked inputs. '
                                        'Do not use TEST data or fake s20 from an s40 mask.')
            self.full=np.load(raw_file,mmap_mode='r')
        if self.full is not None and len(self.full)<len(self.raw):raise ValueError('full history shorter than split')
        meta=self.raw.meta
        self.manifest_family=(np.asarray([family_id(v) for v in meta['family'].astype(str).to_numpy()],dtype=np.int64)
                              if 'family' in meta.columns else None)

    def set_epoch(self,epoch):self.epoch=int(epoch)
    def __len__(self):return len(self.raw)
    @property
    def window_starts(self):return self.raw.window_starts[:len(self)]

    def __getitem__(self,index):
        source=self.raw[index]
        source_valid=np.asarray(source['original_valid'].numpy(),dtype=bool)
        if self.full is None or (self.split=='val' and self.severity is None):
            x=source['x'].clone();mask=source['mask'].clone();sev=int(round(100*(1-mask.float().mean().item())))
            kind=-1;fam=(int(self.manifest_family[index]) if self.manifest_family is not None else -1)
        else:
            if self.split=='train':
                severity=int(self.severity if self.severity is not None else SEVERITIES[(index+self.epoch)%len(SEVERITIES)])
                kind_id=(index//len(SEVERITIES)+self.epoch)%len(self.kinds)
            else:
                severity=int(self.severity);kind_id=index%len(VALIDATION_KINDS)
            kind=(self.kinds[kind_id] if self.split=='train' else VALIDATION_KINDS[kind_id])
            if self.split=='train' and not self.augment:
                observed=source['mask'].numpy().astype(bool);kind_id=-1
            else:
                rng=_rng(self.seed,self.epoch if self.split=='train' else 0,
                         index,severity,kind)
                observed=structured_mask(source_valid,severity,kind,rng,
                                         neighbors=self.neighbors,directions=self.directions)
            full=np.asarray(self.full[index],dtype=np.float32)
            x=torch.from_numpy(np.where(observed,np.nan_to_num(full,nan=0.0,posinf=0.0,neginf=0.0),0).astype(np.float32))
            mask=torch.from_numpy(observed.copy()).float();sev=severity
            fam=family_id(kind) if kind_id>=0 else (int(self.manifest_family[index]) if self.manifest_family is not None else -1)
            kind=kind_id
        result={
            'x':x,'mask':mask,'tod':source['tod'],'dow':source['dow'],
            'y_norm':torch.where(source['future_valid'],torch.nan_to_num(source['y_norm']),torch.zeros_like(source['y_norm'])),
            'y_raw':torch.where(source['future_valid'],torch.nan_to_num(source['y_raw']),torch.zeros_like(source['y_raw'])),
            'future_valid':source['future_valid'],'direct_mask':source['direct_mask'],
            'index':torch.tensor(index,dtype=torch.int64),
            'severity':torch.tensor(sev,dtype=torch.int64),
            'failure_kind':torch.tensor(kind,dtype=torch.int64),
            'family_id':torch.tensor(fam,dtype=torch.int64),
            'window_start':torch.tensor(int(source['window_start']),dtype=torch.int64),
        }
        if self.contrast_views and self.num_views>1:
            if self.full is None:raise RuntimeError('Contrast views require TRAIN complete history')
            # Rotate over 4 severity x 5 failure kinds, without creating 20
            # teacher/student forwards per query. Each view shares X_full and Y.
            # The primary view is separate from these K-1 extra views.
            extra_x=[];extra_mask=[];extra_sev=[];extra_kind=[]
            primary_mask=mask.numpy().astype(bool)
            full_np=np.asarray(self.full[index],dtype=np.float32)
            severity_a=int(sev) if int(sev) in SEVERITIES else SEVERITIES[index%len(SEVERITIES)]
            primary_kind=int(kind) if int(kind)>=0 else index%len(self.kinds)
            for view_id in range(1,self.num_views):
                # Include same-severity different-kind AND different-severity views.
                if self.legacy_multimask and self.num_views==2:
                    # Preserve v0.2.1's exact two-view TRAIN contrast curriculum.
                    sev_b=(severity_a if index%2==0 else
                           SEVERITIES[(SEVERITIES.index(severity_a)+1)%len(SEVERITIES)])
                    kind_b=(primary_kind+1+(index%3))%len(self.kinds)
                    kind_name=self.kinds[kind_b]
                    rng_b=_rng(self.seed,self.epoch,index,sev_b,kind_name+'-contrast')
                else:
                    step=view_id+((self.epoch+index)//len(self.kinds))
                    sev_b=(severity_a if view_id%2==1 else
                           SEVERITIES[(SEVERITIES.index(severity_a)+view_id)%len(SEVERITIES)])
                    kind_b=(primary_kind+step)%len(self.kinds)
                    if kind_b==primary_kind:kind_b=(kind_b+1)%len(self.kinds)
                    kind_name=self.kinds[kind_b]
                    rng_b=_rng(self.seed,self.epoch,index,sev_b,kind_name+f'/view{view_id}')
                mb=structured_mask(source_valid,sev_b,kind_name,rng_b,
                                   neighbors=self.neighbors,directions=self.directions)
                if np.array_equal(mb,primary_mask):
                    # Degenerate small graphs still get a distinct view if possible.
                    for attempt in range(1,5):
                        alt_sev=SEVERITIES[(SEVERITIES.index(sev_b)+attempt)%len(SEVERITIES)]
                        mb=structured_mask(source_valid,alt_sev,kind_name,rng_b,
                                           neighbors=self.neighbors,directions=self.directions)
                        if not np.array_equal(mb,primary_mask):
                            sev_b=alt_sev
                            break
                xb=np.where(mb,np.nan_to_num(full_np,nan=0.,posinf=0.,neginf=0.),0).astype(np.float32)
                extra_x.append(torch.from_numpy(xb))
                extra_mask.append(torch.from_numpy(mb.copy()).float())
                extra_sev.append(sev_b)
                extra_kind.append(kind_b)
            result['view_x']=torch.stack(extra_x)             # [K-1,L,N]
            result['view_mask']=torch.stack(extra_mask)       # [K-1,L,N]
            result['view_severity']=torch.tensor(extra_sev,dtype=torch.int64)
            result['view_kind']=torch.tensor(extra_kind,dtype=torch.int64)
            # Backwards compatibility with v0.2.1 two-view CF-first trainer.
            result['contrast_x']=result['view_x'][0]
            result['contrast_mask']=result['view_mask'][0]
            result['contrast_severity']=result['view_severity'][0]
            result['contrast_kind']=result['view_kind'][0]
        if self.split=='train':
            result['x_full']=source['x_full_training_only']
            result['original_valid']=source['original_valid'].float()
            if not torch.all(mask<=result['original_valid']):raise RuntimeError('Mask increased original observation availability')
            if not torch.allclose(x[mask.bool()],result['x_full'][mask.bool()],atol=1e-6,equal_nan=False):
                raise RuntimeError('Visible TRAIN values do not match original measurements')
        return result


def resolve_shared_cache(root,target,dataset,stage='formal',severity=None):
    """Official Backbone shared cache API, with no retrieval/config-v7 dependency."""
    import sys
    root=Path(root)
    b_root=root/'Backbone'
    if str(b_root) not in sys.path:sys.path.insert(0,str(b_root))
    from backbone.utils import load_yaml
    from backbone.shared_cache import cache_path,validate_shared_cache
    os.environ['TRAFFIDENT_TARGET']=str(target)
    cfg=load_yaml(b_root/'configs'/'backbone.yaml')
    def replace(x):
        if isinstance(x,str):return re.sub(r'TraffiDent_Net_\d{2}',dataset,x)
        if isinstance(x,list):return [replace(v) for v in x]
        if isinstance(x,dict):return {k:replace(v) for k,v in x.items()}
        return x
    cfg=replace(cfg)
    # Upstream Backbone uses its own environment/config target selection.
    validate_shared_cache(cfg,stage,severity)
    return Path(cache_path(cfg,stage,severity))
