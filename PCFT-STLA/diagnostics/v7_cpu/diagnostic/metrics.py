"""Online exact valid-position metrics and stratification; raw units throughout."""
from __future__ import annotations
import math
from collections import defaultdict
import numpy as np

PRED_KEYS=('base','local','propagation','controlled','uniform','fixed_shrink','hard_action','oracle_discrete')

class Accumulator:
    def __init__(self):
        self.d=defaultdict(float)
    def add(self,y,predictions,valid,**extras):
        valid=np.asarray(valid,bool) & np.isfinite(y)
        n=int(valid.sum())
        if not n:return
        self.d['count']+=n
        err={k:np.abs(np.asarray(v)-y) for k,v in predictions.items()}
        for key,error in err.items():
            vals=error[valid]
            self.d[f'{key}_abs']+=float(vals.sum(dtype=np.float64))
            self.d[f'{key}_sq']+=float((vals.astype('float64')**2).sum())
        if 'base' in err and 'local' in err and 'propagation' in err:
            base,local,prop=err['base'],err['local'],err['propagation']
            self.d['local_harm_n']+=int(((local>base+1e-6)&valid).sum())
            self.d['prop_harm_n']+=int(((prop>local+1e-6)&valid).sum())
            self.d['local_harm_abs']+=float(np.maximum(local-base,0)[valid].sum())
            self.d['prop_harm_abs']+=float(np.maximum(prop-local,0)[valid].sum())
            self.d['local_gain_abs']+=float((base-local)[valid].sum())
            self.d['prop_gain_abs']+=float((local-prop)[valid].sum())
        for k,v in extras.items():
            if v is None:continue
            array=np.broadcast_to(v,np.shape(valid))
            self.d[k+'_sum']+=float(array[valid].sum(dtype=np.float64))
        
    def finish(self):
        n=self.d['count']
        out={'valid_positions':int(n)}
        if not n:return out
        for key in PRED_KEYS:
            if f'{key}_abs' in self.d:
                out[f'{key}_mae']=self.d[f'{key}_abs']/n
                out[f'{key}_rmse']=math.sqrt(self.d[f'{key}_sq']/n)
        for k in ('local','prop'):
            out[f'{k}_harm_frequency']=self.d[f'{k}_harm_n']/n
            out[f'{k}_harm_magnitude_mean_all']=self.d[f'{k}_harm_abs']/n
            out[f'{k}_gain_abs_mean']=self.d[f'{k}_gain_abs']/n
        for k,value in list(self.d.items()):
            if k.endswith('_sum') and k not in ('count',):out[k[:-4]+'_mean']=value/n
        if 'base_mae' in out and 'controlled_mae' in out and out['base_mae']:
            out['overall_improvement_pct']=100*(out['base_mae']-out['controlled_mae'])/out['base_mae']
        if 'local_mae' in out and 'propagation_mae' in out and out['local_mae']:
            out['propagation_vs_local_pct']=100*(out['local_mae']-out['propagation_mae'])/out['local_mae']
        return out

class GroupedMetrics:
    def __init__(self):self.groups=defaultdict(Accumulator)
    def add(self,name,label,y,preds,valid,**extra):self.groups[(name,label)].add(y,preds,valid,**extra)
    def rows(self):
        return [{'dimension':k[0],'group':k[1],**a.finish()}
                for k,a in sorted(self.groups.items(),key=lambda it:it[0])]

def bucketize_int(values,edges,names):
    """Boundaries [a,b) except final; labels correspond to digitize positions."""
    return np.asarray(names,dtype=object)[np.digitize(values,edges,right=False)]

def group_masks(batch,extra=None):
    mask=batch['observed_mask'].detach().numpy()[...,extra.get('target_idx',0)]>0.5
    B,H,N=mask.shape
    cov=mask.mean(1)
    valid_pair=(mask[:,1:]&mask[:,:-1])
    pair_count=valid_pair.sum(1)
    last=np.where(mask,np.arange(H)[None,:,None],-1).max(axis=1)
    last_age=np.where(last<0,H,H-1-last)
    longest=np.zeros((B,N),dtype=np.int64)
    streak=np.zeros_like(longest)
    for t in range(H):
        streak=np.where(mask[:,t],0,streak+1)
        longest=np.maximum(longest,streak)
    categories={
        'target_coverage':bucketize_int((cov*100).round(),[.1,25.1,50.1,75.1],['0','1-25','26-50','51-75','76-100']),
        'last_observation_age':bucketize_int(last_age,[1,4,7,12],['0','1-3','4-6','7-11','no_observation']),
        'longest_missing_streak':bucketize_int(longest,[1,4,7,10],['0','1-3','4-6','7-9','10-12']),
        'valid_difference_pairs':bucketize_int(pair_count,[1,3,6],['0','1-2','3-5','6-11'])}
    if extra is not None and extra.get('incoming_available_count') is not None:
        k=np.asarray(extra['incoming_available_count'])
        # For graph interpretation, incoming counts may exceed unique nodes if functional & physical pairs overlap.
        categories['available_incoming_edges']=bucketize_int(k,[1,2,4,8],['0','1','2-3','4-7','8+'])
    if extra is not None and extra.get('weighted_source_age') is not None:
        v=np.asarray(extra['weighted_source_age'])
        categories['weighted_source_age']=bucketize_int(v,[.5,3.5,6.5,11.5],['fresh_0','age_1-3','age_4-6','age_7-11','no_weighted_source'])
    return categories
