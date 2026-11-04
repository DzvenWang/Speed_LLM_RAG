"""Streaming, valid-element-weighted forecasting and oracle metrics."""
from __future__ import annotations
import math
import numpy as np
import torch

class MaskedMetric:
    def __init__(self): self.abs_sum=0.;self.sq_sum=0.;self.count=0
    def add(self,pred,target,valid):
        v=valid.bool();err=(pred.float()-target.float())[v]
        self.abs_sum+=float(err.abs().double().sum());self.sq_sum+=float(err.double().square().sum());self.count+=int(err.numel())
    def to_dict(self):
        return {'mae':self.abs_sum/self.count if self.count else None,
                'rmse':math.sqrt(self.sq_sum/self.count) if self.count else None,
                'valid_elements':self.count}

class ValidAverage:
    def __init__(self):self.total=0.;self.n=0
    def add(self,values,valid):
        values=values.float();valid=valid.bool()
        self.total+=float(values[valid].double().sum());self.n+=int(valid.sum())
    def value(self):return self.total/self.n if self.n else None

class SampleDistribution:
    """Constant-memory, approximately uniform sub-samples for display quantiles."""
    def __init__(self,seed=10,capacity=4096):
        self.rng=np.random.default_rng(seed);self.capacity=capacity;self.sample=[];self.n=0;self.sum=0.;self.sum_sq=0.
    def add(self,data):
        x=torch.as_tensor(data).detach().float().reshape(-1).cpu().numpy()
        x=x[np.isfinite(x)];self.n+=len(x);self.sum+=float(np.sum(x,dtype=np.float64));self.sum_sq+=float(np.dot(x.astype(np.float64),x.astype(np.float64)))
        # Approximate bounded sample; stochastic per-batch thinning prevents large CPU storage.
        if len(x):
            take=min(len(x),256)
            if len(x)>take:x=x[self.rng.choice(len(x),take,replace=False)]
            for z in x:
                if len(self.sample)<self.capacity:self.sample.append(float(z))
                else:
                    j=int(self.rng.integers(max(self.n,1)))
                    if j<self.capacity:self.sample[j]=float(z)
    def to_dict(self):
        if not self.n:return {'count':0}
        x=np.asarray(self.sample)
        return {'count':self.n,'mean':self.sum/self.n,'std':math.sqrt(max(0.,self.sum_sq/self.n-(self.sum/self.n)**2)),
                'p10_sample':float(np.quantile(x,.10)),'p50_sample':float(np.quantile(x,.50)),
                'p90_sample':float(np.quantile(x,.90)), 'quantiles_approximate':True}

def query_mae(pred,y,valid):
    v=valid.float();count=v.sum((1,2));abs_err=((pred.float()-y.float()).abs()*v).sum((1,2))
    return abs_err/count.clamp_min(1),count

def query_oracle(predictions,y,valid):
    """Oracle chooses one predictor per whole query; score with true valid-element counts."""
    maes,counts=zip(*(query_mae(p,y,valid) for p in predictions))
    errors=torch.stack(maes,dim=1);idx=errors.argmin(dim=1)
    chosen=torch.stack(predictions,dim=1)[torch.arange(len(idx),device=idx.device),idx]
    return chosen,idx

def element_oracle(predictions,y):
    stacked=torch.stack(predictions,dim=1)
    idx=(stacked-y[:,None]).abs().argmin(dim=1)
    return torch.gather(stacked,1,idx[:,None]).squeeze(1),idx

class PairwiseGain:
    def __init__(self):self.good_a=0;self.good_b=0;self.both_good=0;self.a_only=0;self.b_only=0;self.both_bad=0;self.den=0
    def add(self,base,a,b,y,valid):
        v=valid.bool();base_err=(base-y).abs();a_good=((a-y).abs()<base_err)&v;b_good=((b-y).abs()<base_err)&v
        self.good_a+=int(a_good.sum());self.good_b+=int(b_good.sum());self.both_good+=int((a_good&b_good).sum())
        self.a_only+=int((a_good&~b_good).sum());self.b_only+=int((b_good&~a_good).sum())
        self.both_bad+=int((~a_good&~b_good&v).sum());self.den+=int(v.sum())
    def to_dict(self):
        d=max(1,self.den)
        return {'valid_elements':self.den,'a_beneficial_fraction':self.good_a/d,'b_beneficial_fraction':self.good_b/d,
                'both_beneficial_fraction':self.both_good/d,'a_only_beneficial_fraction':self.a_only/d,
                'b_only_beneficial_fraction':self.b_only/d,'neither_beneficial_fraction':self.both_bad/d}

def gain_percent(baseline,candidate):
    if baseline is None or candidate is None or baseline<=0:return None
    return 100*(baseline-candidate)/baseline
