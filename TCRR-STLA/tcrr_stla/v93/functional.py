"""Train-only functional neighbourhood: exact v7 pairwise-valid covariance formula."""
import torch

def make_functional_neighbours(x,valid,k=8,min_overlap=64):
    if x.ndim!=3 or valid.shape!=x.shape:raise ValueError('Expected [Q,L,N] aligned tensors')
    Q,L,N=x.shape
    if N<2:return torch.zeros((N,1),dtype=torch.long),torch.zeros((N,1),dtype=torch.bool)
    a=torch.nan_to_num(x.float()).reshape(-1,N)
    m=valid.float().reshape(-1,N)
    v=a*m
    cnt=m.T@m
    sx=v.T@m
    sx2=(v*v).T@m
    xy=v.T@v
    cov=xy-sx*sx.T/cnt.clamp_min(1)
    var=(sx2-sx.square()/cnt.clamp_min(1)).clamp_min(0)
    denom=(var*var.T).sqrt()
    corr=(cov/denom.clamp_min(1e-6)*cnt/(cnt+min_overlap)).clamp(-1,1)
    corr=corr.masked_fill(cnt<min_overlap,-1e9)
    corr.diagonal().fill_(-1e9)
    score,ids=torch.topk(corr,min(k,N-1),dim=1)
    return ids.long().cpu(),(score>-1e8).cpu()
