from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F


class DualQueryRanker(nn.Module):
    def __init__(self,input_dim,d_model=128,hidden=256,dropout=0.1):
        super().__init__()
        self.q=nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,d_model))
        self.k=nn.Sequential(nn.Linear(input_dim,hidden),nn.GELU(),nn.Dropout(dropout),nn.Linear(hidden,d_model))
        self.scale=math.sqrt(float(d_model))
    def forward(self,qdesc,cdesc):
        q=F.normalize(self.q(qdesc),dim=-1); k=F.normalize(self.k(cdesc),dim=-1)
        return (q[:,None,:]*k).sum(-1)


def soft_teacher(distance,temperature=0.2):
    d=torch.nan_to_num(distance,nan=1e4,posinf=1e4,neginf=0.0); s=-d
    z=(s-s.mean(-1,keepdim=True))/s.std(-1,keepdim=True).clamp_min(1e-4)
    return torch.softmax(z/max(float(temperature),1e-4),dim=-1)


def ranker_loss(pred_score,teacher_prob):
    return -(teacher_prob*F.log_softmax(pred_score,dim=-1)).sum(-1).mean()
