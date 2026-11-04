"""C3: gain and harm predictions; default action is the unchanged STLA forecast."""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

class UtilityController(nn.Module):
    def __init__(self,d,hidden=32,temperature=.5,risk_cost=.05):
        super().__init__();self.temperature=float(temperature);self.risk_cost=float(risk_cost)
        self.net=nn.Sequential(nn.Linear(d+8,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU())
        self.gain=nn.Linear(hidden,2)
        self.risk=nn.Linear(hidden,2)
        nn.init.zeros_(self.gain.weight);nn.init.zeros_(self.gain.bias)
        nn.init.zeros_(self.risk.weight);nn.init.zeros_(self.risk.bias)
    def forward(self,context,y0,local,prop,coverage,availability,support_score):
        x=torch.cat((context,y0[...,None],local[...,None],prop[...,None],coverage,
                     availability[:,None,:,:].expand(-1,y0.shape[1],-1,-1),
                     local.abs()[...,None],prop.abs()[...,None],support_score),-1)
        h=self.net(x)
        gain=self.gain(h)
        risk=self.risk(h)
        cost=torch.sigmoid(risk)*self.risk_cost
        zeros=torch.zeros_like(gain[...,:1])
        logits=torch.cat((zeros,(gain-cost)/self.temperature),-1)
        weights=torch.softmax(logits,-1)
        return {'weights':weights,'predicted_gain':gain,'predicted_harm_logits':risk}


def supervised_utility_loss(out,batch,weight_gain=.1,weight_risk=.05,weight_action=.1):
    pred=out['candidates']
    valid=batch['future_valid'].bool() & torch.isfinite(batch['y_norm'])
    target=torch.where(valid,batch['y_norm'],torch.zeros_like(batch['y_norm']))[...,None]
    abs_err=(pred.float()-target.float()).abs()
    gain=(abs_err[...,0:1]-abs_err[...,1:]).detach()
    harm=(abs_err[...,1:]>abs_err[...,0:1]+1e-5).float().detach()
    util=out['policy']
    v=valid[...,None].expand_as(gain)
    gp=F.smooth_l1_loss(util['predicted_gain'].float(),gain,reduction='none')
    rp=F.binary_cross_entropy_with_logits(util['predicted_harm_logits'].float(),harm,reduction='none')
    best=abs_err.detach().argmin(-1)
    ac=F.nll_loss(torch.log(util['weights'].float().clamp_min(1e-8)).reshape(-1,3),best.reshape(-1),reduction='none').reshape_as(valid)
    n=v.sum().clamp_min(1)
    return weight_gain*(gp*v).sum()/n+weight_risk*(rp*v).sum()/n+weight_action*(ac*valid).sum()/valid.sum().clamp_min(1)
