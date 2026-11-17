"""ReShift correction arms, fusion gates and the outcome-driven arm policy.

Arms are corrections added to the frozen Normal anchor P_N:
  cf         psi_d * Dhat_CF for every dose rung d              (parametric)
  ret_res    a_r * (r_ret - Rhat_N)                              (non-parametric residual)
  ret_shift  Dhat_CF + a_d * (d_ret - Dhat_CF)  [or a_d*d_ret without CF]
  ret_both   ret_res + ret_shift
  prior      MLP(tokens): a learned mask-conditioned correction WITHOUT retrieval
             (control for "retrieval is only a mask-dependent bias")
An explicit reject option (correction 0) is always available to the policy.
"""
from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F
from reshift.retrieval.stats import RET_FEATURE_DIM

ARM_TYPES=('cf','ret_res','ret_shift','ret_both','prior')
RET_ARMS=('ret_res','ret_shift','ret_both')
FUSION_EVAL_ORDER=('ret_both','ret_shift','ret_res','prior')


def validate_arms(arms,cf_enabled=True):
    arms=tuple(str(a) for a in arms)
    if not arms:raise ValueError('ReShift needs at least one arm')
    unknown=[a for a in arms if a not in ARM_TYPES]
    if unknown:raise ValueError(f'Unknown arms {unknown}; allowed {ARM_TYPES}')
    if len(set(arms))!=len(arms):raise ValueError(f'Duplicate arms {arms}')
    if 'cf' in arms and not cf_enabled:raise ValueError("arm 'cf' requires cf_enabled=true")
    return arms


def select_arm(gains,corrections,margin=0.0,temperature=None):
    """Pick a correction per element from predicted real gains.

    gains, corrections: [B,A,P,N].  Returns (applied [B,P,N], arm [B,P,N]) where
    arm=-1 means the element is left at the Normal anchor.
    Hard (evaluation): argmax arm, applied only if its gain exceeds `margin`.
    Soft (policy training): softmax over [reject(logit=margin)] + arms.
    """
    if gains.shape!=corrections.shape:raise ValueError('gains/corrections shape mismatch')
    if temperature is None:
        best,idx=gains.max(dim=1)
        accept=best>margin
        applied=corrections.gather(1,idx[:,None]).squeeze(1)*accept.to(corrections.dtype)
        return applied,torch.where(accept,idx,torch.full_like(idx,-1))
    if temperature<=0:raise ValueError('temperature must be positive')
    reject=torch.full_like(gains[:,:1],float(margin))
    weights=torch.softmax(torch.cat([reject,gains],dim=1)/float(temperature),dim=1)
    applied=(weights[:,1:]*corrections).sum(1)
    return applied,weights.argmax(1)-1


class RetrievalHead(nn.Module):
    def __init__(self,token_dim,arms,dose_levels=(1.0,),use_features=True,consistency_dim=0,
                 hidden=32,tau_init=0.1,cf_enabled=True):
        super().__init__()
        self.arms=validate_arms(arms,cf_enabled)
        self.cf_enabled=bool(cf_enabled)
        self.use_features=bool(use_features)
        self.consistency_dim=int(consistency_dim)
        self.levels=tuple(float(v) for v in dose_levels) if 'cf' in self.arms else ()
        if 'cf' in self.arms and (not self.levels or abs(self.levels[-1]-1.0)>1e-9):
            raise ValueError('cf arm levels must end with 1.0')
        kinds=[];levels=[]
        for a in self.arms:
            if a=='cf':
                kinds.extend(['cf']*len(self.levels));levels.extend(self.levels)
            else:
                kinds.append(a);levels.append(1.0)
        self.arm_kinds=tuple(kinds)
        self.register_buffer('arm_type_ids',torch.tensor([ARM_TYPES.index(k) for k in kinds],dtype=torch.long),
                             persistent=False)
        self.register_buffer('arm_level',torch.tensor(levels,dtype=torch.float32),persistent=False)
        self.has_retrieval_arms=any(a in RET_ARMS for a in self.arms)
        self.needs_evidence=self.use_features or self.has_retrieval_arms
        if tau_init<=0:raise ValueError('tau_init must be positive')
        self.log_tau=nn.Parameter(torch.tensor(math.log(float(tau_init))))
        if self.has_retrieval_arms:
            self.gate=nn.Sequential(nn.Linear(token_dim+RET_FEATURE_DIM,hidden),nn.GELU(),
                                    nn.Linear(hidden,2))
            nn.init.zeros_(self.gate[-1].weight);nn.init.constant_(self.gate[-1].bias,-1.0)
        else:self.gate=None
        if 'prior' in self.arms:
            self.prior=nn.Sequential(nn.Linear(token_dim,hidden),nn.GELU(),nn.Linear(hidden,1))
            nn.init.zeros_(self.prior[-1].weight);nn.init.zeros_(self.prior[-1].bias)
        else:self.prior=None
        numeric=len(ARM_TYPES)+4+self.consistency_dim+(RET_FEATURE_DIM if self.use_features else 0)
        self.policy_token=nn.Linear(token_dim,hidden)
        self.policy_numeric=nn.Linear(numeric,hidden)
        self.policy_out=nn.Sequential(nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.policy_out[-1].weight);nn.init.zeros_(self.policy_out[-1].bias)

    @property
    def has_fusion(self):
        return self.gate is not None or self.prior is not None

    def fusion_parameters(self):
        params=[self.log_tau]
        for m in (self.gate,self.prior):
            if m is not None:params+=list(m.parameters())
        return params

    def policy_parameters(self):
        return (list(self.policy_token.parameters())+list(self.policy_numeric.parameters())
                +list(self.policy_out.parameters()))

    def tau(self):
        return self.log_tau.exp()

    def fusion_eval_index(self):
        for kind in FUSION_EVAL_ORDER:
            if kind in self.arm_kinds:return self.arm_kinds.index(kind)
        return None

    def corrections(self,tokens,d_hat,r_hat,psi,stats):
        """tokens [B,P,N,dim]; d_hat/r_hat [B,P,N] (detached parametric CF shift and
        Normal residual); psi [B,len(levels),P,N] or None; stats from evidence_stats
        or None.  Returns corrections [B,A,P,N] and the fusion gates."""
        tokens=tokens.float();d_hat=d_hat.float();r_hat=r_hat.float()
        gates=None;out=[]
        if self.gate is not None:
            if stats is None:raise RuntimeError('retrieval arms need evidence')
            gates=torch.sigmoid(self.gate(torch.cat([tokens,stats['features'].float()],-1)))
            a_r=gates[...,0];a_d=gates[...,1]
            r_corr=a_r*(stats['r_ret'].float()-r_hat)
            d_f=(d_hat+a_d*(stats['d_ret'].float()-d_hat) if self.cf_enabled
                 else a_d*stats['d_ret'].float())
        for a in self.arms:
            if a=='cf':
                if psi is None or psi.shape[1]!=len(self.levels):
                    raise ValueError('psi must cover every cf dose level')
                out.extend(psi.float()[:,j]*d_hat for j in range(len(self.levels)))
            elif a=='ret_res':out.append(r_corr)
            elif a=='ret_shift':out.append(d_f)
            elif a=='ret_both':out.append(r_corr+d_f)
            elif a=='prior':out.append(self.prior(tokens).squeeze(-1))
        return torch.stack(out,dim=1),gates

    def gains(self,tokens,corrections,p_normal,consistency=None,features=None):
        """Predicted REAL gain over the Normal anchor of each arm, [B,A,P,N]."""
        B,A,P,N=corrections.shape
        if A!=len(self.arm_kinds):raise ValueError('arm count mismatch')
        tok=self.policy_token(tokens.float())[:,None]
        corr=corrections.float()
        onehot=F.one_hot(self.arm_type_ids,len(ARM_TYPES)).float().view(1,A,1,1,-1).expand(B,A,P,N,-1)
        level=self.arm_level.view(1,A,1,1,1).expand(B,A,P,N,1)
        parts=[onehot,level,corr[...,None],corr.abs()[...,None],
               p_normal.float()[:,None,...,None].expand(B,A,P,N,1)]
        if self.consistency_dim:
            if consistency is None:consistency=corr.new_zeros((B,P,N,self.consistency_dim))
            parts.append(consistency.float()[:,None].expand(B,A,P,N,self.consistency_dim))
        if self.use_features:
            if features is None:features=corr.new_zeros((B,P,N,RET_FEATURE_DIM))
            parts.append(features.float()[:,None].expand(B,A,P,N,RET_FEATURE_DIM))
        hidden=tok+self.policy_numeric(torch.cat(parts,-1))
        return self.policy_out(hidden).squeeze(-1)
