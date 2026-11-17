"""Mechanism helpers for CoRSTA v0.2.3.

B. Counterfactual Dose Ladder
   The full counterfactual P_full restores *all* missing evidence.  A dose d in
   (0,1] restores a nested random fraction d of the originally observed but
   currently missing entries.  The Student learns a monotone dose-response
   curve  psi_d = d ** gamma  (psi_1 = 1, psi_0 = 0) scaling its full-dose CF
   shift, and a policy that picks, per forecast element, the dose that
   maximizes its predicted *real* gain over the frozen Normal anchor.

A. View-consistency reliability
   Extra views drop a further fraction of the *currently observed* entries.
   If the CF correction is trustworthy the corrected forecasts of the views
   should agree with the primary corrected forecast MORE than the uncorrected
   forecasts do (the correction contracts cross-view disagreement).  The
   features below feed the dose policy; they never touch labels.

Everything here is pure tensor code with no model state so it can be tested
in isolation.
"""
from __future__ import annotations
import torch

CONSISTENCY_DIM=4


def validate_levels(levels):
    """Strictly increasing doses in (0,1] whose top rung is exactly 1.0."""
    lv=tuple(float(v) for v in levels)
    if not lv:
        raise ValueError('dose levels must be non-empty')
    if any(not (0.0<v<=1.0) for v in lv):
        raise ValueError(f'dose levels must lie in (0,1]: {lv}')
    if any(b<=a for a,b in zip(lv,lv[1:])):
        raise ValueError(f'dose levels must be strictly increasing: {lv}')
    if abs(lv[-1]-1.0)>1e-9:
        raise ValueError(f'top dose rung must be exactly 1.0 (full counterfactual): {lv}')
    return lv


def nested_restore_masks(mask,original_valid,levels,generator=None):
    """Masks for partial restoration, shape [B,D,L,N] with D=len(levels).

    Rung d restores each restorable entry (originally observed, currently
    missing) when a shared uniform draw u<d.  Sharing u makes the rungs nested:
    rung d1 < d2 implies mask_d1 <= mask_d2 elementwise.  Entries that were
    never observed are never restored, so every rung is a subset of
    original_valid and a superset of mask.
    """
    mask=mask.float();original_valid=original_valid.float()
    restorable=(original_valid>0.5)&(mask<0.5)
    u=torch.rand(mask.shape,device=mask.device,generator=generator)
    rungs=[torch.where(restorable&(u<float(d)),torch.ones_like(mask),mask) for d in levels]
    return torch.stack(rungs,dim=1)


def dose_psi(levels,gamma):
    """Monotone dose-response fractions psi_d = d**gamma, shape [B,D,P,N].

    levels: [D] tensor in (0,1]; gamma: [B,P,N] with gamma>0.  psi_1 == 1 and
    psi is increasing in d for every element, by construction.
    """
    d=levels.to(device=gamma.device,dtype=gamma.dtype).view(1,-1,1,1)
    return d.pow(gamma[:,None])


def consistency_features(base0,delta0,basek,deltak,eps=1e-3):
    """Per-element view-consistency features, shape [B,P,N,CONSISTENCY_DIM].

    base0/delta0: primary Normal forecast and CF correction, [B,P,N].
    basek/deltak: the same for K degraded views, [K,B,P,N].

    0 log1p(10*|corrected_k - corrected_0|)    corrected disagreement
    1 log1p(10*|normal_k    - normal_0|)       uncorrected disagreement
    2 (uncorrected - corrected) / (uncorrected + eps), clipped to [-3,1]
      positive = the correction pulls the views together (reliable)
    3 fraction of views whose correction has the primary correction's sign
    """
    base0=base0.detach().float();delta0=delta0.detach().float()
    basek=basek.detach().float();deltak=deltak.detach().float()
    cand0=base0+delta0;candk=basek+deltak
    f_cand=(candk-cand0[None]).abs().mean(0)
    f_base=(basek-base0[None]).abs().mean(0)
    contraction=((f_base-f_cand)/(f_base+eps)).clamp(-3.0,1.0)
    agree=(torch.sign(deltak)==torch.sign(delta0)[None]).to(base0.dtype).mean(0)
    return torch.stack([torch.log1p(10.0*f_cand),torch.log1p(10.0*f_base),
                        contraction,agree],dim=-1)


def select_dose(gains,psi,levels,margin=0.0,temperature=None):
    """Choose a dose per element from predicted real gains.

    gains, psi: [B,D,P,N].  Returns (psi_sel, level_sel), both [B,P,N].
    Hard (temperature None, used at evaluation): argmax over doses; the element
    is left uncorrected (psi 0, level 0) unless the best predicted gain exceeds
    `margin`.
    Soft (temperature>0, used while the policy is trained): softmax over the
    rungs plus an explicit "no correction" option whose logit is `margin`.
    """
    lv=levels.to(device=gains.device,dtype=gains.dtype)
    if temperature is None:
        best,idx=gains.max(dim=1)
        accept=(best>margin).to(psi.dtype)
        psi_sel=psi.gather(1,idx[:,None]).squeeze(1)*accept
        level_sel=lv[idx]*accept
        return psi_sel,level_sel
    if temperature<=0:
        raise ValueError('temperature must be positive')
    reject=torch.full_like(gains[:,:1],float(margin))
    weights=torch.softmax(torch.cat([reject,gains],dim=1)/float(temperature),dim=1)
    psi_ext=torch.cat([torch.zeros_like(psi[:,:1]),psi],dim=1)
    lv_ext=torch.cat([lv.new_zeros(1),lv]).view(1,-1,1,1)
    return (weights*psi_ext).sum(1),(weights*lv_ext).sum(1)
