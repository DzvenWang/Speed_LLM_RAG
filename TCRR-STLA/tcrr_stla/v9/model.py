"""TCRR-STLA v9.1: horizon-local CF-aware utility with a fixed Normal anchor.

Teacher data NEVER enters ``forward``. The Normal pathway can be frozen after its
own stage; graph reliability gates historical graph evidence only, and zero CF
response entails exactly zero extra correction.
"""
from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from .graph import GraphBundle

V9_VERSION = '9.1.0'


class TCRRV9(nn.Module):
    def __init__(self, graph: GraphBundle, context_dim: int, hidden: int = 32,
                 bound: float = 3., blocks: int = 3, cf_init_strength: float = .5):
        super().__init__()
        if blocks < 1 or bound <= 0: raise ValueError('Invalid blocks or bound')
        self.graph = graph
        self.hidden, self.bound, self.blocks = int(hidden), float(bound), int(blocks)
        self.cf_init_strength = float(cf_init_strength)
        self.query = nn.Sequential(nn.Linear(5 + 9 + context_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.match = nn.Sequential(nn.Linear(2 + 7, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        # Only the Normal branch uses this encoder; CF training cannot alter it.
        self.normal_fusion = nn.Sequential(nn.Linear(hidden * 2 + 3, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.normal_scale = nn.Linear(hidden, blocks)
        self.none_normal = nn.Parameter(torch.tensor(-1.))
        # CF pathway reads the *target node's* Normal/CF history in each horizon block.
        self.temporal = nn.Sequential(nn.Conv1d(2, 12, 3, padding=1), nn.GELU(), nn.Conv1d(12, 1, 3, padding=1))
        self.cf_local = nn.Sequential(nn.Linear(9, 16), nn.GELU(), nn.Linear(16, 16), nn.GELU())
        # Query and historical graph sources have different meanings; 3 response
        # messages are multiplied by the three actually observable graph masses.
        self.cf_fusion = nn.Sequential(nn.Linear(hidden * 2 + 16 + 6, hidden), nn.GELU(),
                                       nn.Linear(hidden, hidden), nn.GELU())
        self.amplitude = nn.Linear(hidden, 1)
        self.utility = nn.Linear(hidden, 1)
        self.fullinfo = nn.Linear(hidden, blocks)
        self.none = nn.Parameter(torch.tensor(0.))
        self.register_buffer('temp', torch.tensor(.1))
        nn.init.zeros_(self.amplitude.weight)
        nn.init.zeros_(self.amplitude.bias)
        nn.init.zeros_(self.normal_scale.weight)
        nn.init.zeros_(self.normal_scale.bias)

    def bound_fn(self, x):
        return self.bound * torch.tanh(x / self.bound)

    @staticmethod
    def failure_features(mask):
        b, l, n = mask.shape
        m = mask.float()
        recent = m[:, -min(3, l):].mean(1)
        coverage = m.mean(1)
        age = torch.zeros(b, n, device=m.device, dtype=m.dtype)
        longest = torch.zeros_like(age)
        for t in range(l):
            age = (age + 1) * (1 - m[:, t])
            longest = torch.maximum(longest, age)
        transition = (m[:, 1:] - m[:, :-1]).abs().mean(1) if l > 1 else torch.zeros_like(age)
        return torch.stack([coverage, recent, age / l, longest / l, transition], -1)

    def normal_anchor(self, base, normal, query_z, match_feat, score, valid):
        b, k, h, n = normal.shape
        q = query_z[:, None].expand(-1, k, -1, -1)
        norm_stats = torch.stack([
            normal.mean(2), normal.abs().mean(2), normal.std(2, unbiased=False),
            normal[:, :, :max(1, h // 3)].mean(2), normal[:, :, -max(1, h // 3):].mean(2),
            score[:, :, None].expand(-1, -1, n), valid[:, :, None].to(normal.dtype).expand(-1, -1, n)], -1)
        kz = self.match(torch.cat([match_feat, norm_stats], -1))
        null_graph = torch.zeros(b, k, n, 3, dtype=q.dtype, device=q.device)
        scale = 1 + .5 * torch.tanh(self.normal_scale(self.normal_fusion(torch.cat([q, kz, null_graph], -1))))
        scale = scale.permute(0, 1, 3, 2).repeat_interleave(h // self.blocks, dim=2)
        logits = (score[:, :, None] + .1 * match_feat[..., 0]).masked_fill(~valid[:, :, None], -1e4)
        null = self.none_normal.expand(b, 1, n)
        w = torch.softmax(torch.cat([logits, null], 1).float(), dim=1)[:, :k].to(normal.dtype)
        w = w * valid[:, :, None].to(w.dtype)
        raw = (w[:, :, None, :] * normal * scale).sum(1)
        return raw, kz, w

    def _cf_block_features(self, normal, cf):
        """Return block-local features [B,K,G,N,9]; order-sensitive by construction."""
        b, k, h, n = cf.shape
        span = h // self.blocks
        d = cf.reshape(b, k, self.blocks, span, n)
        r = normal.reshape(b, k, self.blocks, span, n)
        conv_input = torch.stack((normal, cf), -1).permute(0, 1, 3, 4, 2).reshape(b*k*n, 2, h)
        response = self.temporal(conv_input).reshape(b, k, n, h).permute(0, 1, 3, 2)
        t = response.reshape(b, k, self.blocks, span, n)
        return torch.stack((d.mean(3), d.abs().mean(3), d.std(3, unbiased=False),
                            d[:, :, :, -1] - d[:, :, :, 0], r.mean(3), r.abs().mean(3),
                            (d*r).mean(3), t.mean(3), t[:, :, :, -1] - t[:, :, :, 0]), -1)

    def forward(self, *, base, context, x, mask, normal, cf, score, valid, match=None,
                enable_cf=True, cf_strength=1.0, rank_mode='learned',
                recoverability_enabled=True, cf_graph_enabled=True, graph_enabled=True):
        if base.ndim != 3: raise ValueError('base must be [B,H,N]')
        b, h, n = base.shape
        if normal.ndim != 4: raise ValueError('normal must be [B,K,H,N]')
        k = normal.shape[1]
        if (n != self.graph.nodes or normal.shape != (b,k,h,n) or cf.shape != normal.shape
                or h % self.blocks != 0): raise ValueError('candidate/base shapes mismatch')
        if context.shape[:2] != (b,n) or x.shape != mask.shape or x.shape[0] != b or x.shape[-1] != n:
            raise ValueError('query shapes mismatch')
        if score.shape != (b,k) or valid.shape != (b,k): raise ValueError('shortlist shapes mismatch')
        valid = valid.bool()
        if match is None: match = torch.zeros(b,k,n,2,device=base.device,dtype=base.dtype)
        if match.shape != (b,k,n,2): raise ValueError('candidate match mismatch')
        if rank_mode not in ('learned','uniform'): raise ValueError('Invalid CF ranking mode')
        if cf_strength < 0: raise ValueError('CF strength must be nonnegative')
        graph_q = self.graph.query_summary(x, mask) if graph_enabled else torch.zeros(b,n,9,device=x.device,dtype=x.dtype)
        q = self.query(torch.cat([self.failure_features(mask), graph_q.to(context.dtype), context], -1))
        recoverability = self.fullinfo(q).permute(0,2,1)  # [B,G,N]
        nraw, kz, wn = self.normal_anchor(base, normal, q, match, score, valid)
        normal_prediction = base + self.bound_fn(nraw)
        if not enable_cf:
            zero = torch.zeros_like(normal_prediction)
            return dict(base=base, normal_prediction=normal_prediction, prediction=normal_prediction,
                        cf_increment=zero, normal_raw=nraw, recoverability=recoverability,
                        normal_weights=wn)
        span = h // self.blocks
        local = self.cf_local(self._cf_block_features(normal, cf))  # [B,K,G,N,16]
        qg = q[:,None,None].expand(-1,k,self.blocks,-1,-1)
        kg = kz[:,:,None].expand(-1,-1,self.blocks,-1,-1)
        # Three directional graph mass terms are query-observable and cannot
        # be confused with hidden full-data signals.
        reliability = graph_q[..., [1,4,7]].clamp(0,1).to(q.dtype)
        rel = reliability[:,None,None].expand(-1,k,self.blocks,-1,-1)
        if graph_enabled and cf_graph_enabled:
            from_graph = self.graph.cf_summary(cf).reshape(b,k,self.blocks,span,n,3).mean(3).to(q.dtype)
            from_graph = from_graph * rel
        else:
            from_graph = torch.zeros(b,k,self.blocks,n,3,device=q.device,dtype=q.dtype)
        z = self.cf_fusion(torch.cat([qg,kg,local,from_graph,rel], -1))
        amplitude = 1 + .5*torch.tanh(self.amplitude(z).squeeze(-1))  # [B,K,G,N]
        utility = self.utility(z).squeeze(-1).float()  # [B,K,G,N]
        amp_h = amplitude.repeat_interleave(span,dim=2)
        candidate_cf = cf * amp_h
        if rank_mode == 'learned':
            logits = utility/self.temp.float().clamp_min(.025) + score[:,:,None,None].float()
            if recoverability_enabled: logits = logits + .1*recoverability[:,None].float()
            null_logit = self.none.float()
        else:
            logits = torch.zeros_like(utility)
            null_logit = torch.zeros_like(self.none).float()
        logits = logits.masked_fill(~valid[:,:,None,None], -1e4)
        logits = torch.cat([logits, null_logit.expand(b,1,self.blocks,n)], 1)
        weights = torch.softmax(logits,dim=1)
        wc = weights[:,:k].repeat_interleave(span,dim=2)
        cfraw = (wc*candidate_cf).sum(1) * float(cf_strength)
        prediction = base + self.bound_fn(nraw + cfraw)
        return dict(base=base,normal_prediction=normal_prediction,prediction=prediction,
                    normal_raw=nraw, cf_raw=cfraw, cf_increment=prediction-normal_prediction,
                    bound=self.bound, candidate_cf=candidate_cf, cf_utility=utility,
                    cf_weights=weights, normal_weights=wn, amplitude=amplitude,
                    recoverability=recoverability, cf_graph_reliability=reliability)


def utility_targets(base, normal_raw, candidate_cf, target, valid, bound, blocks=3,
                    future_valid=None, return_valid=False):
    """TRAIN-only signed marginal utility; missing target steps NEVER contribute.

    Args are backward compatible with v9.0.0. Set return_valid=True to also
    obtain candidate/block/node supervision validity.
    """
    b,k,h,n = candidate_cf.shape
    if h % blocks: raise ValueError('blocks/horizon mismatch')
    with torch.no_grad():
        future = (torch.ones_like(target,dtype=torch.bool) if future_valid is None
                  else future_valid.bool())
        if future.shape != target.shape: raise ValueError('future_valid shape mismatch')
        target_safe = torch.where(future,target,torch.zeros_like(target))
        f = lambda z: bound*torch.tanh(z/bound)
        anchor = base + f(normal_raw)
        proposal = base[:,None] + f(normal_raw[:,None]+candidate_cf)
        per_horizon = (anchor[:,None]-target_safe[:,None]).abs() - (proposal-target_safe[:,None]).abs()
        per_horizon = torch.where(future[:,None],per_horizon,torch.zeros_like(per_horizon))
        span = h//blocks
        numerator = per_horizon.reshape(b,k,blocks,span,n).sum(3)
        counts = future.reshape(b,blocks,span,n).sum(2)
        out = numerator / counts[:,None].clamp_min(1)
        supervision_valid = (counts[:,None]>0) & valid[:,:,None,None].bool()
        out = torch.where(supervision_valid,out,torch.zeros_like(out))
        return (out,supervision_valid) if return_valid else out


def masked_mae(pred,y,valid):
    v = valid.bool()
    y = torch.where(v,y,torch.zeros_like(y))
    delta = torch.where(v,pred-y,torch.zeros_like(pred))
    return delta.abs().sum()/v.sum().clamp_min(1)


def train_loss(out, target, future_valid, candidate_valid, utility_weight=.02, rank_weight=.01,
               teacher_cf=None, full_info_weight=0., teacher_alignment_weight=0.):
    """Forecast loss plus candidate-local utility, node-local ranking, teacher supervision."""
    forecast = masked_mae(out['prediction'].float(),target.float(),future_valid)
    if 'cf_utility' not in out: return forecast,dict(forecast=forecast.detach())
    u, mask = utility_targets(out['base'].float(),out['normal_raw'].float(),out['candidate_cf'].float(),
                              target.float(),candidate_valid,float(out.get('bound',3.0)),
                              out['cf_utility'].shape[2],future_valid=future_valid,return_valid=True)
    reg = F.smooth_l1_loss(out['cf_utility'].float(),u,reduction='none')
    reg = torch.where(mask,reg,torch.zeros_like(reg)).sum()/mask.sum().clamp_min(1)
    # Node-local class: candidate with greatest positive marginal gain, or Null.
    best = u.masked_fill(~mask,-1e6)
    null = torch.full_like(best[:,:1],1e-6)
    teacher = torch.cat([best,null],1).argmax(1)  # [B,G,N]
    valid_location = future_valid.bool().reshape(future_valid.shape[0],u.shape[2],-1,
                                                 future_valid.shape[-1]).any(2)
    logits = out['cf_weights'].float().clamp_min(1e-9).log()
    selected = logits.gather(1,teacher[:,None]).squeeze(1)
    rank = -(selected*valid_location).sum()/valid_location.sum().clamp_min(1)
    total = forecast + utility_weight*reg + rank_weight*rank
    full_loss = forecast.new_zeros(())
    align_loss = forecast.new_zeros(())
    if teacher_cf is not None and teacher_cf.numel() and (full_info_weight>0 or teacher_alignment_weight>0):
        if teacher_cf.shape != target.shape: raise ValueError('Teacher response/target shapes differ')
        valid_f = future_valid.bool()
        clean_y = torch.where(valid_f,target.float(),torch.zeros_like(target.float()))
        bt = out['base'].float(); teacher = teacher_cf.float()
        # Full-observation supervision is relative to the *already corrected*
        # Normal anchor, not to the base forecast. Otherwise CF is trained
        # to duplicate the part of the correction already supplied by Normal.
        normal_pred = out['normal_prediction'].float()
        full_teacher_pred = bt + teacher
        actual_gain = (clean_y-normal_pred).abs() - (clean_y-full_teacher_pred).abs()
        actual_gain = torch.where(valid_f,actual_gain,torch.zeros_like(actual_gain))
        b,h,n=actual_gain.shape; g=out['recoverability'].shape[1];span=h//g
        group_valid=valid_f.reshape(b,g,span,n)
        gain_target=actual_gain.reshape(b,g,span,n).sum(2)/group_valid.sum(2).clamp_min(1)
        flags=group_valid.any(2)
        if full_info_weight>0:
            r = F.smooth_l1_loss(out['recoverability'].float(),gain_target.detach(),reduction='none')
            full_loss=torch.where(flags,r,torch.zeros_like(r)).sum()/flags.sum().clamp_min(1)
            total = total + full_info_weight*full_loss
        if teacher_alignment_weight>0:
            # Copy full-information response ONLY when it actually improves TRAIN
            # forecast at this point. Otherwise supervised correction is harmful.
            beneficial=(actual_gain>0)&valid_f
            align_to = full_teacher_pred - normal_pred
            # Increment must target ONLY the incremental residual needed to
            # reach the Full Teacher from our fixed Normal anchor.
            align_to = align_to.clamp(-float(out.get('bound',3.0)),float(out.get('bound',3.0)))
            distance=F.smooth_l1_loss(out['cf_increment'].float(),align_to.detach(),reduction='none')
            align_loss=torch.where(beneficial,distance,torch.zeros_like(distance)).sum()/beneficial.sum().clamp_min(1)
            total=total+teacher_alignment_weight*align_loss
    return total,dict(forecast=forecast.detach(),utility=reg.detach(),ranking=rank.detach(),
                      full_information=full_loss.detach(),teacher_alignment=align_loss.detach())


def cross_failure_pairing(ids_a,ids_b,valid_a,valid_b):
    """Return same-ID mapping; disallow duplicate valid candidates in either view."""
    for ids,v in ((ids_a,valid_a),(ids_b,valid_b)):
        matches_self=(ids[:,:,None]==ids[:,None,:]) & v[:,:,None] & v[:,None,:]
        diagonal=torch.eye(ids.shape[1],device=ids.device,dtype=torch.bool)[None]
        if bool((matches_self & ~diagonal).any()):
            raise ValueError('Duplicate valid history candidate ID')
    matches=(ids_a[:,:,None]==ids_b[:,None,:])&valid_a[:,:,None]&valid_b[:,None,:]
    lookup=matches.float().argmax(-1)
    paired=matches.any(-1)
    return lookup,paired


def paired_cross_failure_loss(out_a,out_b,ids_a,ids_b,target_a,target_b,valid_a,valid_b,
                              future_valid,weight=1.0,return_stats=False):
    """Predict how the *same* historical candidate's CF utility changes across masks.

    The reference view is detached; the optimized view receives gradients.
    """
    future_b = future_valid
    ua=utility_targets(out_a['base'].float(),out_a['normal_raw'].float(),out_a['candidate_cf'].float(),
                       target_a.float(),valid_a,float(out_a['bound']),out_a['cf_utility'].shape[2],
                       future_valid=future_valid)
    ub=utility_targets(out_b['base'].float(),out_b['normal_raw'].float(),out_b['candidate_cf'].float(),
                       target_b.float(),valid_b,float(out_b['bound']),out_b['cf_utility'].shape[2],
                       future_valid=future_b)
    lookup,paired=cross_failure_pairing(ids_a,ids_b,valid_a,valid_b)
    index=lookup[:,:,None,None].expand(-1,-1,ub.shape[2],ub.shape[3])
    ub_aligned=ub.gather(1,index)
    pb_aligned=out_b['cf_utility'].gather(1,index)
    true_delta=(ua-ub_aligned).detach()
    pred_delta=out_a['cf_utility'].detach()-pb_aligned
    group_valid=future_valid.bool().reshape(future_valid.shape[0],ub.shape[2],-1,
                                           future_valid.shape[-1]).any(2)
    mask=paired[:,:,None,None]&group_valid[:,None]
    d=F.smooth_l1_loss(pred_delta,true_delta,reduction='none')
    loss=torch.where(mask,d,torch.zeros_like(d)).sum()/mask.sum().clamp_min(1)
    if return_stats:
        total = int(valid_a.sum().item())
        matched = int(paired.sum().item())
        return weight*loss,dict(matched_candidates=matched,reference_valid_candidates=total,
                                matching_rate=matched/max(total,1))
    return weight*loss
