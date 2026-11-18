"""ReShift-STLA v0.2: coverage-allocated evolution retrieval, failure replay and
counterfactual restoration on a missing-aware STLAformer.

    Y_hat = B + g_E(phi) * C_E + g_R(phi) * C_R + g_S(phi) * C_S

B    missing-aware fine-tuned STLAformer forecast of the incomplete query;
C_E  evolution retrieval: anchored future evolution of retrieved complete Train
     windows, reranked per node and horizon against a None option, risk-gated;
C_R  failure replay: restoration shifts of the same retrieved windows when the
     QUERY's own observation mask is replayed on them through a frozen teacher;
C_S  counterfactual restoration: a parametric estimate of the query's own
     restoration shift, distilled from the teacher on Train complete histories;
g    allocation gates that, in the default `coverage` mode, see ONLY
     label-free observability descriptors phi (coverage, neighbour coverage,
     missing age, horizon).

No query future and no complete query history is ever an input of forward().
All futures used as evidence come from the Train split.
"""
from __future__ import annotations
import copy
import torch
from torch import nn
from .encoders import StateEncoder
from .stla import MaskAwareSTLA, FrozenTeacher
from .seasonal import seasonal_innovation
from .wrong_evidence import legal_wrong_indices

EVAL_MODES = ('full', 'wrong_evidence', 'drop_evolution', 'drop_replay', 'drop_cf', 'base')


def weighted_dispersion(weights, values):
    """Stable sqrt of the weighted variance over candidates (dim 1); FP32."""
    with torch.autocast(device_type=weights.device.type, enabled=False):
        w = weights.float(); v = values.float()
        mean = (w * v).sum(1, keepdim=True)
        var = (w * (v - mean).square()).sum(1).clamp_min(0.0)
        return (torch.sqrt(var + 1e-6) - 1e-3).clamp(0.0, 10.0)


def _mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, o))


class FactorizedScorer(nn.Module):
    """MLP over [element | candidate | context] features without materialising the concat.

    The first layer is linear, so W [e; c; x] = W_e e + W_c c + W_x x.  Candidate
    features live on [B,C,1,N], context features on [B,1,P,N]; only the few
    element features need the full [B,C,P,N] grid.  Mathematically identical to
    _mlp on the concatenation, but the large input tensor and most of the
    first-layer FLOPs disappear.
    """

    def __init__(self, elem_dim, cand_dim, ctx_dim, hidden, dropout=0.0):
        super().__init__()
        self.elem = nn.Linear(elem_dim, hidden); self.cand = nn.Linear(cand_dim, hidden, bias=False)
        self.ctx = nn.Linear(ctx_dim, hidden, bias=False)
        self.out = nn.Sequential(nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, elem, cand, ctx):
        h = self.elem(elem) + self.cand(cand) + self.ctx(ctx)
        return self.out(h).squeeze(-1)


class ReShiftSTLA(nn.Module):
    def __init__(self, stla: MaskAwareSTLA, teacher: FrozenTeacher, num_nodes, history, horizon, cfg):
        super().__init__()
        ec = cfg['encoder']; mc = cfg['model']; rc = cfg['retrieval']; pc = cfg['replay']
        fc = cfg['fusion']; cc = cfg['counterfactual']
        br = mc.get('branches', {})
        self.use_evolution = bool(br.get('evolution', True)); self.use_replay = bool(br.get('replay', True))
        self.use_cf = bool(br.get('cf', True)); self.use_risk = bool(mc.get('use_risk_gate', True))
        self.H = int(history); self.P = int(horizon); self.N = int(num_nodes)
        self.top_k = int(rc['top_k']); self.temperature = float(rc['temperature']); self.top_r = int(pc['top_r'])
        self.max_delta = float(mc.get('max_normalized_correction', 3.0))
        self.fusion_mode = str(fc['mode'])
        td, sd, qd, rd = (int(ec[k]) for k in ('temporal_dim', 'spatial_dim', 'query_dim', 'retrieval_dim'))
        graph_k = min(int(rc.get('graph_neighbors', 8)), max(self.N - 1, 1))
        self.stla = stla; self.teacher = teacher
        self.query_encoder = StateEncoder(self.N, history, td, sd, qd, rd, int(ec['spatial_heads']),
                                          float(ec['dropout']), graph_k)
        self.memory_encoder = copy.deepcopy(self.query_encoder)
        for p in self.memory_encoder.parameters(): p.requires_grad_(False)
        mid = int(mc.get('node_hidden', 16)); hid = int(mc.get('fusion_hidden', 96)); drop = float(mc['dropout'])
        self.qsummary = nn.Sequential(nn.Linear(qd, mid), nn.GELU())
        self.horizon_embed = nn.Embedding(self.P, 8)
        ctx_dim = 5 + mid + 8      # base, cov, neighbour cov, missing age, global cov, qsum, horizon
        sh = int(mc.get('scorer_hidden', 48))
        # evolution retrieval (anchored Train futures): elem 3, candidate 3, context ctx_dim
        self.evo_scorer = FactorizedScorer(3, 3, ctx_dim, sh, drop)
        self.evo_none = _mlp(ctx_dim, hid // 2, 1)
        self.evo_risk = _mlp(ctx_dim + 2, hid // 2, 1)
        # failure replay (teacher restoration shifts of the same Train windows)
        # failure replay: elem 2 (shift, |shift|), candidate 2 (mean |shift|, score), context
        self.rep_scorer = FactorizedScorer(2, 2, ctx_dim, sh, drop)
        self.rep_none = _mlp(ctx_dim, hid // 2, 1)
        self.rep_risk = _mlp(ctx_dim + 2, hid // 2, 1)
        # counterfactual restoration head
        cfd = int(cc.get('hidden', 64))
        self.cf_latent = nn.Sequential(nn.Linear(stla.model_dim, cfd), nn.GELU())
        self.cf_query = nn.Sequential(nn.Linear(qd, cfd), nn.GELU())
        self.cf_head = _mlp(5 + 2 * cfd + 8, hid, 1, drop)
        nn.init.zeros_(self.cf_head[-1].weight); nn.init.zeros_(self.cf_head[-1].bias)
        for m in (self.evo_risk, self.rep_risk):
            nn.init.zeros_(m[-1].weight); nn.init.constant_(m[-1].bias, 1.5)
        nb = float(mc.get('none_bias', 0.8))
        for m in (self.evo_none, self.rep_none): nn.init.constant_(m[-1].bias, nb)
        # allocation gates
        phi_dim = 4 + 8 + (7 if self.fusion_mode == 'learned' else 0)
        self.allocator = _mlp(phi_dim, int(fc.get('hidden', 32)), 3)
        nn.init.zeros_(self.allocator[-1].weight)
        with torch.no_grad(): self.allocator[-1].bias.copy_(torch.tensor([float(b) for b in fc['init_bias']]))
        self.register_buffer('hids', torch.arange(self.P), persistent=False)
        self.register_buffer('future_reference', torch.zeros(168, self.P, self.N), persistent=True)

    # ------------------------------------------------------------------ setup
    @property
    def uses_retrieval(self): return self.use_evolution

    def set_future_reference(self, reference):
        if tuple(reference.shape) != tuple(self.future_reference.shape):
            raise ValueError('Train-only future reference dimensions disagree')
        self.future_reference.copy_(reference.to(self.future_reference.device))

    def set_graph(self, neighbours):
        self.query_encoder.set_graph(neighbours); self.memory_encoder.set_graph(neighbours)

    @torch.no_grad()
    def momentum_update(self, momentum=0.995):
        for m, q in zip(self.memory_encoder.parameters(), self.query_encoder.parameters()):
            m.mul_(momentum).add_(q, alpha=1. - momentum)

    def set_train_phase(self, phase, last_layers=2):
        self.stla.set_train_phase(phase, last_layers)
        for p in self.teacher.parameters(): p.requires_grad_(False)
        for p in self.memory_encoder.parameters(): p.requires_grad_(False)

    def train(self, mode=True):
        # The pretrained STLA keeps its dropout disabled while it is fine-tuned.
        super().train(mode); self.stla.base.eval(); self.teacher.eval(); self.memory_encoder.eval(); return self

    # --------------------------------------------------------------- branches
    def _candidates(self, batch, q, index, repo, mode):
        x = batch['x']; B = x.shape[0]; span = self.H + self.P
        idx_list = []; score_list = []
        for kind in ('temporal', 'spatial'):
            ind, sc, _, _ = index.search(q[f'{kind}_key'].detach(), kind, self.top_k,
                                         batch['window_start'].long(), span, self.temperature)
            idx_list.append(ind); score_list.append(sc)
        idx = torch.cat(idx_list, 1); scores = torch.cat(score_list, 1)
        if mode == 'wrong_evidence':
            # Same query, same scores, but unrelated (still legal) Train windows.
            idx = legal_wrong_indices(batch['window_start'], idx.shape[1], repo.starts, span, x.device)
        gx, _, gt, gd = repo.fetch(idx, x.device)
        return idx, scores.float(), gx, gt, gd

    def _evolution(self, x, mask, base, q, ctx, cand, repo, mode):
        idx, scores, gx, gt, gd = cand
        B, H, N = x.shape; P = self.P; C = idx.shape[1]
        gf = repo.fetch_future(idx, x.device); gv = repo.fetch_future_valid(idx, x.device)
        with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
            mem = self.memory_encoder(gx.reshape(B * C, H, N).float(), torch.ones_like(gx).reshape(B * C, H, N),
                                      gt.reshape(B * C, H), gd.reshape(B * C, H))
        node_sim = torch.nn.functional.cosine_similarity(q['node_query'][:, None].float(),
                                                         mem['node_query'].reshape(B, C, N, -1).float(), dim=-1)
        # Anchor each candidate's future evolution at the query's last OBSERVED step.
        ix = (H - 1 - torch.argmax(mask.flip(1), dim=1)).clamp_min(0); observed = mask.any(1)
        xk_last = torch.gather(gx, 2, ix[:, None, None, :].expand(B, C, 1, N)).squeeze(2)
        future_delta = gf - xk_last[:, :, None, :]
        last = torch.gather(x, 1, ix[:, None]).squeeze(1)
        neigh = self.query_encoder.neighbors; ncv = observed[:, neigh].float()
        local_anchor = (last[:, neigh] * ncv).sum(2) / ncv.sum(2).clamp_min(1.)
        anchor = torch.where(observed, last, torch.where(ncv.sum(2) > 0, local_anchor, base[:, 0].detach()))
        cand_pred = anchor[:, None, None, :] + future_delta
        bucket = ((gd[:, :, -1] % 7) * 24 + (gt[:, :, -1] % 288) // 12)[:, :, None].expand(B, C, N)
        innovation = seasonal_innovation(gf, bucket, self.future_reference)
        no_obs = ~observed[:, None, None, :]
        cand_pred = torch.where(no_obs, base[:, None] + innovation, cand_pred)
        residual = torch.where(gv, torch.nan_to_num(cand_pred - base[:, None]), torch.zeros_like(cand_pred))
        dist = ((gx - x[:, None]).abs() * mask[:, None]).sum(2) / mask[:, None].sum(2).clamp_min(1.)
        elem = torch.stack([cand_pred, residual, torch.where(no_obs, innovation, future_delta)], -1)
        cand_f = torch.stack([dist, scores[:, :, None].expand(B, C, N), node_sim], -1)[:, :, None]   # [B,C,1,N,3]
        logits = self.evo_scorer(elem, cand_f, ctx['feat'][:, None]) + scores[:, :, None, None]
        logits = logits.masked_fill(~gv, -1e4)
        all_logits = torch.cat([logits, self.evo_none(ctx['feat']).squeeze(-1)[:, None]], 1)
        w = torch.softmax(all_logits.float(), 1)
        evidence = (w[:, :C] * residual.float()).sum(1)
        disp = weighted_dispersion(w[:, :C], residual)
        risk_logit = self.evo_risk(torch.cat([ctx['feat'], disp[..., None], (1 - w[:, -1])[..., None]], -1)).squeeze(-1)
        risk = torch.sigmoid(risk_logit) if self.use_risk else torch.ones_like(risk_logit)
        ungated = self.max_delta * torch.tanh(evidence / self.max_delta)
        return {'correction': risk * ungated, 'ungated': ungated, 'logits': all_logits, 'residuals': residual,
                'available': gv, 'risk_logit': risk_logit, 'none_weight': w[:, -1], 'dispersion': disp}

    def _replay_online(self, x, mask, cand, pfull):
        """Teacher replay of the query mask on R retrieved rows (audit / no-bank fallback)."""
        idx, scores, gx, gt, gd = cand
        B, H, N = x.shape; R = self.top_r; P = self.P
        rid = idx[:, :R]; rx = gx[:, :R]; rt = gt[:, :R]; rd = gd[:, :R]
        qm = mask[:, None].expand(B, R, H, N)
        with torch.no_grad(), torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=x.device.type == 'cuda'):
            pm = self.teacher((rx * qm).reshape(B * R, H, N), qm.reshape(B * R, H, N),
                              rt.reshape(B * R, H), rd.reshape(B * R, H)).float().reshape(B, R, P, N)
        return torch.nan_to_num(pfull.fetch(rid, x.device) - pm), scores[:, :R]

    def _replay(self, ctx, shift, rscore):
        """Failure replay: rerank precomputed restoration shifts S_c of R Train windows."""
        shift = shift.float(); rscore = rscore.float()
        B, R, P, N = shift.shape
        elem = torch.stack([shift, shift.abs()], -1)
        cand_f = torch.stack([shift.abs().mean(2), rscore[:, :, None].expand(B, R, N)], -1)[:, :, None]
        logits = self.rep_scorer(elem, cand_f, ctx['feat'][:, None]) + rscore[:, :, None, None]
        all_logits = torch.cat([logits, self.rep_none(ctx['feat']).squeeze(-1)[:, None]], 1)
        w = torch.softmax(all_logits.float(), 1)
        evidence = (w[:, :R] * shift).sum(1)
        disp = weighted_dispersion(w[:, :R], shift)
        risk_logit = self.rep_risk(torch.cat([ctx['feat'], disp[..., None], (1 - w[:, -1])[..., None]], -1)).squeeze(-1)
        risk = torch.sigmoid(risk_logit) if self.use_risk else torch.ones_like(risk_logit)
        ungated = self.max_delta * torch.tanh(evidence / self.max_delta)
        return {'correction': risk * ungated, 'ungated': ungated, 'logits': all_logits, 'residuals': shift,
                'available': torch.ones_like(shift, dtype=torch.bool), 'risk_logit': risk_logit,
                'none_weight': w[:, -1], 'dispersion': disp}

    def _counterfactual(self, z, q, ctx):
        B, P, N = ctx['base'].shape
        lat = self.cf_latent(z)[:, None].expand(B, P, N, -1)
        qq = self.cf_query(q['node_query'])[:, None].expand(B, P, N, -1)
        raw = self.cf_head(torch.cat([ctx['feat'][..., :5], lat, qq, ctx['h']], -1)).squeeze(-1)
        return self.max_delta * torch.tanh(raw / self.max_delta)

    # ---------------------------------------------------------------- forward
    def forward(self, batch, index=None, repo=None, pfull=None, mode='full'):
        if mode not in EVAL_MODES: raise ValueError(f'Unknown mode {mode}')
        x = batch['x'].float(); mask = batch['mask'].float(); tod = batch['tod'].long(); dow = batch['dow'].long()
        B, H, N = x.shape
        with torch.autocast(device_type=x.device.type, enabled=False):
            q = self.query_encoder(x, mask, tod, dow)
        base, z = self.stla(x, mask, tod, dow); P = base.shape[1]
        cov = q['node_coverage'][:, None, :, None].expand(B, P, N, 1)
        ncov = q['neighbour_coverage'][:, None, :, None].expand(B, P, N, 1)
        age = q['missing_age_last'][:, None, :, None].expand(B, P, N, 1)
        gcov = mask.mean((1, 2))[:, None, None, None].expand(B, P, N, 1)
        qsum = self.qsummary(q['node_query'])[:, None].expand(B, P, N, -1)
        h = self.horizon_embed(self.hids)[None, :, None].expand(B, P, N, -1)
        ctx = {'base': base, 'cov': cov, 'ncov': ncov, 'qsum': qsum, 'h': h,
               'feat': torch.cat([base[..., None], cov, ncov, age, gcov, qsum, h], -1)}
        zero = torch.zeros_like(base, dtype=torch.float32)
        evo = rep = None; c_cf = zero
        if self.use_evolution and mode != 'base':
            if index is None or repo is None: raise ValueError('Evolution retrieval needs a memory index and repository')
            index.to(x.device)
            cand = self._candidates(batch, q, index, repo, mode)
            evo = self._evolution(x, mask, base, q, ctx, cand, repo, mode)
        if self.use_replay and mode != 'base':
            if mode == 'wrong_evidence' or 'replay_shift' not in batch:
                # Online path: audits with unrelated Train windows, or runs without a bank.
                if repo is None or pfull is None: raise ValueError('Online replay needs repository and teacher bank')
                if mode == 'wrong_evidence':
                    wid = legal_wrong_indices(batch['window_start'], self.top_r, repo.starts, self.H + self.P, x.device)
                    gx, _, gt, gd = repo.fetch(wid, x.device)
                    sc = batch['replay_score'].float() if 'replay_score' in batch else torch.zeros(B, self.top_r, device=x.device)
                    shift, rscore = self._replay_online(x, mask, (wid, sc, gx, gt, gd), pfull)
                else:
                    if index is None: raise ValueError('Online replay without a bank needs the memory index')
                    shift, rscore = self._replay_online(x, mask, self._candidates(batch, q, index.to(x.device), repo, mode), pfull)
            else:
                shift, rscore = batch['replay_shift'], batch['replay_score']
            rep = self._replay(ctx, shift, rscore)
        if self.use_cf and mode != 'base': c_cf = self._counterfactual(z, q, ctx).float()
        c_evo = evo['correction'] if evo is not None else zero
        c_rep = rep['correction'] if rep is not None else zero
        phi = [cov, ncov, age, gcov, h]
        if self.fusion_mode == 'learned':
            nw = lambda b: b['none_weight'][..., None] if b is not None else zero[..., None]
            dp = lambda b: b['dispersion'][..., None] if b is not None else zero[..., None]
            phi += [c_evo.abs()[..., None], c_rep.abs()[..., None], c_cf.abs()[..., None], nw(evo), nw(rep), dp(evo), dp(rep)]
        gates = (torch.ones(B, P, N, 3, device=x.device) if self.fusion_mode == 'sum'
                 else torch.sigmoid(self.allocator(torch.cat([t.float() for t in phi], -1).float())))
        drop = {'drop_evolution': 0, 'drop_replay': 1, 'drop_cf': 2}.get(mode)
        if drop is not None: gates = gates.clone(); gates[..., drop] = 0.0
        contrib = torch.stack([c_evo, c_rep, c_cf], -1) * gates
        correction = contrib.sum(-1)
        if mode == 'base': correction = zero
        return {'prediction': base + correction.to(base.dtype), 'base_prediction': base, 'correction': correction,
                'contributions': contrib, 'gates': gates, 'evolution': evo, 'replay': rep, 'cf_shift': c_cf,
                'query_temporal_keys': q['temporal_key'], 'query_spatial_keys': q['spatial_key']}
