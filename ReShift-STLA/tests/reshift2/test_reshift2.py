"""Fast CPU unit tests for ReShift-STLA v0.2 with a tiny fake STLAformer."""
import sys
from pathlib import Path
import numpy as np, pytest, torch
from torch import nn
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from reshift2.config import load_config, apply_ablation
from reshift2.stla import MaskAwareSTLA, FrozenTeacher
from reshift2.model import ReShiftSTLA, EVAL_MODES
from reshift2.memory import build_memory_index
from reshift2.losses import compute_losses, alignment_loss
from reshift2.teacher import ForecastBank

N, H, P, M = 7, 12, 12, 64


class FakeLayer(nn.Module):
    def __init__(self, d): super().__init__(); self.l = nn.Linear(d, d)
    def forward(self, h, node_emb, time_emb): return h + torch.tanh(self.l(h + node_emb + time_emb))


class FakeSTLA(nn.Module):
    def __init__(self):
        super().__init__(); self.num_nodes = N; self.in_steps = H; self.out_steps = P; self.output_dim = 1
        self.steps_per_week = 2016; self.input_dim = 1; self.tow_embedding_dim = 4; self.model_dim = 12
        self.input_proj = nn.Linear(1, 8); self.tow_embedding = nn.Embedding(2016, 4)
        self.node_position_encoding = lambda z: torch.zeros_like(z); self.time_position_encoding = lambda z: torch.zeros_like(z)
        self.MLP = nn.Linear(12, 12); self.attn_layers_st = nn.ModuleList([FakeLayer(12), FakeLayer(12)])
        self.output_proj = nn.Linear(H * 12, P)


class FakeRepo:
    def __init__(self):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(M, H, N, generator=g); self.y = torch.randn(M, P, N, generator=g)
        self.tod = torch.randint(0, 288, (M, H), generator=g); self.dow = torch.randint(0, 7, (M, H), generator=g)
        self.starts = np.arange(M, dtype=np.int64) * 30; self._device = None
    def __len__(self): return M
    def batch(self, s, e, device=None): return self.x[s:e], torch.ones_like(self.x[s:e]), self.tod[s:e], self.dow[s:e]
    def fetch(self, idx, device): i = idx.long(); return self.x[i], torch.ones_like(self.x[i]), self.tod[i], self.dow[i]
    def fetch_future(self, idx, device): return self.y[idx.long()]
    def fetch_future_valid(self, idx, device): return torch.ones_like(self.y[idx.long()], dtype=torch.bool)


def make(ablation='full'):
    torch.manual_seed(0); cfg = apply_ablation(load_config(), ablation)
    cfg['retrieval']['top_k'] = 3; cfg['replay']['top_r'] = 2; cfg['retrieval']['graph_neighbors'] = 3
    base = FakeSTLA(); model = ReShiftSTLA(MaskAwareSTLA(base), FrozenTeacher(base), N, H, P, cfg)
    model.set_graph(torch.stack([(torch.arange(N) + k) % N for k in (1, 2, 3)], 1))
    repo = FakeRepo(); index = build_memory_index(model.memory_encoder, repo, 'cpu', 16)
    with torch.no_grad():
        pf = torch.cat([model.teacher(*repo.batch(s, s + 16)) for s in range(0, M, 16)]).numpy()
    return cfg, model, repo, index, ForecastBank(pf, 'cpu')


def batch(repo, B=4):
    g = torch.Generator().manual_seed(1); rows = torch.tensor([3, 17, 33, 50][:B])
    mask = (torch.rand(B, H, N, generator=g) > 0.5).float(); mask[0, :, 0] = 0  # one fully unobserved node
    yn = repo.y[rows]
    return {'x': repo.x[rows] * mask, 'mask': mask, 'tod': repo.tod[rows], 'dow': repo.dow[rows],
            'window_start': torch.as_tensor(repo.starts[rows.numpy()]), 'y_norm': yn, 'y_raw': yn * 2 + 1,
            'future_valid': torch.ones(B, P, N, dtype=torch.bool), 'direct_mask': torch.ones(B, N, dtype=torch.bool),
            'cf_target': torch.randn(B, P, N, generator=g) * .1, 'cf_target_valid': torch.ones(B, P, N, dtype=torch.bool)}


def test_initial_prediction_is_base_plus_gated_branches_and_cf_starts_at_zero():
    cfg, model, repo, index, pf = make(); model.eval(); out = model(batch(repo), index, repo, pf)
    assert torch.allclose(out['cf_shift'], torch.zeros_like(out['cf_shift']))
    torch.testing.assert_close(out['prediction'], out['base_prediction'] + out['contributions'].sum(-1))


def test_mask_projection_zero_init_reproduces_teacher():
    cfg, model, repo, index, pf = make(); b = batch(repo)
    with torch.no_grad():
        torch.testing.assert_close(model.stla(b['x'], b['mask'], b['tod'], b['dow'])[0],
                                   model.teacher(b['x'], b['mask'], b['tod'], b['dow']))


def test_forward_never_reads_labels():
    cfg, model, repo, index, pf = make(); model.eval(); b = batch(repo)
    o1 = model(b, index, repo, pf)['prediction']
    b2 = dict(b, y_norm=torch.randn_like(b['y_norm']), y_raw=torch.randn_like(b['y_raw']), cf_target=torch.randn_like(b['cf_target']))
    torch.testing.assert_close(o1, model(b2, index, repo, pf)['prediction'])


def test_retrieved_windows_never_overlap_query():
    cfg, model, repo, index, pf = make(); b = batch(repo)
    q = model.query_encoder(b['x'], b['mask'], b['tod'], b['dow'])
    idx, *_ = model._candidates(b, q, index, repo, 'full')
    gap = (torch.as_tensor(repo.starts)[idx] - b['window_start'][:, None]).abs()
    assert bool((gap >= H + P).all())


@pytest.mark.parametrize('mode', EVAL_MODES)
def test_eval_modes(mode):
    cfg, model, repo, index, pf = make(); model.eval(); out = model(batch(repo), index, repo, pf, mode=mode)
    assert torch.isfinite(out['prediction']).all()
    if mode == 'base': torch.testing.assert_close(out['prediction'], out['base_prediction'])
    if mode == 'drop_cf': assert float(out['gates'][..., 2].abs().max()) == 0.0


@pytest.mark.parametrize('ablation', ['full', 'base_only', 'no_evolution', 'no_replay', 'no_cf', 'fusion_sum', 'fusion_learned'])
def test_losses_backward(ablation):
    cfg, model, repo, index, pf = make(ablation); model.train(); model.set_train_phase('partial', 1); b = batch(repo)
    out = model(b, index, repo, pf)
    al = alignment_loss(model, b, repo, 'cpu', .1, out) if model.uses_retrieval else None
    loss, parts = compute_losses(out, b, 1.0, 2.0, cfg, al)
    from reshift2.losses import parts_dict; parts = parts_dict(parts)
    loss.backward(); assert torch.isfinite(loss)
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(p.grad is None for p in model.memory_encoder.parameters())
    if model.use_cf: assert model.cf_head[-1].weight.grad is not None and parts['cf_distill'] > 0


def test_factorized_scorer_equals_mlp_on_concat():
    from reshift2.model import FactorizedScorer
    torch.manual_seed(0); f = FactorizedScorer(3, 2, 5, 8).eval()
    e = torch.randn(2, 4, 6, 7, 3); c = torch.randn(2, 4, 1, 7, 2); x = torch.randn(2, 1, 6, 7, 5)
    cat = torch.cat([e, c.expand(2, 4, 6, 7, 2), x.expand(2, 4, 6, 7, 5)], -1)
    W = torch.cat([f.elem.weight, f.cand.weight, f.ctx.weight], 1)
    ref = f.out(cat @ W.T + f.elem.bias).squeeze(-1)
    torch.testing.assert_close(f(e, c, x), ref)


def test_bank_replay_path_matches_online_inputs():
    cfg, model, repo, index, pf = make(); model.eval(); b = batch(repo)
    q = model.query_encoder(b['x'], b['mask'], b['tod'], b['dow'])
    shift, sc = model._replay_online(b['x'], b['mask'], model._candidates(b, q, index, repo, 'full'), pf)
    o1 = model(dict(b, replay_shift=shift, replay_score=sc), index, repo, pf)['prediction']
    o2 = model(b, index, repo, pf)['prediction']
    torch.testing.assert_close(o1, o2)


def test_masked_shortlist_is_legal_and_uses_only_observed_entries():
    from reshift2.replay_bank import masked_shortlist
    repo = FakeRepo(); g = torch.Generator().manual_seed(3)
    m = (torch.rand(5, H, N, generator=g) > .5).float(); xq = repo.x[:5] * m; sq = repo.starts[:5]
    ids, sc = masked_shortlist(xq, m, sq, repo.x, repo.starts, 3, H + P, embargo=60)
    gap = (torch.as_tensor(repo.starts)[ids] - torch.as_tensor(sq)[:, None]).abs()
    assert bool((gap >= H + P + 60).all())
    # changing unobserved entries of the query must not change the shortlist
    ids2, _ = masked_shortlist(xq + (1 - m) * 7.0, m, sq, repo.x, repo.starts, 3, H + P, embargo=60)
    assert torch.equal(ids, ids2)


def test_wrong_evidence_indices_are_legal_and_batch_invariant():
    from reshift2.wrong_evidence import legal_wrong_indices
    starts = np.arange(M) * 30; q = torch.tensor([0, 300, 900])
    a = legal_wrong_indices(q, 5, starts, 24, 'cpu'); b = legal_wrong_indices(q[1:], 5, starts, 24, 'cpu')
    assert torch.equal(a[1:], b) and bool(((torch.as_tensor(starts)[a] - q[:, None]).abs() >= 24).all())
