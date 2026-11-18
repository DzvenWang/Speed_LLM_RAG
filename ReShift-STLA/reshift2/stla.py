"""Missing-aware STLAformer wrapper and the frozen restoration teacher.

`MaskAwareSTLA` is the deployed (fine-tuned) backbone: the pretrained STLAformer
receives an additive, zero-initialised observation-pattern embedding, so at
initialisation it is exactly the pretrained model.  `FrozenTeacher` is a second
copy of the *unchanged* pretrained model.  It only ever runs under no_grad and
defines the restoration shift  T(x_full) - T(x_obs)  used by the replay and
counterfactual branches.
"""
from __future__ import annotations
import copy, importlib.util, sys
from pathlib import Path
import torch
from torch import nn


def _load_module(path: Path):
    if str(path.parent) not in sys.path: sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location('reshift2_vendor_stla', path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def load_pretrained_stla(project_root: Path, checkpoint: Path, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False); mc = dict(ck['model_config'])
    src = Path(project_root) / 'Backbone' / 'vendor' / 'STLAformer' / 'model' / 'STLformer.py'
    if str(ck.get('model')) != 'STLAformer': raise RuntimeError('Backbone checkpoint is not STLAformer')
    if not src.is_file(): raise FileNotFoundError(src)
    mod = _load_module(src)
    b = mod.STLformer(num_nodes=int(ck['num_nodes']), in_steps=12, out_steps=12,
                      steps_per_week=int(mc.get('steps_per_week', 2016)), input_dim=int(mc.get('input_dim', 2)),
                      output_dim=1, input_embedding_dim=int(mc.get('input_embedding_dim', 32)),
                      tow_embedding_dim=int(mc.get('tow_embedding_dim', 32)),
                      feed_forward_dim=int(mc.get('feed_forward_dim', 256)), num_heads=int(mc.get('num_heads', 4)),
                      num_layers=int(mc.get('num_layers', 6)), dropout=float(mc.get('dropout', 0.3)))
    b.load_state_dict(ck['state_dict'], strict=True)
    return b.to(device), ck


def missing_age(mask):
    """Steps since the last observation, normalised by history length; [B,H,N]."""
    B, H, N = mask.shape; r = torch.zeros(B, N, device=mask.device, dtype=mask.dtype); out = []
    for t in range(H): r = (r + 1) * (1 - mask[:, t]); out.append(r)
    return torch.stack(out, 1) / max(H, 1)


class MaskAwareSTLA(nn.Module):
    def __init__(self, base):
        super().__init__(); self.base = base; d = int(base.model_dim)
        self.mask_proj = nn.Sequential(nn.Linear(4, d), nn.GELU(), nn.Linear(d, d))
        nn.init.zeros_(self.mask_proj[-1].weight); nn.init.zeros_(self.mask_proj[-1].bias)

    @property
    def model_dim(self): return int(self.base.model_dim)

    def set_train_phase(self, phase, last_layers=2):
        for p in self.base.parameters(): p.requires_grad = (phase == 'full')
        if phase == 'partial':
            for p in self.base.output_proj.parameters(): p.requires_grad = True
            for layer in list(self.base.attn_layers_st)[-int(last_layers):]:
                for p in layer.parameters(): p.requires_grad = True
        for p in self.mask_proj.parameters(): p.requires_grad = True

    def _tow(self, tod, dow):
        slot = dow.long() * 288 + tod.long(); f = slot.float() / float(self.base.steps_per_week)
        return torch.nextafter(f, torch.full_like(f, float('inf')))

    def forward(self, x, mask, tod, dow):
        B, H, N = x.shape; b = self.base
        tow = self._tow(tod, dow); tow_grid = tow.unsqueeze(-1).expand(B, H, N)
        raw = torch.stack([x, tow_grid], -1); feat = [b.input_proj(raw[..., :b.input_dim])]
        if b.tow_embedding_dim > 0: feat.append(b.tow_embedding((tow_grid * b.steps_per_week).long()))
        z = torch.cat(feat, -1)
        nc = mask.mean(1, keepdim=True).expand(-1, H, -1); tc = mask.mean(2, keepdim=True).expand(-1, -1, N)
        z = z + self.mask_proj(torch.stack([mask, nc, tc, missing_age(mask)], -1))
        node_emb = b.node_position_encoding(z); time_emb = b.time_position_encoding(z)
        bx = b.MLP(z); h = z - bx
        for a in b.attn_layers_st: h = a(h, node_emb, time_emb)
        h = h + bx
        out = h.transpose(1, 2).reshape(B, b.num_nodes, b.in_steps * b.model_dim)
        out = b.output_proj(out).view(B, b.num_nodes, b.out_steps, b.output_dim).transpose(1, 2).squeeze(-1)
        return out, h.mean(1)


class FrozenTeacher(nn.Module):
    """Unchanged pretrained STLA; mask projection stays exactly zero."""

    def __init__(self, base):
        super().__init__(); self.net = MaskAwareSTLA(copy.deepcopy(base))
        for p in self.net.parameters(): p.requires_grad_(False)
        self.net.eval()

    def train(self, mode=True):
        super().train(False); self.net.eval(); return self

    @torch.no_grad()
    def forward(self, x, mask, tod, dow, chunk=256):
        outs = []
        for s in range(0, x.shape[0], chunk):
            outs.append(self.net(x[s:s + chunk], mask[s:s + chunk], tod[s:s + chunk], dow[s:s + chunk])[0])
        return torch.cat(outs, 0)
