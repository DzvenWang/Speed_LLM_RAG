from __future__ import annotations
from pathlib import Path
import json, numpy as np, torch

class LatentMemoryIndex:
    """Train-only retrieval with explicit temporal and numerical validity checks.

    Retrieval similarity is always computed in FP32, even when the caller uses
    CUDA FP16 autocast. An invalid key is never treated as a legal candidate.
    """

    def __init__(self, temporal_keys, spatial_keys, starts, region_keys=None):
        self.temporal_np = np.asarray(temporal_keys, dtype=np.float32)
        self.spatial_np = np.asarray(spatial_keys, dtype=np.float32)
        self.starts_np = np.asarray(starts, dtype=np.int64)
        self.region_np = None if region_keys is None else np.asarray(region_keys, dtype=np.float32)
        count = len(self.starts_np)
        if count == 0 or self.temporal_np.shape[0] != count or self.spatial_np.shape[0] != count:
            raise ValueError('Retrieval index row count disagrees with Train-only starts')
        if self.region_np is not None and self.region_np.shape[0] != count:
            raise ValueError('Regional index row count disagrees with Train-only starts')
        self._device = None
        self.tk = self.sk = self.rk = self.starts = None
        self.tk_valid = self.sk_valid = self.rk_valid = None

    def to(self, device):
        if self._device != str(device):
            self.starts = torch.from_numpy(self.starts_np).to(device)
            self.tk_valid = torch.from_numpy(np.isfinite(self.temporal_np).all(axis=1)).to(device)
            self.sk_valid = torch.from_numpy(np.isfinite(self.spatial_np).all(axis=1)).to(device)
            self.tk = torch.nn.functional.normalize(torch.nan_to_num(torch.from_numpy(self.temporal_np).to(device)), dim=-1)
            self.sk = torch.nn.functional.normalize(torch.nan_to_num(torch.from_numpy(self.spatial_np).to(device)), dim=-1)
            if self.region_np is not None:
                self.rk_valid = torch.from_numpy(np.isfinite(self.region_np).all(axis=-1)).to(device)
                self.rk = torch.nn.functional.normalize(torch.nan_to_num(torch.from_numpy(self.region_np).to(device)), dim=-1)
            else:
                self.rk_valid = self.rk = None
            self._device = str(device)
            bad_t = len(self.starts_np) - int(self.tk_valid.sum().item())
            bad_s = len(self.starts_np) - int(self.sk_valid.sum().item())
            bad_r = 0 if self.rk_valid is None else self.rk_valid.numel() - int(self.rk_valid.sum().item())
            print(f'MEMORY_INDEX_HEALTH rows={len(self.starts_np)} bad_temporal={bad_t} bad_spatial={bad_s} bad_regional={bad_r}', flush=True)
        return self

    def _legal(self, query_starts, span):
        qs = torch.as_tensor(query_starts, dtype=torch.long, device=self.starts.device).flatten()
        if int(span) <= 0:
            raise ValueError('History + horizon overlap exclusion span must be positive')
        overlap = ((self.starts[None, :] < (qs + int(span))[:, None]) &
                   (qs[:, None] < (self.starts + int(span))[None, :]))
        return ~overlap

    @staticmethod
    def _validate_query(q, label):
        if not bool(torch.isfinite(q).all().item()):
            bad = int((~torch.isfinite(q).all(dim=-1)).sum().item())
            raise RuntimeError(f'Non-finite {label} query encoding in {bad} rows; inspect Query Encoder numerical stability')

    @staticmethod
    def _select(sim, k, kind, legal_counts):
        if k <= 0 or k > sim.shape[-1]:
            raise ValueError(f'{kind}: invalid top_k={k} for memory rows={sim.shape[-1]}')
        # The caller masks invalid numerical keys AND temporally overlapping rows.
        scores, idx = torch.topk(sim, k=k, dim=-1)
        if not bool(torch.isfinite(scores).all().item()):
            insufficient = int((legal_counts < k).sum().item())
            raise RuntimeError(
                f'{kind}: fewer than top_k={k} finite, temporally nonoverlapping candidates '
                f'for {insufficient} query/region rows (min_available={int(legal_counts.min().item())}). '
                'Do not relax temporal exclusion; check Train memory validity and encoder health.'
            )
        return scores, idx

    def search(self, q, key_type, top_k, query_starts, span, temperature=0.1):
        if self._device is None:
            self.to(q.device)
        if key_type not in ('temporal', 'spatial'):
            raise ValueError(f'Unknown global retrieval branch: {key_type}')
        keys = self.tk if key_type == 'temporal' else self.sk
        key_valid = self.tk_valid if key_type == 'temporal' else self.sk_valid
        with torch.autocast(device_type=q.device.type, enabled=False):
            query = q.float()
            self._validate_query(query, key_type)
            query = torch.nn.functional.normalize(query, dim=-1)
            sim = query @ keys.float().T
            legal = self._legal(query_starts, span) & key_valid[None, :]
            sim = sim.masked_fill(~legal, -float('inf'))
            k = int(top_k)
            scores, idx = self._select(sim, k, key_type, legal.sum(dim=-1))
            weights = torch.softmax(scores / max(float(temperature), 1e-6), dim=1)
            entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(1)
            entropy = entropy / np.log(max(weights.shape[1], 2))
            margin = scores[:, 0] - scores[:, 1] if k > 1 else torch.zeros_like(scores[:, 0])
            confidence = torch.stack([scores[:, 0], margin, 1. - entropy], dim=-1)
        return idx, scores, weights, confidence

    def search_regions(self, q, top_k, query_starts, span):
        if self._device is None:
            self.to(q.device)
        if self.rk is None:
            raise RuntimeError('v2.3 regional memory index is missing')
        if q.ndim != 3 or q.shape[1] != self.rk.shape[1]:
            raise ValueError('Query region layout disagrees with stored memory')
        with torch.autocast(device_type=q.device.type, enabled=False):
            query = q.float()
            self._validate_query(query, 'regional')
            query = torch.nn.functional.normalize(query, dim=-1)
            sim = torch.einsum('brd,mrd->brm', query, self.rk.float())
            legal = self._legal(query_starts, span)[:, None, :] & self.rk_valid.T[None, :, :]
            sim = sim.masked_fill(~legal, -float('inf'))
            scores, idx = self._select(sim, int(top_k), 'regional', legal.sum(dim=-1))
        return idx, scores

    def save(self, path, metadata=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, temporal_keys=self.temporal_np, spatial_keys=self.spatial_np,
                            starts=self.starts_np, region_keys=self.region_np)
        path.with_suffix('.json').write_text(json.dumps(metadata or {}, indent=2), encoding='utf-8')

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        return cls(z['temporal_keys'], z['spatial_keys'], z['starts'],
                   z['region_keys'] if 'region_keys' in z.files else None)

def build_memory_index(encoder,repo,device,batch_size=64):
    encoder.eval();t=[];s=[];r=[]
    with torch.no_grad():
        for st in range(0,len(repo),batch_size):
            en=min(st+batch_size,len(repo));x,m,tod,dow=repo.batch(st,en,device);e=encoder(x,m,tod,dow);t.append(e['temporal_key'].cpu());s.append(e['spatial_key'].cpu());r.append(e['region_keys'].cpu())
    return LatentMemoryIndex(torch.cat(t).numpy(),torch.cat(s).numpy(),repo.starts.copy(),torch.cat(r).numpy())
