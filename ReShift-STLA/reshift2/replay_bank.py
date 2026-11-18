"""Precomputed failure-replay and counterfactual-target banks.

Online replay costs R frozen-teacher forwards per query and dominated the step
time.  Because benchmark masks are fixed per split row, the replay evidence can
be computed ONCE per (split, mask file, teacher) and shared by every run, seed
and ablation:

  shortlist   R legal Train windows ranked by masked cosine similarity of the
              OBSERVED query history (label-free, independent of learned keys)
  shift       S_c = T(X_c, 1) - T(X_c * M_q, M_q)                [Q,R,P,N] fp16
  score       masked cosine similarity                            [Q,R]

For Train queries the counterfactual target  T(X_q,1) - T(X_q*M_q, M_q)  is
cached as T(X_q*M_q, M_q) ("pobs"); T(X_q,1) is the shared complete-history bank.
"""
from __future__ import annotations
import hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from .config import file_sha256


def _key(path):
    p = Path(path); st = p.stat(); return {'path': str(p.resolve()), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


class RowBank:
    """Row-indexed array, on GPU when it fits, else memory-mapped."""

    def __init__(self, arrays: dict, device, split_rows: int):
        self.np = {k: v for k, v in arrays.items() if k != 'rows'}; self.gpu = {}
        d = torch.device(device)
        pos = np.full(int(split_rows), -1, dtype=np.int64); pos[arrays['rows']] = np.arange(len(arrays['rows']))
        self.pos_np = pos; self.pos = torch.from_numpy(pos).to(d)
        arrays = self.np
        if d.type == 'cuda':
            try:
                for k, a in arrays.items(): self.gpu[k] = torch.from_numpy(np.array(a, copy=True)).to(d)
                gib = sum(t.numel() * t.element_size() for t in self.gpu.values()) / 1024 ** 3
                print(f'RESHIFT_BANK_GPU keys={list(arrays)} GiB={gib:.3f}', flush=True)
            except torch.cuda.OutOfMemoryError:
                self.gpu = {}; torch.cuda.empty_cache(); print('RESHIFT_BANK_CPU_FALLBACK', flush=True)

    def has(self, key): return key in self.np

    def fetch(self, key, split_index, device):
        """Fetch bank rows for split row indices (raises if a row was not banked)."""
        p = self.pos[split_index.long().to(self.pos.device)]
        if bool((p < 0).any()): raise KeyError(f'split rows missing from the {key} bank')
        if key in self.gpu: return self.gpu[key][p]
        ids = p.detach().cpu().numpy()
        return torch.from_numpy(np.array(self.np[key][ids], copy=True)).to(device)


@torch.no_grad()
def masked_shortlist(xq, mq, sq, mem_x, mem_starts, R, span, embargo=0, chunk=1024):
    """Top-R legal memory rows by cosine similarity restricted to the query's observed entries.

    xq, mq [Q,L,N] (xq already zero at missing entries); mem_x [M,L,N] complete; sq [Q] starts.
    """
    Q = xq.shape[0]; dev = mem_x.device; dt = torch.float16 if dev.type == 'cuda' else torch.float32
    X = mem_x.reshape(len(mem_x), -1).to(dt); X2 = X * X
    starts = torch.as_tensor(mem_starts, device=dev, dtype=torch.long)
    ids = torch.empty(Q, R, dtype=torch.long, device=dev); sc = torch.empty(Q, R, dtype=torch.float32, device=dev)
    gap = int(span) + int(embargo)
    for s in range(0, Q, chunk):
        e = min(Q, s + chunk)
        a = (xq[s:e] * mq[s:e]).reshape(e - s, -1).to(dev, dt); m = mq[s:e].reshape(e - s, -1).to(dev, dt)
        num = (a @ X.T).float(); den = ((a.float() ** 2).sum(1, keepdim=True) * (m @ X2.T).float()).clamp_min(1e-6).sqrt()
        sim = num / den
        illegal = (starts[None, :] - torch.as_tensor(sq[s:e], device=dev, dtype=torch.long)[:, None]).abs() < gap
        sim = sim.masked_fill(illegal, -float('inf'))
        v, i = torch.topk(sim, R, dim=1)
        if not bool(torch.isfinite(v).all()): raise RuntimeError('Fewer than R legal replay candidates; lower embargo')
        ids[s:e] = i; sc[s:e] = v
    return ids, sc


def prepare_query_bank(split_name, ds, teacher, repo, pfull, backbone, store, device, R, span, embargo=0,
                       with_cf=False, rows=None, smoke=False, batch=64):
    """Build or load the replay (R>0) and/or CF-observed (with_cf) bank for `rows` of split `ds`.

    Bank position k corresponds to split row rows[k]; use RowBank.position(index) to map.
    """
    store = Path(store); store.mkdir(parents=True, exist_ok=True)
    rows = np.arange(len(ds), dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
    meta = {'schema': 'reshift2-replay-v2', 'split': split_name, 'rows': int(len(rows)),
            'rows_sha': hashlib.sha256(rows.tobytes()).hexdigest()[:16], 'R': int(R), 'span': int(span),
            'embargo': int(embargo), 'with_cf': bool(with_cf), 'memory_rows': int(len(repo)),
            'masks': _key(Path(ds.root) / f'{ds.split}_history_observed.npy'),
            'history': _key(Path(ds.root) / f'{ds.split}_x_obs_norm.npy'), 'backbone_sha256': file_sha256(backbone)}
    digest = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()[:12]
    tag = f"{split_name}_R{R}_e{embargo}_{digest}"
    path = store / f'replay_{tag}.npz'; mfile = path.with_suffix('.json')
    if path.is_file() and mfile.is_file() and json.loads(mfile.read_text()) == meta:
        z = np.load(path, allow_pickle=False); print(f'RESHIFT_REPLAY_BANK_HIT {split_name} path={path}', flush=True)
        return RowBank({k: z[k] for k in z.files}, device, len(ds))
    t0 = time.time(); n = len(rows); d = torch.device(device); R = int(R)
    print(f'RESHIFT_REPLAY_BANK_BUILD {split_name} queries={n} R={R} embargo={embargo} cf={with_cf}', flush=True)
    xs = torch.from_numpy(np.array(ds.x_obs[rows], np.float32)); ms = torch.from_numpy(np.array(ds.observed[rows], np.float32))
    starts = ds.meta.window_start.to_numpy()[rows].astype(np.int64)
    tod = torch.from_numpy(np.array(ds.tod[rows], np.int64)); dow = torch.from_numpy(np.array(ds.dow[rows], np.int64))
    P = pfull.array.shape[1]; N = pfull.array.shape[2]; amp = d.type == 'cuda'
    arrays = {'rows': rows}
    if R > 0:
        mem = repo.x_device if repo.x_device is not None else torch.from_numpy(np.array(repo.x[:len(repo)], np.float32)).to(d)
        ids, sc = masked_shortlist(xs, ms, starts, mem, repo.starts, R, span, embargo)
        shift = np.empty((n, R, P, N), np.float16)
    if with_cf: pobs = np.empty((n, P, N), np.float16)
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(n, s + batch); B = e - s
            with torch.autocast(device_type=d.type, dtype=torch.float16, enabled=amp):
                if R > 0:
                    rid = ids[s:e]; cx, _, ct, cd = repo.fetch(rid, d); qm = ms[s:e].to(d)[:, None].expand(B, R, *ms.shape[1:])
                    pm = teacher((cx * qm).reshape(B * R, *cx.shape[2:]), qm.reshape(B * R, *qm.shape[2:]),
                                 ct.reshape(B * R, -1), cd.reshape(B * R, -1)).float().reshape(B, R, P, N)
                    shift[s:e] = torch.nan_to_num(pfull.fetch(rid, d) - pm).cpu().numpy()
                if with_cf:
                    pobs[s:e] = teacher(xs[s:e].to(d), ms[s:e].to(d), tod[s:e].to(d), dow[s:e].to(d)).float().cpu().numpy()
            if (s // batch) % 100 == 0:
                print(f'RESHIFT_REPLAY_BANK_PROGRESS {split_name} {e}/{n} elapsed={time.time() - t0:.0f}s', flush=True)
    if R > 0: arrays.update(ids=ids.cpu().numpy(), score=sc.cpu().numpy(), shift=shift)
    if with_cf: arrays['pobs'] = pobs
    tmp = path.with_name(path.name + '.partial.npz'); np.savez(tmp, **arrays); os.replace(tmp, path)
    mfile.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'RESHIFT_REPLAY_BANK_PASS {split_name} seconds={time.time() - t0:.1f} path={path}', flush=True)
    return RowBank(arrays, device, len(ds))
