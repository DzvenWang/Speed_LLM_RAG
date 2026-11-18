"""Train-only bank of frozen-teacher complete-history forecasts  T(x_full).

The restoration shift of a window under an observation mask m is
    s(x, m) = T(x_full) - T(x_full * m, m).
T(x_full) does not depend on the mask, so it is computed once for every Train
memory row and cached on disk; only the masked forecast is recomputed online.
"""
from __future__ import annotations
import json, os, time
from pathlib import Path
import numpy as np
import torch
from .config import file_sha256


class ForecastBank:
    def __init__(self, array: np.ndarray, device):
        self.array = array; self.gpu = None
        d = torch.device(device)
        if d.type == 'cuda':
            try:
                self.gpu = torch.from_numpy(np.array(array, copy=True)).to(d)
                print(f'RESHIFT_PFULL_GPU GiB={self.gpu.numel() * self.gpu.element_size() / 1024 ** 3:.3f}', flush=True)
            except torch.cuda.OutOfMemoryError:
                self.gpu = None; torch.cuda.empty_cache(); print('RESHIFT_PFULL_CPU_FALLBACK', flush=True)

    def __len__(self): return len(self.array)

    def fetch(self, indices, device):
        if self.gpu is not None: return self.gpu[indices.long().to(self.gpu.device)].float()
        ids = indices.detach().cpu().numpy().astype(np.int64)
        return torch.from_numpy(np.array(self.array[ids], np.float32, copy=True)).to(device)


def prepare_pfull_bank(teacher, repo, backbone_checkpoint, store_dir, device, batch_size=128, smoke=False):
    store_dir = Path(store_dir); store_dir.mkdir(parents=True, exist_ok=True)
    meta = {'schema': 'reshift2-pfull-v1', 'rows': int(len(repo)), 'backbone_sha256': file_sha256(backbone_checkpoint),
            'history_file': str(Path(repo.root) / 'train_x_full_norm.npy'),
            'history_mtime_ns': (Path(repo.root) / 'train_x_full_norm.npy').stat().st_mtime_ns,
            'teacher': 'pretrained STLAformer, zero mask projection, frozen'}
    tag = meta['backbone_sha256'][:12] + f'_{len(repo)}' + ('_smoke' if smoke else '')
    path = store_dir / f'pfull_{tag}.npy'; mfile = path.with_suffix('.json')
    if path.is_file() and mfile.is_file() and json.loads(mfile.read_text()) == meta:
        arr = np.load(path, mmap_mode='r', allow_pickle=False)
        print(f'RESHIFT_PFULL_HIT path={path} shape={arr.shape}', flush=True)
        return ForecastBank(arr, device)
    print(f'RESHIFT_PFULL_BUILD rows={len(repo)} path={path}', flush=True)
    t0 = time.time(); chunks = []
    teacher.eval()
    with torch.no_grad():
        for s in range(0, len(repo), batch_size):
            e = min(len(repo), s + batch_size); x, m, tod, dow = repo.batch(s, e, device)
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                enabled=torch.device(device).type == 'cuda'):
                p = teacher(x, m, tod, dow)
            chunks.append(p.float().cpu().numpy().astype(np.float16))
    arr = np.concatenate(chunks, 0)
    tmp = path.with_name(path.name + '.partial.npy'); np.save(tmp, arr); os.replace(tmp, path)
    mfile.write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(f'RESHIFT_PFULL_PASS rows={len(arr)} seconds={time.time() - t0:.1f}', flush=True)
    return ForecastBank(np.load(path, mmap_mode='r', allow_pickle=False), device)
