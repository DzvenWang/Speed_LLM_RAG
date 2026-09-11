from __future__ import annotations

import json, os, random, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from baseline.evaluation import evaluate_prediction_file
from baseline.run_identity import build_run_manifest, write_run_manifest
from baseline.utils import load_yaml, write_json

ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')


def config() -> dict:
    return load_yaml(ROOT / 'configs' / 'baseline.yaml')


def require_gpu() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit('ERROR: method runner requires a GPU. Submit the corresponding PBS script.')
    return torch.device('cuda:0')


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def validate_stage(stage: str, method: str) -> tuple[dict, Path]:
    cfg = config()
    if stage not in ('development', 'formal'):
        raise SystemExit(f'{method}: only development/formal are allowed before evaluation authorization')
    subprocess.check_call([cfg['project']['python'], str(ROOT / 'scripts' / 'validate_cache.py'), '--stage', stage])
    subprocess.check_call([cfg['project']['python'], str(ROOT / 'scripts' / 'source_lock.py'), 'verify', '--root', str(ROOT), '--sources', str(ROOT / 'configs' / 'official_sources.yaml')])
    cache = Path(cfg['project']['exp_root']) / '_cache' / cfg['experiment']['subgraph'] / stage
    return cfg, cache


def make_run_dir(cfg: dict, method: str, stage: str, seed: int) -> Path:
    stamp = time.strftime('%Y%m%d_%H%M%S')
    p = Path(cfg['project']['exp_root']) / method / cfg['experiment']['subgraph'] / stage / f'{stamp}_seed{seed}'
    p.mkdir(parents=True, exist_ok=False)
    return p


def load_cache_manifest(cache: Path) -> dict:
    return json.loads((cache / 'cache_manifest.json').read_text())


def scale_stats(cache: Path) -> tuple[float, float]:
    j = load_cache_manifest(cache)
    split = 'train' if 'train' in j['splits'] else next(iter(j['splits']))
    info = j['splits'][split]
    return float(info['mean']), max(float(info['std']), 1e-6)


def as_unicode_array(values) -> np.ndarray:
    """Return a pickle-free NumPy Unicode array for portable NPZ artifacts."""
    vals = [str(v) for v in values]
    width = max(1, max((len(v) for v in vals), default=1))
    return np.asarray(vals, dtype=f'<U{width}')


def save_val_outputs(cfg: dict, cache: Path, run_dir: Path, pred_raw: np.ndarray, method: str, seed: int, stage: str, extra_summary: dict) -> dict:
    meta = pd.read_csv(cache / 'val_manifest.csv')
    pred_path = run_dir / 'val_predictions.npz'
    view_ids = as_unicode_array(meta.view_id.astype(str).tolist())
    np.savez_compressed(pred_path, view_id=view_ids, pred=np.asarray(pred_raw, np.float32))
    metrics = evaluate_prediction_file(cache, 'val', pred_path)
    metrics.to_csv(run_dir / 'val_metrics.csv', index=False)
    all_row = metrics.loc[metrics.family.astype(str) == 'ALL'].iloc[0].to_dict()
    manifest = build_run_manifest(cache, cfg['version'], method, seed, cfg['methods'][method]['formal_track'], {
        'artifact_type': 'method_result', 'runner_project_version': cfg['version'], 'status': 'COMPLETE'
    })
    write_run_manifest(run_dir / 'RUN_MANIFEST.json', manifest)
    summary = {
        **manifest,
        'val': all_row,
        'prediction_file': str(pred_path),
        **extra_summary,
    }
    write_json(run_dir / 'summary.json', summary)
    latest = run_dir.parent / 'LATEST_RUN.txt'; latest.write_text(str(run_dir) + '\n')
    return summary


def masked_l1(pred: torch.Tensor, true: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    m = valid.bool()
    if not torch.any(m):
        return (pred - true).abs().mean() * 0.0
    return (pred[m] - true[m]).abs().mean()


def masked_mse(pred: torch.Tensor, true: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    m = valid.bool()
    if not torch.any(m):
        return ((pred - true) ** 2).mean() * 0.0
    return ((pred[m] - true[m]) ** 2).mean()


class ArrayForecastDataset(Dataset):
    def __init__(self, cache: Path, split: str, query_signal: str = 'obs'):
        self.cache = Path(cache); self.split = split
        xname = 'x_obs_norm' if query_signal == 'obs' else 'x_full_norm'
        self.x = np.load(self.cache / f'{split}_{xname}.npy', mmap_mode='r')
        self.y = np.load(self.cache / f'{split}_y_norm.npy', mmap_mode='r')
        self.valid = np.load(self.cache / f'{split}_future_valid.npy', mmap_mode='r')
        self.tod = np.load(self.cache / f'{split}_tod_hist.npy', mmap_mode='r')
        self.dow = np.load(self.cache / f'{split}_dow_hist.npy', mmap_mode='r')
        self.meta = pd.read_csv(self.cache / f'{split}_manifest.csv')
        self.pred_len = int(self.y.shape[1]); self.label_len = 0
    def __len__(self): return int(self.x.shape[0])


class GinARDataset(ArrayForecastDataset):
    def __getitem__(self, i):
        x = np.asarray(self.x[i], np.float32)
        H,N = x.shape
        tod = np.asarray(self.tod[i], np.float32) / 287.0
        dow = np.asarray(self.dow[i], np.float32) / 6.0
        feat = np.empty((H,N,3), np.float32)
        feat[...,0] = x
        feat[...,1] = tod[:,None]
        feat[...,2] = dow[:,None]
        return torch.from_numpy(feat), torch.from_numpy(np.array(self.y[i],dtype=np.float32,copy=True)), torch.from_numpy(np.array(self.valid[i],dtype=np.uint8,copy=True)), int(i)


def _valid_datetime_candidate(raw: pd.Series, max_idx: int) -> list[pd.Timestamp] | None:
    """Parse only genuine timestamp-like columns and reject numeric row indices.

    ``pd.to_datetime`` interprets integers as nanoseconds from the Unix epoch, so a
    plain 0,1,2,... index silently collapses a 5-minute series into nearly identical
    calendar marks.  SARAF's time-aware bonus then degenerates to a constant.
    """
    if len(raw) < max_idx:
        return None
    if pd.api.types.is_numeric_dtype(raw.dtype):
        return None
    text = raw.astype(str)
    # Require obvious date/time syntax rather than arbitrary identifiers.
    probe = text.head(min(len(text), 256))
    if not probe.str.contains(r'[-/:T ]', regex=True).mean() > 0.8:
        return None
    s = pd.to_datetime(text, errors='coerce')
    if s.notna().mean() <= 0.95:
        return None
    s = s.iloc[:max_idx].reset_index(drop=True)
    if len(s) >= 3:
        d = s.diff().dropna().dt.total_seconds().to_numpy()
        d = d[np.isfinite(d) & (d > 0)]
        if d.size == 0:
            return None
        med = float(np.median(d))
        # TraffiDent is 5-minute data. Allow mild source irregularity, but reject
        # nanosecond/index-like or day-level columns.
        if not (240.0 <= med <= 360.0):
            return None
    return list(s)


def _load_datetime_axis(cfg: dict, cache: Path, split: str) -> list[pd.Timestamp]:
    meta = pd.read_csv(cache / f'{split}_manifest.csv')
    H = int(cfg['experiment']['history']); P = int(cfg['experiment']['horizon']); max_idx = int(meta.window_start.max()) + H + P + 1
    sg = Path(cfg['project']['benchmark_root']) / cfg['experiment']['subgraph']
    candidates = [sg/'core'/'time_index.csv', sg/'time_index.csv']
    for p in candidates:
        if not p.exists():
            continue
        df = pd.read_csv(p)
        for c in df.columns:
            parsed = _valid_datetime_candidate(df[c], max_idx)
            if parsed is not None:
                return parsed
    # Deterministic fallback preserves exact 5-min TOD/DOW periodicity and is safer
    # than interpreting an integer row-id column as nanosecond timestamps.
    base = pd.Timestamp('2000-01-03 00:00:00')  # Monday
    return list(pd.date_range(base, periods=max_idx, freq='5min'))


def build_time_marks(cfg: dict, cache: Path, split: str) -> tuple[np.ndarray, np.ndarray, str]:
    meta = pd.read_csv(cache / f'{split}_manifest.csv')
    H = int(cfg['experiment']['history']); P = int(cfg['experiment']['horizon'])
    axis = _load_datetime_axis(cfg, cache, split)
    xm = np.zeros((len(meta),H,6),np.float32); ym = np.zeros((len(meta),P,6),np.float32)
    for i,st in enumerate(meta.window_start.astype(int).to_numpy()):
        hs=axis[st:st+H]; fs=axis[st+H:st+H+P]
        for out,seq in ((xm[i],hs),(ym[i],fs)):
            for j,t in enumerate(seq): out[j]=[t.year,t.month,t.day,t.dayofweek,t.hour,t.minute]
    source='benchmark_time_index_or_deterministic_5min_fallback'
    return xm,ym,source


class SARAFTensorDataset(Dataset):
    """Dataset contract expected by RAFT/SARAF RetrievalTool.

    Each item is (local_index, x, y, x_mark, y_mark). ``limit`` is used only
    for GPU contract smoke tests; production runs use the complete frozen split.
    """
    def __init__(self, cache: Path, split: str, x_marks: np.ndarray, y_marks: np.ndarray, query_signal='obs', limit: int | None = None):
        self.cache=Path(cache); self.split=split; xname='x_obs_norm' if query_signal=='obs' else 'x_full_norm'
        x=np.load(self.cache/f'{split}_{xname}.npy',mmap_mode='r'); y=np.load(self.cache/f'{split}_y_norm.npy',mmap_mode='r')
        valid=np.load(self.cache/f'{split}_future_valid.npy',mmap_mode='r'); meta=pd.read_csv(self.cache/f'{split}_manifest.csv')
        n=int(x.shape[0]) if limit is None else min(int(limit),int(x.shape[0]))
        self.x=x[:n]; self.y=y[:n]; self.valid=valid[:n]; self.xm=x_marks[:n]; self.ym=y_marks[:n]; self.meta=meta.iloc[:n].reset_index(drop=True)
        self.pred_len=int(self.y.shape[1]); self.label_len=0
    def __len__(self): return int(self.x.shape[0])
    def __getitem__(self,i):
        return int(i), np.array(self.x[i],dtype=np.float32,copy=True), np.array(self.y[i],dtype=np.float32,copy=True), np.array(self.xm[i],dtype=np.float32,copy=True), np.array(self.ym[i],dtype=np.float32,copy=True)


def simple_namespace(**kwargs): return SimpleNamespace(**kwargs)
