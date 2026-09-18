from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .backbone import CachedTorch


def _first_tensor(x: Any):
    if torch.is_tensor(x):
        return x
    if isinstance(x, (list, tuple)):
        for z in x:
            t = _first_tensor(z)
            if t is not None:
                return t
    if isinstance(x, dict):
        for z in x.values():
            t = _first_tensor(z)
            if t is not None:
                return t
    return None


def canonical_node_tensor(x: torch.Tensor, batch_size: int, num_nodes: int,
                          min_width: int = 16, max_width: int = 8192):
    """Convert an intermediate tensor to [B,N,D] only when N is explicit.

    v0.9.5.1 deliberately refuses heuristic node reconstruction. A hidden tensor is
    admissible only when one of its non-batch axes exactly equals the benchmark node
    count. This keeps the layer sweep auditable and prevents channel/time axes from
    silently being interpreted as nodes.
    """
    if not torch.is_tensor(x) or x.ndim < 3 or int(x.shape[0]) != int(batch_size):
        return None
    dims = [i for i in range(1, x.ndim) if int(x.shape[i]) == int(num_nodes)]
    if not dims:
        return None
    # Traffic models commonly expose [B,T,N,D], [B,C,N,T], [B,N,D].
    # Prefer the right-most explicit N axis; the complete raw shape is logged.
    ni = dims[-1]
    order = [0, ni] + [i for i in range(1, x.ndim) if i != ni]
    z = x.permute(order).contiguous().reshape(batch_size, num_nodes, -1)
    w = int(z.shape[-1])
    if w < int(min_width) or w > int(max_width):
        return None
    return z


def _name_bonus(name: str) -> float:
    s = name.lower()
    bonus = 0.0
    # Prefer representation-bearing / deep spatiotemporal modules.
    for k in ['encoder', 'block', 'layer', 'temporal', 'spatial', 'patch', 'attention',
              'transformer', 'graph', 'stg', 'backbone', 'mixer']:
        if k in s:
            bonus += 0.35
    # Penalise obvious input/output projections. They are still catalogued and may be
    # swept if the candidate cap permits, but they should not win by name alone.
    for k in ['input', 'stem', 'embed', 'embedding', 'head', 'output', 'predictor',
              'projection', 'proj', 'fc']:
        if k in s:
            bonus -= 0.35
    return bonus


@dataclass(frozen=True)
class HiddenLayerChoice:
    module_name: str
    raw_shape: list[int]
    raw_width: int
    execution_index: int
    call_ordinal: int
    score: float
    module_depth: int = 0
    is_leaf: bool = True

    def to_dict(self):
        return asdict(self)


def discover_hidden_layers(adapter, sample_batch, num_nodes: int, cfg: dict) -> list[HiddenLayerChoice]:
    """Catalogue node-aligned PatchSTG activations, including non-leaf blocks.

    v0.9.5 used leaf modules only and therefore selected ``input_st_fc``. v0.9.5.1
    registers *all* named modules so that parent encoder/block outputs are observable.
    Candidate selection happens after the full catalogue is built; no layer is used to
    inspect validation labels during discovery.
    """
    model = adapter.model
    records = []
    handles = []
    counter = {'i': 0}
    calls = {}
    min_width = int(cfg.get('min_raw_width', 16))
    max_width = int(cfg.get('max_raw_width', 8192))
    include = str(cfg.get('include_regex', '.*'))
    exclude = str(cfg.get('exclude_regex', r'(^$|output|head$|predictor)'))

    modules = dict(model.named_modules())

    def hook(name):
        def _h(_m, _inp, out):
            counter['i'] += 1
            calls[name] = calls.get(name, 0) + 1
            ordinal = calls[name]
            if not re.search(include, name):
                return
            if exclude and re.search(exclude, name, flags=re.I):
                return
            tt = _first_tensor(out)
            if tt is None or tt.ndim < 3:
                return
            z = canonical_node_tensor(tt, int(tt.shape[0]), num_nodes, min_width, max_width)
            if z is None:
                return
            records.append((name, list(tt.shape), int(z.shape[-1]), counter['i'], ordinal,
                            name.count('.') + 1, len(list(modules[name].children())) == 0))
        return _h

    # All modules, not just leaves. Parent blocks frequently carry the useful predictive
    # state even when their leaf linear/attention submodules temporarily lose the N axis.
    for name, mod in model.named_modules():
        if name:
            handles.append(mod.register_forward_hook(hook(name)))
    try:
        with torch.no_grad():
            adapter.eval_forward(sample_batch)
    finally:
        for h in handles:
            h.remove()

    if not records:
        raise RuntimeError(
            'No node-aligned PatchSTG hidden tensor discovered. '
            'Inspect the official PatchSTG forward path or adjust layer_sweep include/exclude regex.'
        )

    max_exec = max(r[3] for r in records)
    max_depth = max(r[5] for r in records)
    choices = []
    seen = set()
    for name, shape, width, idx, ordinal, depth, is_leaf in records:
        key = (name, ordinal, tuple(shape), width)
        if key in seen:
            continue
        seen.add(key)
        exec_frac = idx / max(max_exec, 1)
        depth_frac = depth / max(max_depth, 1)
        score = (2.4 * exec_frac + 0.55 * depth_frac +
                 np.log2(max(width, 2)) / 10.0 + _name_bonus(name) +
                 (0.10 if not is_leaf else 0.0))
        choices.append(HiddenLayerChoice(name, shape, width, idx, ordinal, float(score), depth, is_leaf))
    choices.sort(key=lambda c: (c.score, c.execution_index, c.module_depth, c.raw_width), reverse=True)
    return choices


def select_sweep_candidates(catalog: list[HiddenLayerChoice], cfg: dict) -> list[HiddenLayerChoice]:
    """Select a compact but depth-diverse layer sweep from the discovery catalogue."""
    if not catalog:
        return []
    max_candidates = max(1, int(cfg.get('max_candidates', 10)))
    # Start with strongest score-ranked candidates.
    selected = []
    keys = set()

    def add(c):
        k = (c.module_name, c.call_ordinal)
        if k not in keys and len(selected) < max_candidates:
            selected.append(c); keys.add(k)

    # Guarantee an explicit input projection reference when present; it is useful to show
    # whether deeper states beat the exact layer v0.9.5 previously selected.
    for c in catalog:
        if c.module_name.endswith('input_st_fc') or c.module_name == 'input_st_fc':
            add(c); break

    # Then favour score-ranked deep candidates.
    for c in catalog:
        if len(selected) >= max_candidates:
            break
        add(c)

    # If score ranking is clustered, replace low-score tail entries with execution-depth
    # quantile representatives. This makes the sweep genuinely cover the forward path.
    if len(catalog) > max_candidates and max_candidates >= 4:
        by_exec = sorted(catalog, key=lambda c: c.execution_index)
        qidx = np.linspace(0, len(by_exec)-1, min(5, max_candidates), dtype=int)
        for qi in qidx:
            c = by_exec[int(qi)]
            if (c.module_name, c.call_ordinal) in keys:
                continue
            # Replace the current weakest non-input candidate.
            repl = None
            for j in range(len(selected)-1, -1, -1):
                if not selected[j].module_name.endswith('input_st_fc'):
                    repl = j; break
            if repl is None:
                continue
            keys.discard((selected[repl].module_name, selected[repl].call_ordinal))
            selected[repl] = c; keys.add((c.module_name, c.call_ordinal))
    selected.sort(key=lambda c: c.execution_index)
    return selected


def discover_hidden_layer(adapter, sample_batch, num_nodes: int, cfg: dict) -> HiddenLayerChoice:
    """Backward-compatible v0.9.5 helper: return the highest-scoring catalogue entry."""
    return discover_hidden_layers(adapter, sample_batch, num_nodes, cfg)[0]


def _projection_matrix(raw_width: int, out_width: int, seed: int):
    if raw_width == out_width:
        return None
    rng = np.random.default_rng(int(seed) + 911 * int(raw_width) + int(out_width))
    r = rng.standard_normal((raw_width, out_width), dtype=np.float32)
    r /= np.sqrt(max(raw_width, 1))
    return r.astype(np.float32)


def _capture_module(adapter, module_name: str, call_ordinal: int, raw_width: int,
                    num_nodes: int, max_width: int = 8192):
    target = dict(adapter.model.named_modules()).get(module_name)
    if target is None:
        raise RuntimeError(f'Configured PatchSTG hidden module no longer exists: {module_name}')
    captured = {'x': None, 'call': 0}

    def _hook(_m, _inp, out):
        captured['call'] += 1
        if captured['call'] != int(call_ordinal):
            return
        tt = _first_tensor(out)
        if tt is None:
            raise RuntimeError(f'Hidden module {module_name} produced no tensor')
        z = canonical_node_tensor(tt, int(tt.shape[0]), num_nodes, 1, max(max_width, int(raw_width)))
        if z is None:
            raise RuntimeError(f'Hidden module {module_name} output lost explicit node axis: {tuple(tt.shape)}')
        if int(z.shape[-1]) != int(raw_width):
            raise RuntimeError(f'Hidden width changed for {module_name}: {z.shape[-1]} != {raw_width}')
        captured['x'] = z.detach()

    return target, captured, _hook


def infer_hidden_split(adapter, cache, split: str, module_name: str, call_ordinal: int, raw_width: int,
                       out_width: int, seed: int, batch_size: int, num_nodes: int):
    ids = np.arange(len(CachedTorch(cache, split)), dtype=np.int64)
    return infer_hidden_ids(adapter, cache, split, ids, module_name, call_ordinal, raw_width,
                            out_width, seed, batch_size, num_nodes)


def infer_hidden_ids(adapter, cache, split: str, ids: Iterable[int], module_name: str,
                     call_ordinal: int, raw_width: int, out_width: int, seed: int,
                     batch_size: int, num_nodes: int):
    """Infer one audited hidden layer for an explicit ordered subset of cache rows."""
    ids = np.asarray(list(ids), dtype=np.int64)
    target, captured, hook_fn = _capture_module(adapter, module_name, call_ordinal, raw_width, num_nodes)
    proj = _projection_matrix(int(raw_width), int(out_width), int(seed))
    handle = target.register_forward_hook(hook_fn)
    ds = CachedTorch(cache, split)
    dl = DataLoader(Subset(ds, ids.tolist()), batch_size=int(batch_size), shuffle=False,
                    num_workers=0, pin_memory=torch.cuda.is_available())
    arr = []
    try:
        for b in dl:
            captured['x'] = None; captured['call'] = 0
            with torch.no_grad():
                adapter.eval_forward(b)
            z = captured['x']
            if z is None:
                raise RuntimeError(f'Forward hook for {module_name} call {call_ordinal} did not fire')
            x = z.float().cpu().numpy().astype(np.float32)
            if proj is not None:
                x = x @ proj
            rms = np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + 1e-6)
            x = np.clip(x / rms, -8.0, 8.0).astype(np.float32)
            arr.append(x.astype(np.float16))
    finally:
        handle.remove()
    if not arr:
        return np.zeros((0, num_nodes, out_width), np.float16)
    return np.concatenate(arr, axis=0)
