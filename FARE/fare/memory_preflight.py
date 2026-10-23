"""Low-memory metadata/header inspection for existing schema-6 OOF Memory.

A pre-submission gate must never instantiate EpisodeMemory: constructing the
35k-episode store copies multi-GB tensors and computes all node capability keys.
The compute-node pipeline still uses EpisodeMemory.load(expected=...) for full
content materialization and provenance checks before training.
"""
from pathlib import Path
import json
import numpy as np

_TEMPS = {'x', 'mask', 'future', 'future_valid', 'residual', 'residual_valid',
          'failure_delta', 'failure_valid'}
_TIMES = {'starts', 'origin'}
_EXPECTED = _TEMPS | _TIMES
_BOOL = {'mask', 'future_valid', 'residual_valid', 'failure_valid'}


def inspect_cached_memory(memory_dir, *, nodes, history, horizon, expected=None):
    """Validate provenance, schema, array header shape/dtype without reading arrays."""
    p = Path(memory_dir)
    meta = json.loads((p/'manifest.json').read_text(encoding='utf8'))
    if meta.get('schema_version') != 6 or meta.get('format') != 'fare5_memmap':
        raise RuntimeError('Expected schema-6 memmapped OOF Memory')
    for key, value in (expected or {}).items():
        if meta.get(key) != value:
            raise RuntimeError(f'OOF Memory provenance mismatch for {key}: {meta.get(key)!r} != {value!r}')
    if meta.get('residual_provenance') != 'external_oos_paired':
        raise RuntimeError('Formal preflight requires paired out-of-fold residuals')
    m = int(meta.get('episodes', 0))
    if m < 1 or int(meta.get('nodes', -1)) != int(nodes):
        raise RuntimeError(f'Memory count/node mismatch: episodes={m}, nodes={meta.get("nodes")}, expected nodes={nodes}')
    if not _EXPECTED.issubset(set(meta.get('fields', []))):
        raise RuntimeError(f'Memory schema missing fields: {sorted(_EXPECTED-set(meta.get("fields", [])))}')
    for name in sorted(_EXPECTED):
        file = p/f'{name}.npy'
        # mmap_mode only loads the .npy header; no full 35k x 388 x 12 read.
        array = np.load(file, mmap_mode='r', allow_pickle=False)
        if name in {'x', 'mask'}:
            wanted = (m, int(history), int(nodes))
        elif name in _TIMES:
            wanted = (m,)
        else:
            wanted = (m, int(horizon), int(nodes))
        if array.shape != wanted:
            raise RuntimeError(f'Memory {name} shape {array.shape} != {wanted}')
        if name in _BOOL and array.dtype != np.dtype('bool'):
            raise RuntimeError(f'Memory {name} expected bool, got {array.dtype}')
        if name not in _BOOL and name not in _TIMES and not np.issubdtype(array.dtype, np.floating):
            raise RuntimeError(f'Memory {name} expected floating, got {array.dtype}')
        if name in _TIMES and not np.issubdtype(array.dtype, np.integer):
            raise RuntimeError(f'Memory {name} expected integer, got {array.dtype}')
        del array
    for name in meta.get('episode_info_fields', []):
        array = np.load(p/f'info_{name}.npy', mmap_mode='r', allow_pickle=False)
        if array.ndim != 1 or len(array) != m:
            raise RuntimeError(f'Invalid OOF episode-info field {name} shape {array.shape}')
        del array
    return meta
