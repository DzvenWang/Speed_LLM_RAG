#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import inspect
import json
import os
import sys
import tempfile
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import torch

GADI_ROOT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/Baseline')
ROOT = GADI_ROOT if GADI_ROOT.exists() else Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SUPPORTBENCH_PROJECT = Path('/scratch/mh20/zw1165/Speed_LLM_RAG/TraffiDent_SupportBench')
if not SUPPORTBENCH_PROJECT.exists():
    SUPPORTBENCH_PROJECT = ROOT.parent / 'TraffiDent_SupportBench'
if SUPPORTBENCH_PROJECT.exists() and str(SUPPORTBENCH_PROJECT) not in sys.path:
    sys.path.insert(0, str(SUPPORTBENCH_PROJECT))

from baseline.backbones.patchstg_bridge import FrozenPatchSTG
from baseline.runners.common import (
    ImputationDataset,
    make_run_dir,
    require_gpu,
    save_val_outputs,
    scale_stats,
    set_seed,
    validate_stage,
)
from baseline.sprint_causal_history import (
    SprintCanonicalHistory,
    context_to_sprint_csv,
    sprint_result_to_current_query,
    resolve_canonical_history_source,
)


def import_sprint():
    repo = ROOT / 'vendor' / 'SPRINT'
    for candidate in (repo / 'src', repo):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
    try:
        from SPRINT_imputations import impute
    except Exception as e:
        raise RuntimeError(
            'Cannot import official SPRINT package from vendor/SPRINT. '
            'Run scripts/setup_official_models.sh and install lightweight SPRINT dependencies. '
            f'Original error: {e!r}'
        ) from e
    return impute


def sprint_vendor_defaults(impute_fn) -> dict:
    """Read SPRINT algorithm defaults from the pinned vendor function itself.

    Baseline code must not silently replace the method's progressive ranks,
    iteration budget, prior strengths, or batching schedule with custom values.
    """
    sig = inspect.signature(impute_fn)
    names = (
        'ranks', 'max_iter', 'hyp_prior_list', 'hy1_list', 'Batch_per', 'Batch_iter',
        'miss_type', 'missing_rate', 'missing_rate_val', 'zero_as_missing',
    )
    out = {}
    for name in names:
        p = sig.parameters.get(name)
        if p is None or p.default is inspect._empty:
            raise RuntimeError(
                f'Pinned SPRINT impute() does not expose a default for {name}: {sig}'
            )
        value = p.default
        if isinstance(value, (list, tuple)):
            value = list(value)
        out[name] = value
    return out


def sprint_impute_query(
    cfg: dict,
    impute_fn,
    history_source: SprintCanonicalHistory,
    window_start: int,
    x_norm: np.ndarray,
    observed: np.ndarray,
    mean: float,
    std: float,
    *,
    smoke: bool = False,
) -> np.ndarray:
    """Causal historical-day SPRINT adaptation for one incomplete H12 query.

    Context is ``D previous days at the same H12 clock-aligned segment + current H12``.
    Every historical interval ends strictly before ``window_start``. The current row
    contains only benchmark-observed cells. No P12 target/future and no later-day data
    are available to SPRINT. Only the imputed current H12 is sent to frozen PatchSTG.
    """
    c = cfg['runners']['SPRINT']
    x_norm = np.asarray(x_norm, np.float32)
    observed = np.asarray(observed).astype(bool)
    if x_norm.ndim != 2 or observed.shape != x_norm.shape:
        raise ValueError(f'Expected x/observed [H,N], got {x_norm.shape}/{observed.shape}')
    H, N = x_norm.shape
    if (H, N) != (history_source.history, history_source.nodes):
        raise RuntimeError('SPRINT query geometry does not match canonical-history source')

    raw = x_norm * np.float32(std) + np.float32(mean)
    context, historical_starts = history_source.build_context(window_start, raw, observed)
    days = int(context.shape[0])
    native = sprint_vendor_defaults(impute_fn)
    native_ranks = [int(v) for v in native['ranks']]
    if native_ranks and days <= max(native_ranks):
        raise RuntimeError(
            'SPRINT causal day dimension must exceed the pinned vendor maximum rank: '
            f'context_days={days}, native_ranks={native_ranks}. Increase historical_days; '
            'do not shrink the SPRINT decomposition rank in the baseline adapter.'
        )
    # Only benchmark/data semantics are passed explicitly. Algorithmic defaults
    # (progressive ranks, NTD iteration budget, priors and batch schedule) remain
    # exactly those exposed by the pinned official vendor function. Smoke mode
    # changes only the number of benchmark queries, never SPRINT itself.
    kwargs = dict(
        device=str(c.get('device', 'CPU')),
        zero_as_missing=bool(c.get('zero_as_missing', False)),
        # Do not inject synthetic benchmark missingness. Keep SPRINT's pinned
        # vendor defaults for miss_type/internal validation and all algorithmic
        # settings; zero handling is the only parser-level override.
        missing_rate=0.0,
    )

    with tempfile.TemporaryDirectory(prefix='baseline_sprint_') as td:
        td_path = Path(td)
        csv_path = td_path / 'causal_history_query.csv'
        context_to_sprint_csv(csv_path, context)
        old = Path.cwd()
        captured = io.StringIO()
        try:
            os.chdir(td_path)
            with redirect_stdout(captured), redirect_stderr(captured):
                out = impute_fn(str(csv_path), (days, N, H), **kwargs)
        except Exception as e:
            tail = captured.getvalue()[-6000:]
            raise RuntimeError(
                'Official SPRINT failed on causal historical-day context. '
                f'tensor={(days,N,H)}, current_window_start={int(window_start)}, '
                f'historical_start_range=({historical_starts[0]},{historical_starts[-1]}), '
                f'vendor_defaults={native}. Captured tail:\n{tail}'
            ) from e
        finally:
            os.chdir(old)

    imp = sprint_result_to_current_query(out, days, H, N)
    # Evaluation contract: reconstruction may fill missing cells but cannot alter
    # any value that the current query actually observed.
    completed_raw = np.where(observed, raw, imp).astype(np.float32)
    if not np.allclose(completed_raw[observed], raw[observed], rtol=0.0, atol=1e-6):
        raise RuntimeError('SPRINT observed-value protection failed')
    return ((completed_raw - np.float32(mean)) / np.float32(std)).astype(np.float32)


def forecast_split(
    cfg: dict,
    cache: Path,
    split: str,
    patch: FrozenPatchSTG,
    device: torch.device,
    *,
    smoke: bool = False,
    impute_fn=None,
) -> np.ndarray:
    if impute_fn is None:
        impute_fn = import_sprint()
    ds = ImputationDataset(cache, split)
    mean, std = scale_stats(cache)
    source = resolve_canonical_history_source(cfg, cache, split, mean=mean, std=std)
    limit = min(len(ds), 1 if smoke else len(ds))
    completed, tods, dows = [], [], []
    outputs = []
    bs = int(cfg['runners']['SPRINT'].get('patchstg_batch_size', 32))

    def flush():
        if not completed:
            return
        x = torch.from_numpy(np.stack(completed)).float().to(device)
        tod = torch.from_numpy(np.stack(tods)).long().to(device)
        dow = torch.from_numpy(np.stack(dows)).long().to(device)
        with torch.no_grad():
            pred = patch.predict_raw(x, tod, dow)
        if not torch.isfinite(pred).all():
            raise RuntimeError('SPRINT -> frozen PatchSTG produced non-finite predictions')
        outputs.append(pred.cpu().numpy().astype(np.float32))
        completed.clear(); tods.clear(); dows.clear()

    for i in range(limit):
        _idx, x, _full, obs, _art, tod, dow = ds[i]
        st = int(ds.meta.iloc[i].window_start)
        completed.append(
            sprint_impute_query(cfg, impute_fn, source, st, x, obs, mean, std, smoke=smoke)
        )
        tods.append(np.asarray(tod, np.int64)); dows.append(np.asarray(dow, np.int64))
        if len(completed) >= bs:
            flush()
        if (i + 1) % 25 == 0:
            print(f'SPRINT causal-history progress {i+1}/{limit}', flush=True)
    flush()
    return np.concatenate(outputs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['development', 'formal'], default='development')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg, cache = validate_stage(a.stage, 'SPRINT')
    set_seed(a.seed)
    device = require_gpu()
    patch = FrozenPatchSTG(cfg, device)
    c = cfg['runners']['SPRINT']
    impute_fn = import_sprint()
    native = sprint_vendor_defaults(impute_fn)
    native_ranks = [int(v) for v in native['ranks']]
    if int(c['historical_days']) + 1 <= max(native_ranks):
        raise RuntimeError(
            f'SPRINT historical_days={c["historical_days"]} is incompatible with '
            f'pinned vendor ranks={native_ranks}; keep vendor ranks and enlarge causal context.'
        )
    print('SPRINT CAUSAL HISTORICAL-DAY CONTRACT', {
        'historical_days': int(c['historical_days']),
        'context_days': int(c['historical_days']) + 1,
        'day_steps': int(c['day_steps']),
        'within_day_intervals': int(cfg['experiment']['history']),
        'timeline_causal_past_observations_allowed': True,
        'current_hidden_values_allowed': False,
        'current_query_future_allowed': False,
        'other_query_predictions_or_labels_allowed': False,
        'reporting_track': cfg['methods']['SPRINT']['table_group'],
        'algorithm_policy': 'pinned_vendor_native_defaults',
        'vendor_defaults': native,
    }, flush=True)
    t0 = time.time()
    raw = forecast_split(
        cfg, cache, 'val', patch, device, smoke=a.smoke, impute_fn=impute_fn
    )
    if a.smoke:
        exp = (1, int(cfg['experiment']['horizon']), np.load(cache / 'val_x_obs_norm.npy', mmap_mode='r').shape[2])
        if tuple(raw.shape) != exp:
            raise RuntimeError(f'SPRINT smoke prediction shape {raw.shape} != {exp}')
        print('SPRINT CAUSAL HISTORY -> FROZEN PATCHSTG GPU SMOKE PASS', {
            'shape': raw.shape,
            'patchstg_reference_seed': patch.reference_seed,
        }, flush=True)
        return

    rd = make_run_dir(cfg, 'SPRINT', a.stage, a.seed)
    context_label = f"{int(c['historical_days'])}past_days_plus_current_xH{int(cfg['experiment']['history'])}"
    ck = {
        'method': 'SPRINT', 'seed': int(a.seed), 'best_epoch': 0,
        'official_imputer': 'vendor/SPRINT::SPRINT_imputations.impute',
        'adaptation': f'causal_historical_day_context_{context_label}',
        'reporting_track': cfg['methods']['SPRINT']['table_group'],
        'historical_days': int(c['historical_days']),
        'day_steps': int(c['day_steps']),
        'vendor_algorithm_defaults': native,
        'patchstg_reference_seed': patch.reference_seed,
        'patchstg_checkpoint': str(patch.checkpoint),
    }
    torch.save(ck, rd / 'best.pt')
    sm = save_val_outputs(cfg, cache, rd, raw, 'SPRINT', a.seed, a.stage, {
        'best_epoch': 0, 'params': 0, 'wall_seconds': time.time() - t0,
        'official_source': 'vendor/SPRINT',
        'adaptation': (
            f'causal historical-day SPRINT: previous {int(c["historical_days"])} days at the same '
            f'H{int(cfg["experiment"]["history"])} clock-aligned segment plus current incomplete query; '
            'canonical raw speed/raw-valid history only; current observed values protected; no query future'
        ),
        'reporting_track': cfg['methods']['SPRINT']['table_group'],
        'input_matching_note': 'specialized comparator with additional causal historical-day context; not strict H12 input-matched main-table baseline',
        'sprint_historical_days': int(c['historical_days']),
        'sprint_day_steps': int(c['day_steps']),
        'downstream_forecaster': 'Frozen PatchSTG v1.0.10 reference checkpoint',
        'patchstg_reference_seed': patch.reference_seed,
        'patchstg_checkpoint': str(patch.checkpoint),
        'checkpoint_selection': 'no learned benchmark checkpoint; SPRINT internal reconstruction validation; downstream PatchSTG frozen',
        'vendor_algorithm_defaults': native,
        'algorithm_policy': 'pinned_vendor_native_defaults',
        'test_access': False,
    })
    print('SPRINT COMPLETE', json.dumps({'run_dir': str(rd), 'val': sm['val']}, default=str))


if __name__ == '__main__':
    main()
