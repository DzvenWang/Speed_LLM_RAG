from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .utils import project_root

MODELS = ("STAEformer", "HimNet", "BigST", "STDN", "PatchSTG", "STLAformer")


def _ensure_backbone_importable() -> Path:
    root = project_root()
    path = root / "Backbone"
    if not path.exists():
        raise FileNotFoundError(f"Backbone project not found: {path}")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return path


def load_backbone_config(target: str) -> dict:
    _ensure_backbone_importable()
    os.environ["TRAFFIDENT_TARGET"] = str(target)
    utils = importlib.import_module("backbone.utils")
    return utils.load_yaml(project_root() / "Backbone" / "configs" / "backbone.yaml")


def cache_path(target: str, stage: str, severity: int | None = None) -> Path:
    _ensure_backbone_importable()
    os.environ["TRAFFIDENT_TARGET"] = str(target)
    cfg = load_backbone_config(target)
    sc = importlib.import_module("backbone.shared_cache")
    return Path(sc.cache_path(cfg, stage, severity))


def backbone_metric_summary(pred, y, valid, direct) -> dict:
    _ensure_backbone_importable()
    metrics = importlib.import_module("backbone.metrics")
    return metrics.backbone_metric_summary(pred, y, valid, direct)


@dataclass(frozen=True)
class BackboneArtifacts:
    model: str
    target: str
    formal_dir: Path
    checkpoint: Path
    val_predictions: Path


def _formal_candidates(target: str, model: str, dataset: str, seed: int) -> list[Path]:
    base = project_root() / "Backbone_exps" / target / model / dataset / "formal"
    if not base.exists():
        return []
    out = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        if f"seed{seed}" not in p.name:
            continue
        if (p / "best.pt").exists() and (p / "val_predictions.npz").exists():
            out.append(p)
    return sorted(out, key=lambda p: (p.name, p.stat().st_mtime))


def discover_formal_artifacts(target: str, model: str, dataset: str, seed: int) -> BackboneArtifacts:
    cands = _formal_candidates(target, model, dataset, seed)
    if not cands:
        base = project_root() / "Backbone_exps" / target / model / dataset / "formal"
        raise FileNotFoundError(
            f"No completed Formal run with best.pt + val_predictions.npz for {target}/{model}. Searched {base}"
        )
    run = cands[-1]
    return BackboneArtifacts(
        model=model,
        target=target,
        formal_dir=run,
        checkpoint=run / "best.pt",
        val_predictions=run / "val_predictions.npz",
    )


def discover_test_predictions(art: BackboneArtifacts, dataset: str, severity: int) -> Path:
    # Evaluation tree is keyed by the selected Formal run directory name.
    base = (
        project_root() / "Backbone_exps" / art.target / art.model / dataset /
        "checkpoint_evaluation" / art.formal_dir.name / f"s{int(severity)}"
    )
    cands = sorted(base.glob("*/test_predictions.npz"), key=lambda p: (p.parent.name, p.stat().st_mtime))
    if not cands:
        raise FileNotFoundError(
            f"Missing test_predictions.npz for {art.target}/{art.model}/s{severity}. Expected under {base}. "
            f"Run Backbone checkpoint evaluation for {art.checkpoint}."
        )
    return cands[-1]


def _choose_prediction_key(z: np.lib.npyio.NpzFile) -> str:
    preferred = ("prediction", "predictions", "pred", "y_pred", "forecast")
    for key in preferred:
        if key in z.files:
            return key
    numeric = []
    for key in z.files:
        arr = z[key]
        if np.issubdtype(arr.dtype, np.number) and arr.ndim >= 3:
            numeric.append(key)
    if len(numeric) == 1:
        return numeric[0]
    raise RuntimeError(f"Cannot identify prediction array in NPZ; keys={z.files}")


def _choose_id_key(z: np.lib.npyio.NpzFile) -> str | None:
    for key in ("view_id", "view_ids", "ids", "id"):
        if key in z.files:
            return key
    for key in z.files:
        arr = z[key]
        if arr.ndim == 1 and arr.dtype.kind in ("U", "S", "O"):
            return key
    return None


def load_prediction_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    p = Path(path)
    with np.load(p, allow_pickle=False) as z:
        pkey = _choose_prediction_key(z)
        pred = np.asarray(z[pkey], dtype=np.float32)
        ikey = _choose_id_key(z)
        ids = None if ikey is None else np.asarray(z[ikey]).astype(str)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if pred.ndim != 3:
        raise RuntimeError(f"Prediction array must be [Q,P,N], got {pred.shape} from {p}")
    return pred, ids


def select_prediction_rows(
    pred: np.ndarray,
    ids: np.ndarray | None,
    expected_ids: Iterable[str],
    context: str,
    *,
    allow_prefix_without_ids: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Align a full prediction artifact to the requested cache rows.

    Formal validation/test prediction artifacts normally contain every cache row,
    while bounded smoke runs intentionally load only a prefix of the cache.  If
    view IDs are present we always align by ID (and therefore also tolerate a
    different row order).  If IDs are absent, prefix selection is allowed only
    when the caller explicitly states that the requested rows are a cache prefix.
    """
    exp = np.asarray(list(expected_ids)).astype(str)
    pred = np.asarray(pred)

    if ids is not None:
        got = np.asarray(ids).astype(str)
        if len(got) != pred.shape[0]:
            raise RuntimeError(
                f"{context}: prediction rows={pred.shape[0]} but id rows={len(got)}"
            )
        if len(np.unique(got)) != len(got):
            raise RuntimeError(f"{context}: prediction view_id array contains duplicates")
        pos = {v: i for i, v in enumerate(got.tolist())}
        missing = [v for v in exp.tolist() if v not in pos]
        if missing:
            raise RuntimeError(
                f"{context}: {len(missing)} expected view_ids are missing from prediction artifact; "
                f"examples={missing[:5]}"
            )
        take = np.fromiter((pos[v] for v in exp.tolist()), dtype=np.int64, count=len(exp))
        return pred[take], got[take]

    if pred.shape[0] == len(exp):
        return pred, None
    if allow_prefix_without_ids and pred.shape[0] >= len(exp):
        return pred[: len(exp)], None
    raise RuntimeError(
        f"{context}: prediction rows={pred.shape[0]} != expected rows={len(exp)} and no view_id "
        f"array is available for safe subset alignment"
    )


def assert_prediction_alignment(pred: np.ndarray, ids: np.ndarray | None, expected_ids: Iterable[str], context: str) -> None:
    exp = np.asarray(list(expected_ids)).astype(str)
    if pred.shape[0] != len(exp):
        raise RuntimeError(f"{context}: prediction rows={pred.shape[0]} != expected rows={len(exp)}")
    if ids is not None:
        got = np.asarray(ids).astype(str)
        if len(got) != len(exp) or not np.array_equal(got, exp):
            mismatch = np.flatnonzero(got != exp)[:5] if len(got) == len(exp) else []
            raise RuntimeError(f"{context}: view_id order mismatch; first mismatches={mismatch}")
