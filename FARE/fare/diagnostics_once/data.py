"""Strict, leakage-aware contract for FARE one-pass validation exports.

The export adapter must create one compressed NPZ per target x severity x family.
All modes within a file must originate from the same frozen checkpoint and
identical query/target tensors. This reader never invokes training.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np

REQUIRED_MODES = ('full', 'no_evidence')
RECOMMENDED_MODES = ('hidden_only', 'forecast_only', 'null_evidence',
                     'no_gate', 'similarity_only', 'random', 'shuffled',
                     'wrong_evidence', 'no_mask_replay', 'cross_history')

class InvalidDiagnosticInput(ValueError):
    pass

def metadata_read(npz,allowed_splits=('val',)):
    if 'metadata_json' not in npz:
        raise InvalidDiagnosticInput('metadata_json is required for provenance, including split, checkpoint, and model version')
    meta = npz['metadata_json']
    if meta.size != 1:
        raise InvalidDiagnosticInput('metadata_json must be a scalar')
    try:
        obj=json.loads(str(meta.item()))
    except Exception as exc:
        raise InvalidDiagnosticInput('metadata_json is not valid JSON') from exc
    for key in ('target', 'severity', 'failure_family', 'split', 'checkpoint', 'checkpoint_sha256', 'seed', 'metrics_space'):
        if key not in obj:
            raise InvalidDiagnosticInput(f'metadata missing {key}')
    if obj['target'] not in ('speed','flow') or int(obj['severity']) not in (20,40,60,90):
        raise InvalidDiagnosticInput('target/severity invalid')
    if not obj.get('checkpoint_sha256') or len(str(obj['checkpoint_sha256']))!=64:
        raise InvalidDiagnosticInput('checkpoint_sha256 should be a 64-character SHA256 hex digest')
    if obj['split'] not in allowed_splits:
        raise InvalidDiagnosticInput(f'Diagnostic split {obj["split"]} not allowed; default split=val, allowed={allowed_splits}')
    return obj

@dataclass
class DiagnosticShard:
    path: Path
    metadata: dict
    truth: np.ndarray
    mask: np.ndarray
    modes: dict[str, np.ndarray]
    candidate_predictions: np.ndarray | None
    extras: dict[str, np.ndarray]
    query_id: np.ndarray

    @property
    def n(self): return self.truth.shape[0]


def load_shard(path:Path, max_queries=0, seed=10, allowed_splits=('val',)):
    path=Path(path)
    with np.load(path, allow_pickle=False) as npz:
        meta=metadata_read(npz,allowed_splits=allowed_splits)
        for k in ('y_true','valid_mask','query_id'):
            if k not in npz: raise InvalidDiagnosticInput(f'{path}: missing {k}')
        truth=np.asarray(npz['y_true'],dtype=np.float64)
        if truth.ndim not in (3,4):
            raise InvalidDiagnosticInput(f'{path}: expected y_true [Q,H,N] or [Q,H,N,C]')
        mask=np.asarray(npz['valid_mask'],dtype=bool)
        if mask.shape != truth.shape:
            try:mask=np.broadcast_to(mask,truth.shape)
            except ValueError as e:raise InvalidDiagnosticInput(f'{path}: mask not broadcastable to truth shape') from e
        qid=np.asarray(npz['query_id'])
        if qid.ndim != 1 or len(qid)!=len(truth):
            raise InvalidDiagnosticInput('query_id must have Q entries')
        if len(set(map(str,qid.tolist()))) != len(qid):
            raise InvalidDiagnosticInput(f'{path}: duplicate query_id in shard')
        modes={k[6:]:np.asarray(npz[k],dtype=np.float64) for k in npz.files if k.startswith('pred__')}
        for m in REQUIRED_MODES:
            if m not in modes:raise InvalidDiagnosticInput(f'{path}: missing pred__{m}')
        for m,a in modes.items():
            if a.shape!=truth.shape:raise InvalidDiagnosticInput(f'{path}: pred__{m} shape {a.shape} != truth {truth.shape}')
        candidate = np.asarray(npz['pred_candidates'],dtype=np.float64) if 'pred_candidates' in npz else None
        if candidate is not None and (candidate.ndim!=truth.ndim+1 or candidate.shape[:1]+candidate.shape[2:]!=truth.shape):
            raise InvalidDiagnosticInput(f'{path}: pred_candidates must be [Q,K,...]')
        extra_names=['mask_ratio','demand','alpha','null_weight','propagation', 'candidate_score','traffic_regime','future_start','memory_last_future_end']
        extras={k:np.asarray(npz[k]) for k in extra_names if k in npz}
        if 'future_start' in extras and 'memory_last_future_end' in extras:
            memory_end=np.asarray(extras['memory_last_future_end'])
            query_start=np.asarray(extras['future_start'])
            while query_start.ndim < memory_end.ndim:query_start=query_start[...,None]
            if np.any(memory_end>=query_start):
                raise InvalidDiagnosticInput('future leakage: at least one candidate future overlaps/query begins before memory known')
    if not np.any(mask):raise InvalidDiagnosticInput(f'{path}: valid_mask is empty')
    # Deterministic equal-probability sampling within shard, without peeking at labels.
    if max_queries and len(qid)>max_queries:
        hashes=np.array([int(hashlib.sha256((str(x)+'|'+str(seed)).encode()).hexdigest()[:16],16) for x in qid],dtype=np.uint64)
        inds=np.sort(np.argsort(hashes,kind='stable')[:max_queries])
        truth=truth[inds];mask=mask[inds];qid=qid[inds]
        modes={k:v[inds] for k,v in modes.items()}
        if candidate is not None: candidate=candidate[inds]
        extras={k:v[inds] if v.ndim>0 and v.shape[0]==len(hashes) else v for k,v in extras.items()}
    return DiagnosticShard(path,meta,truth,mask,modes,candidate,extras,qid)


def sha256_file(path:Path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
