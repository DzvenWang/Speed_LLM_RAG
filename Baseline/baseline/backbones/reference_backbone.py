from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Iterable

import torch

from baseline.shared_cache import cache_path
from baseline.path_pointers import portable_project_path, resolve_portable_pointer
from baseline.backbones.training_mode import current_mode, checkpoint_mode, reference_stage

SUPPORTED_BACKBONES = ('STAEformer','HimNet','BigST','STDN','PatchSTG','STLAformer')
_SHARED_KEYS = (
    'dataset','target_name','target_channel_index','protocol_name','protocol_version','severity',
    'paired_group_id','protocol_stage_hash','split_view_hashes','cache_fingerprint',
    'shared_contract_version','cache_schema_version','data_contract_revision','time_feature_revision',
    'metrics_revision','prediction_contract_revision','time_feature_fingerprint','time_feature_source',
    'time_feature_parser','contract_fingerprint',
)


def sha256_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def _extract_state_dict(obj: dict) -> dict:
    for k in ('state_dict','model','model_state_dict','net'):
        if isinstance(obj.get(k),dict):
            return obj[k]
    if obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    raise RuntimeError('Cannot identify model state_dict in Backbone checkpoint')


def _try_load(model: torch.nn.Module, state: dict, label: str) -> None:
    candidates=[state]
    for prefix in ('model.','module.','adapter.model.'):
        if any(str(k).startswith(prefix) for k in state):
            candidates.append({str(k)[len(prefix):] if str(k).startswith(prefix) else str(k):v for k,v in state.items()})
    last=None
    for st in candidates:
        try:
            model.load_state_dict(st,strict=True)
            return
        except Exception as e:
            last=e
    raise RuntimeError(f'{label} checkpoint is incompatible with current Backbone adapter: {last}')


def _formal_cache_identity(cfg: dict) -> dict:
    from baseline.run_identity import load_cache_identity
    p=cache_path(cfg,'formal')
    if not (p/'cache_manifest.json').exists():
        raise FileNotFoundError(f'Baseline formal cache required before loading Backbone: {p / "cache_manifest.json"}')
    return load_cache_identity(p)


def _validate_checkpoint_dict(cfg: dict, ck: dict, *, model: str, seed: int, path: Path, train_mode: str | None = None) -> None:
    if not isinstance(ck,dict):
        raise RuntimeError(f'Backbone checkpoint must be a dict: {path}')
    got_model=str(ck.get('model') or '').strip()
    if got_model != model:
        raise RuntimeError(f'Backbone checkpoint model={got_model!r} != requested {model!r}: {path}')
    got_seed=ck.get('seed')
    if got_seed is not None and int(got_seed)!=int(seed):
        raise RuntimeError(f'Backbone checkpoint seed={got_seed} != requested {seed}: {path}')
    expected_target=str(cfg['experiment']['target'])
    got_target=str(ck.get('target') or ck.get('run_identity',{}).get('target_name') or '').strip()
    if got_target and got_target != expected_target:
        raise RuntimeError(f'Backbone checkpoint target={got_target} != requested {expected_target}: {path}')
    requested_mode = current_mode(train_mode)
    actual_mode = checkpoint_mode(ck)
    if actual_mode != requested_mode:
        raise RuntimeError(
            f'Backbone checkpoint train_data_mode={actual_mode!r} != requested {requested_mode!r}: {path}'
        )
    for key in ('state_dict','model_config','mean','std'):
        if key not in ck:
            raise RuntimeError(f'Backbone checkpoint missing {key}: {path}')
    ident=ck.get('run_identity')
    if not isinstance(ident,dict):
        raise RuntimeError(f'Backbone checkpoint missing run_identity: {path}')
    if str(ident.get('project'))!='Backbone' or str(ident.get('consumer_stage'))!='formal':
        raise RuntimeError(f'Backbone checkpoint must come from Backbone formal stage: {path}')
    cm=_formal_cache_identity(cfg)
    # Compare the immutable shared-cache identity directly; schema/project labels differ
    # between Backbone and Baseline but these shared fields must match exactly.
    mapping={k:k for k in _SHARED_KEYS}
    for ik,ckey in mapping.items():
        if ckey not in cm:
            continue
        gv=ident.get(ik); ev=cm.get(ckey)
        if ik=='split_view_hashes':
            gv=json.dumps(gv or {},sort_keys=True); ev=json.dumps(ev or {},sort_keys=True)
        if str(gv)!=str(ev):
            raise RuntimeError(f'Backbone checkpoint shared identity mismatch {ik}: checkpoint={gv} baseline={ev}: {path}')


def _candidate_checkpoints(cfg: dict, model: str, seed: int, *, train_mode: str | None = None) -> Iterable[Path]:
    root=Path(cfg['project']['backbone_exp_root'])/cfg['experiment']['target']/model/cfg['experiment']['subgraph']/reference_stage(train_mode)
    seen=set()
    latest_ck=root/'LATEST_CHECKPOINT.txt'
    if latest_ck.exists():
        raw=latest_ck.read_text().strip()
        if raw:
            p=resolve_portable_pointer(raw,cfg['project']['root'],pointer_parent=root)
            seen.add(str(p)); yield p
    latest_run=root/'LATEST_RUN.txt'
    if latest_run.exists():
        raw=latest_run.read_text().strip()
        if raw:
            d=resolve_portable_pointer(raw,cfg['project']['root'],pointer_parent=root)
            p=d/'best.pt'
            if str(p) not in seen:
                seen.add(str(p)); yield p
    if root.exists():
        for d in sorted(root.glob(f'*_seed{seed}'),reverse=True):
            p=d/'best.pt'
            if str(p) not in seen:
                seen.add(str(p)); yield p


def resolve_reference_checkpoint(cfg: dict, model: str, *, seed: int = 10, checkpoint: str | Path | None = None, train_mode: str | None = None) -> tuple[Path,dict]:
    model=str(model)
    train_mode=current_mode(train_mode)
    if model not in SUPPORTED_BACKBONES:
        raise ValueError(f'Unsupported Backbone {model!r}; choose from {SUPPORTED_BACKBONES}')
    candidates=[]
    explicit=checkpoint or os.environ.get('BASELINE_BACKBONE_CHECKPOINT','').strip() or None
    if explicit:
        p=resolve_portable_pointer(explicit,cfg['project']['root'])
        candidates=[p]
    else:
        candidates=list(_candidate_checkpoints(cfg,model,int(seed),train_mode=train_mode))
    errors=[]
    for p in candidates:
        try:
            if not p.is_file(): raise FileNotFoundError(p)
            ck=torch.load(p,map_location='cpu',weights_only=False)
            _validate_checkpoint_dict(cfg,ck,model=model,seed=int(seed),path=p,train_mode=train_mode)
            return p.resolve(),ck
        except Exception as e:
            errors.append(f'{p}: {type(e).__name__}: {e}')
    detail='\n'.join(errors[-8:]) if errors else '(no candidate checkpoints found)'
    raise RuntimeError(
        f'No compatible Backbone checkpoint found for model={model} target={cfg["experiment"]["target"]} '
        f'seed={seed} train_mode={train_mode} (expected Backbone/{reference_stage(train_mode)}). '
        'Train the corresponding Backbone formal model first or pass --backbone-checkpoint.\n'+detail
    )


class ReferenceBackbone:
    """Read-only consumer of an explicitly selected trained Backbone model.

    There is no freeze/registration step. The experiment selects a model name (and
    optionally an exact checkpoint); the chosen checkpoint is recorded by the Baseline
    run and reused for evaluation.
    """
    def __init__(self, cfg: dict, device: torch.device, *, model: str, seed: int = 10, checkpoint: str | Path | None = None, train_mode: str | None = None):
        self.cfg=cfg; self.device=device; self.model_name=str(model); self.reference_seed=int(seed)
        self.train_mode=current_mode(train_mode)
        self.checkpoint,self.checkpoint_obj=resolve_reference_checkpoint(cfg,self.model_name,seed=self.reference_seed,checkpoint=checkpoint,train_mode=self.train_mode)
        self.checkpoint_sha256=sha256_file(self.checkpoint)
        self.checkpoint_reference=portable_project_path(self.checkpoint,cfg['project']['root'])
        ck=self.checkpoint_obj
        broot=Path(cfg['project']['backbone_root'])
        if not (broot/'backbone/adapters.py').exists():
            raise FileNotFoundError(f'Missing sibling Backbone project: {broot}')
        sys.path.insert(0,str(broot)); sys.path.insert(0,str(broot.parent/'Base_SupportBench'))
        from backbone.adapters import build_adapter
        from supportbench.dataset import SupportBench
        bench=SupportBench(cfg['experiment']['subgraph'],root=cfg['project']['benchmark_root'])
        if int(ck.get('num_nodes',bench.N))!=int(bench.N):
            raise RuntimeError(f'Backbone checkpoint num_nodes={ck.get("num_nodes")} != benchmark N={bench.N}')
        self.mean=float(ck['mean']); self.std=float(ck['std'])
        adapter=build_adapter(
            self.model_name,broot/'vendor',bench.N,device,self.mean,self.std,ck['model_config'],bench.adj,bench.nodes
        )
        _try_load(adapter.model,_extract_state_dict(ck),self.model_name)
        adapter.model.to(device).eval()
        for p in adapter.model.parameters(): p.requires_grad=False
        self.adapter=adapter
        self.registry={
            'selected_model':self.model_name,
            'reference_backbone_train_mode':self.train_mode,
            'reference_seed':self.reference_seed,
            'reference_checkpoint':self.checkpoint_reference,
            'reference_checkpoint_sha256':self.checkpoint_sha256,
            'selection_mode':'explicit_model_direct_checkpoint_resolution',
        }

    @torch.no_grad()
    def predict_norm(self,x_norm:torch.Tensor,tod_hist:torch.Tensor,dow_hist:torch.Tensor)->torch.Tensor:
        # Some official Backbone adapters (HimNet/STDN) consume known future calendar
        # features, and BigST uses y_norm only to canonicalize output shape. These are
        # deterministic metadata, not future traffic values, so derive them from the
        # final historical 5-minute slot and provide a zero shape placeholder for y_norm.
        tod_hist=tod_hist.to(self.device).long(); dow_hist=dow_hist.to(self.device).long()
        x_norm=x_norm.to(self.device).float()
        B,T,N=x_norm.shape; P=int(self.cfg['experiment']['horizon'])
        last_abs=dow_hist[:,-1]*288 + tod_hist[:,-1]
        offsets=torch.arange(1,P+1,device=self.device,dtype=torch.long).unsqueeze(0)
        future_abs=last_abs.unsqueeze(1)+offsets
        tod_future=future_abs.remainder(288)
        dow_future=torch.div(future_abs,288,rounding_mode='floor').remainder(7)
        batch={
            'x':x_norm,'tod_hist':tod_hist,'dow_hist':dow_hist,
            'tod_future':tod_future,'dow_future':dow_future,
            'y_norm':torch.zeros((B,P,N),device=self.device,dtype=x_norm.dtype),
        }
        p=self.adapter.eval_forward(batch)
        if p.ndim!=3:
            raise RuntimeError(f'{self.model_name} prediction shape {tuple(p.shape)} != [B,P,N]')
        return p

    @torch.no_grad()
    def predict_raw(self,x_norm:torch.Tensor,tod_hist:torch.Tensor,dow_hist:torch.Tensor)->torch.Tensor:
        return self.adapter.raw(self.predict_norm(x_norm,tod_hist,dow_hist))


def build_reference_backbone(cfg: dict, device: torch.device, *, model: str, seed: int = 10, checkpoint: str | Path | None = None, train_mode: str | None = None) -> ReferenceBackbone:
    return ReferenceBackbone(cfg,device,model=model,seed=seed,checkpoint=checkpoint,train_mode=train_mode)


class ReferenceBackboneRASTBridge(torch.nn.Module):
    """Expose any selected ReferenceBackbone through RAST's backbone contract."""
    def __init__(self, reference: ReferenceBackbone):
        super().__init__()
        object.__setattr__(self, 'reference', reference)
        # Register the actual model only for correct train/eval recursion; parameters remain frozen.
        self.reference_model = reference.adapter.model
        self.last_prediction = None
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference_model.eval()
        return self

    def forward(self, history_data, future_data=None, batch_seen=0, epoch=0, train=False):
        if history_data.ndim != 4 or history_data.shape[-1] < 3:
            raise RuntimeError(
                f'Reference Backbone RAST bridge expects [B,H,N,>=3], got {tuple(history_data.shape)}'
            )
        x = history_data[..., 0]
        tod = torch.round(history_data[..., 1].mean(dim=2) * 287.0).long().clamp(0, 287)
        dow = torch.round(history_data[..., 2].mean(dim=2) * 6.0).long().clamp(0, 6)
        with torch.no_grad():
            pred = self.reference.predict_norm(x, tod, dow)
        self.last_prediction = pred.detach()
        return pred.unsqueeze(-1)
