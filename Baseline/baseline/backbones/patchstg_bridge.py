from __future__ import annotations
import json, subprocess, sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml

from baseline.shared_cache import cache_path

from .frozen_registry import (
    load_frozen_patchstg_registry,
    reference_checkpoint,
    require_reference_checkpoint_identity,
)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ('state_dict', 'model', 'model_state_dict', 'net'):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise RuntimeError('Cannot identify model state_dict in PatchSTG checkpoint')


def _try_load(model, state):
    candidates = [state]
    for prefix in ('model.', 'module.', 'adapter.model.'):
        if any(str(k).startswith(prefix) for k in state):
            candidates.append({
                str(k)[len(prefix):] if str(k).startswith(prefix) else str(k): v
                for k, v in state.items()
            })
    last = None
    for st in candidates:
        try:
            model.load_state_dict(st, strict=True)
            return
        except Exception as e:
            last = e
    raise RuntimeError(f'PatchSTG checkpoint is incompatible with frozen model: {last}')


class FrozenPatchSTG:
    """Read-only bridge to the single Backbone-owned reference PatchSTG checkpoint.

    Baseline random seeds never select different PatchSTG seeds. This keeps Baseline
    seed variance attributable to the Baseline method itself rather than the numerical
    forecasting backbone.
    """

    def __init__(self, cfg: dict, device: torch.device):
        self.cfg = cfg
        self.device = device
        baseline_cache = cache_path(cfg, 'formal') / 'cache_manifest.json'
        if not baseline_cache.exists():
            raise FileNotFoundError(f'Baseline formal cache required before loading frozen PatchSTG: {baseline_cache}')
        cm = json.loads(baseline_cache.read_text())
        contract = cfg.get('frozen_backbone_contract', {})
        reg = load_frozen_patchstg_registry(
            cfg['project']['frozen_backbone_json'],
            expected_protocol_hash=cm.get('protocol_stage_hash'),
            expected_time_feature_revision=cm.get('time_feature_revision'),
            expected_time_feature_fingerprint=cm.get('time_feature_fingerprint'),
            expected_time_feature_source=cm.get('time_feature_source'),
            expected_dataset=cfg['experiment']['subgraph'],
            expected_backbone_project_version=str(contract.get('required_project_version', '1.0.10')),
        )
        self.registry = reg
        self.reference_seed = int(reg['reference_seed'])
        self.checkpoint = reference_checkpoint(reg)

        broot = Path(cfg['project']['backbone_root'])
        if not (broot / 'backbone/adapters.py').exists():
            raise FileNotFoundError(f'Missing sibling Backbone project: {broot}')

        # Backbone v1.0.10 owns a single PatchSTG source lock and its source_lock.py
        # interface is: `source_lock.py {verify,create} --config <backbone.yaml>`.
        # Older Baseline code incorrectly called a Baseline-only `verify-one` API on
        # the Backbone script, causing every PatchSTG-dependent baseline to fail
        # before training.  Verify the current Backbone source through its native API.
        live_backbone_config = broot / 'configs/backbone.yaml'
        if not live_backbone_config.exists():
            raise FileNotFoundError(f'Missing Backbone config: {live_backbone_config}')
        subprocess.check_call([
            sys.executable, str(broot / 'scripts/source_lock.py'), 'verify',
            '--config', str(live_backbone_config),
        ])
        sys.path.insert(0, str(broot))
        sys.path.insert(0, str(broot.parent / 'TraffiDent_SupportBench'))
        from backbone.adapters import build_adapter
        from supportbench.dataset import SupportBench

        ck = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        require_reference_checkpoint_identity(ck, reg, label=str(self.checkpoint))
        bcfg = yaml.safe_load(live_backbone_config.read_text())
        live_backbone_version = str(bcfg.get('version'))
        frozen_backbone_version = str(reg.get('project_version'))

        # The Backbone project may evolve after the reference checkpoint was frozen.
        # Reproducibility is anchored by the frozen manifest/checkpoint identity above
        # (checkpoint SHA256, protocol hash, time-feature fingerprint, checkpoint
        # run_identity) rather than by requiring the *current* Backbone project version
        # string to remain identical to the historical producer version.
        #
        # If the project versions differ, never fall back to the current PatchSTG
        # configuration: construct the adapter from the model_config stored inside the
        # frozen checkpoint and rely on strict state_dict loading to catch an
        # incompatible current PatchSTG implementation.
        if 'model_config' in ck:
            mcfg = ck['model_config']
        elif live_backbone_version == frozen_backbone_version:
            mcfg = bcfg['models']['PatchSTG']
        else:
            raise RuntimeError(
                'Frozen PatchSTG checkpoint does not contain model_config while the '
                f'current Backbone version ({live_backbone_version}) differs from the '
                f'frozen producer version ({frozen_backbone_version}); refusing to use '
                'the current Backbone config as an implicit compatibility fallback.'
            )

        if live_backbone_version != frozen_backbone_version:
            print(
                'INFO Frozen PatchSTG producer/current Backbone versions differ '
                f'frozen={frozen_backbone_version} current={live_backbone_version}; '
                'accepting the frozen checkpoint by immutable identity and validating '
                'the current adapter with strict state_dict loading.'
            )
        sg = cfg['experiment']['subgraph']
        bench = SupportBench(sg, root=cfg['project']['benchmark_root'])
        st = bench.stats['channels'][bench.manifest['primary_channel']]
        self.mean = float(st['mean'])
        self.std = float(st['std'])
        adapter = build_adapter(
            'PatchSTG', broot / 'vendor', bench.N, device,
            float(st['mean']), float(st['std']), mcfg, bench.adj, bench.nodes,
        )
        _try_load(adapter.model, _extract_state_dict(ck))
        adapter.model.to(device).eval()
        for p in adapter.model.parameters():
            p.requires_grad = False
        self.adapter = adapter

    @torch.no_grad()
    def predict_norm(self, x_norm: torch.Tensor, tod_hist: torch.Tensor, dow_hist: torch.Tensor) -> torch.Tensor:
        batch = {'x': x_norm, 'tod_hist': tod_hist, 'dow_hist': dow_hist}
        p = self.adapter.eval_forward(batch)
        if p.ndim != 3:
            raise RuntimeError(f'PatchSTG prediction shape {tuple(p.shape)} != [B,P,N]')
        return p

    @torch.no_grad()
    def predict_raw(self, x_norm: torch.Tensor, tod_hist: torch.Tensor, dow_hist: torch.Tensor) -> torch.Tensor:
        return self.adapter.raw(self.predict_norm(x_norm, tod_hist, dow_hist))


class PatchSTGRASTBridge(nn.Module):
    def __init__(self, frozen: FrozenPatchSTG):
        super().__init__()
        self.frozen = frozen
        self.patchstg = frozen.adapter.model
        self.last_prediction = None
        for p in self.patchstg.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        self.patchstg.eval()
        return self

    def forward(self, history_data, future_data=None, batch_seen=0, epoch=0, train=False):
        # RAST input feature convention: [speed_norm, tod/287, dow/6]
        x = history_data[..., 0]
        tod = torch.round(history_data[..., 1].mean(dim=2) * 287.0).long().clamp(0, 287)
        dow = torch.round(history_data[..., 2].mean(dim=2) * 6.0).long().clamp(0, 6)
        with torch.no_grad():
            p = self.frozen.predict_norm(x, tod, dow)
        # Diagnostic-only cache used by the RAST-PatchSTG runner to measure
        # the learned residual correction.  It is detached and never enters
        # the optimization graph.
        self.last_prediction = p.detach()
        return p.unsqueeze(-1)
