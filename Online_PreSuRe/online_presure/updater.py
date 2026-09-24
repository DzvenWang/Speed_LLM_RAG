from __future__ import annotations
import numpy as np
import torch
from .model_io import forward_norm, per_sample_masked_mae
from .utils import parameter_l2_delta


def configure_trainable(model, mode):
    for p in model.parameters():
        p.requires_grad = False
    if mode == 'frozen':
        return []
    if mode == 'head':
        if not hasattr(model, 'regression_conv'):
            raise RuntimeError('PatchSTG model missing regression_conv')
        for p in model.regression_conv.parameters():
            p.requires_grad = True
    elif mode == 'last_block_head':
        if not hasattr(model, 'spa_encoder') or len(model.spa_encoder) == 0:
            raise RuntimeError('PatchSTG model missing spa_encoder')
        for p in model.spa_encoder[-1].parameters():
            p.requires_grad = True
        for p in model.regression_conv.parameters():
            p.requires_grad = True
    elif mode == 'full':
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise ValueError(mode)
    return [p for p in model.parameters() if p.requires_grad]


class GuidedUpdater:
    """Persistent slow-path updater using random recent replay only."""
    def __init__(self, model, dataset, device, adapt_cfg, seed=10):
        self.model = model
        self.ds = dataset
        self.device = device
        self.cfg = adapt_cfg
        self.params = configure_trainable(model, adapt_cfg['trainable'])
        self.rng = np.random.default_rng(int(seed))
        self.anchor = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
        self.optimizer = None
        if self.params:
            self.optimizer = torch.optim.AdamW(
                self.params,
                lr=float(adapt_cfg['learning_rate']),
                weight_decay=float(adapt_cfg['weight_decay']),
            )
        self.update_count = 0
        self.step_count = 0

    def _anchor_penalty(self):
        lam = float(self.cfg.get('anchor_lambda', 0.0))
        if lam <= 0:
            return None
        pen = None
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.anchor:
                z = torch.sum((p - self.anchor[name]) ** 2)
                pen = z if pen is None else pen + z
        return None if pen is None else lam * pen

    def update(self, candidate_ids, candidate_scores=None, mode='random_replay'):
        if mode != 'random_replay':
            raise ValueError('Online PreSuRe v3 slow path supports random_replay only')
        if self.optimizer is None:
            return {'updated':False,'loss':float('nan'),'steps':0,'param_delta_l2':0.0,'batch_indices':[]}
        ids_all = np.asarray(candidate_ids, np.int64)
        if len(ids_all) < int(self.cfg['min_mature_cases_for_update']):
            return {'updated':False,'loss':float('nan'),'steps':0,'param_delta_l2':parameter_l2_delta(self.model,self.anchor),'batch_indices':[]}
        self.model.train()
        losses=[]; used=[]
        for _ in range(int(self.cfg['steps_per_update'])):
            bs=min(int(self.cfg['batch_size']),len(ids_all))
            sel=self.rng.choice(np.arange(len(ids_all)),size=bs,replace=False)
            ids=ids_all[sel].tolist()
            batch=self.ds.batch(ids)
            self.optimizer.zero_grad(set_to_none=True)
            pred=forward_norm(self.model,batch,self.device)
            y=batch['y_norm'].to(self.device,non_blocking=True).float()
            valid=batch['future_valid'].to(self.device,non_blocking=True).bool()
            loss=per_sample_masked_mae(pred,y,valid).mean()
            anchor=self._anchor_penalty(); total=loss if anchor is None else loss+anchor
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.params,float(self.cfg['grad_clip']))
            self.optimizer.step()
            losses.append(float(loss.item())); used.extend(ids); self.step_count+=1
        self.model.eval()
        if losses: self.update_count+=1
        return {
            'updated':bool(losses),'loss':float(np.mean(losses)) if losses else float('nan'),
            'steps':len(losses),'param_delta_l2':parameter_l2_delta(self.model,self.anchor),'batch_indices':used,
        }
