from __future__ import annotations
import numpy as np
import torch
from .model_io import forward_norm, masked_mae
from .utils import parameter_l2_delta


def configure_trainable(model, mode):
    for p in model.parameters(): p.requires_grad = False
    selected = []
    if mode == "frozen":
        pass
    elif mode == "head":
        if not hasattr(model, "regression_conv"):
            raise RuntimeError("PatchSTG model missing regression_conv")
        for p in model.regression_conv.parameters(): p.requires_grad=True; selected.append(p)
    elif mode == "last_block_head":
        if not hasattr(model, "spa_encoder") or len(model.spa_encoder) == 0:
            raise RuntimeError("PatchSTG model missing spa_encoder")
        for p in model.spa_encoder[-1].parameters(): p.requires_grad=True; selected.append(p)
        for p in model.regression_conv.parameters(): p.requires_grad=True; selected.append(p)
    elif mode == "full":
        for p in model.parameters(): p.requires_grad=True; selected.append(p)
    else:
        raise ValueError(mode)
    return [p for p in model.parameters() if p.requires_grad]


class OnlineUpdater:
    def __init__(self, model, dataset, device, variant_cfg, online_cfg, seed=10):
        self.model=model; self.ds=dataset; self.device=device; self.cfg=variant_cfg; self.ocfg=online_cfg
        self.params=configure_trainable(model, variant_cfg["trainable"])
        self.rng=np.random.default_rng(int(seed))
        self.anchor={name:p.detach().clone() for name,p in model.named_parameters() if p.requires_grad}
        self.optimizer=None
        if self.params:
            self.optimizer=torch.optim.AdamW(self.params,lr=float(variant_cfg["learning_rate"]),weight_decay=float(variant_cfg["weight_decay"]))
        self.update_count=0; self.step_count=0

    def _sample_batch(self, replay):
        bs=min(int(self.ocfg["batch_size"]),len(replay))
        if bs<=0: return []
        if self.ocfg.get("replay_sampling","recent_uniform") == "recent_uniform":
            pool=np.asarray(replay[-int(self.ocfg["replay_size"]):],np.int64)
            replace=len(pool)<bs
            return self.rng.choice(pool,size=bs,replace=replace).tolist()
        raise ValueError(self.ocfg.get("replay_sampling"))

    def _anchor_penalty(self):
        lam=float(self.cfg.get("anchor_lambda",0.0))
        if lam<=0: return None
        pen=None
        for name,p in self.model.named_parameters():
            if p.requires_grad and name in self.anchor:
                z=torch.sum((p-self.anchor[name])**2)
                pen=z if pen is None else pen+z
        return None if pen is None else lam*pen

    def update(self, replay):
        if self.optimizer is None: return {"updated":False,"loss":float("nan"),"steps":0,"param_delta_l2":0.0}
        if len(replay)<int(self.ocfg["min_mature_cases"]): return {"updated":False,"loss":float("nan"),"steps":0,"param_delta_l2":parameter_l2_delta(self.model,self.anchor)}
        self.model.train(); losses=[]
        steps=int(self.ocfg["steps_per_update"])
        for _ in range(steps):
            ids=self._sample_batch(replay); batch=self.ds.batch(ids)
            self.optimizer.zero_grad(set_to_none=True)
            pred=forward_norm(self.model,batch,self.device)
            y=batch["y_norm"].to(self.device,non_blocking=True).float()
            valid=batch["future_valid"].to(self.device,non_blocking=True).bool()
            loss=masked_mae(pred,y,valid)
            anchor=self._anchor_penalty()
            total=loss if anchor is None else loss+anchor
            total.backward()
            torch.nn.utils.clip_grad_norm_(self.params,float(self.ocfg["grad_clip"]))
            self.optimizer.step(); losses.append(float(loss.item())); self.step_count+=1
        self.update_count+=1; self.model.eval()
        return {"updated":True,"loss":float(np.mean(losses)),"steps":steps,"param_delta_l2":parameter_l2_delta(self.model,self.anchor)}
