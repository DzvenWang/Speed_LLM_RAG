from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path: str|Path, model, optimizer, epoch: int, score: float, meta: dict):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"state_dict":model.state_dict(),"optimizer":None if optimizer is None else optimizer.state_dict(),"epoch":int(epoch),"score":float(score),"meta":dict(meta)},p)


def load_model_checkpoint(path: str|Path, model, device, expected_meta: dict|None=None):
    ck=torch.load(Path(path),map_location=device,weights_only=False)
    if expected_meta:
        for k,v in expected_meta.items():
            if ck.get("meta",{}).get(k)!=v: raise RuntimeError(f"Checkpoint meta mismatch {k}: {ck.get('meta',{}).get(k)} != {v}")
    model.load_state_dict(ck["state_dict"],strict=True); return ck
