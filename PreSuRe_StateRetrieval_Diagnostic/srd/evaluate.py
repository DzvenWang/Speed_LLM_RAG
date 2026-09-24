from __future__ import annotations
import numpy as np
import torch
from .deep import make_inputs, forward_with_prehead


def predict_raw(adapter, ck, x_np, tod_np, dow_np, device, return_prehead=False):
    x = torch.from_numpy(np.asarray(x_np, np.float32)).to(device)
    tod = torch.from_numpy(np.asarray(tod_np, np.int64)).to(device)
    dow = torch.from_numpy(np.asarray(dow_np, np.int64)).to(device)
    if x.ndim == 2: x = x.unsqueeze(0)
    if tod.ndim == 1: tod = tod.unsqueeze(0)
    if dow.ndim == 1: dow = dow.unsqueeze(0)
    tx, te = make_inputs(x, tod, dow, device)
    with torch.no_grad():
        pn, prehead = forward_with_prehead(adapter.model, tx, te)
    raw = pn * float(ck["std"]) + float(ck["mean"])
    if return_prehead: return raw.detach().cpu().numpy().astype(np.float32), prehead.detach().cpu().numpy().astype(np.float32)
    return raw.detach().cpu().numpy().astype(np.float32)


def pooled_prehead(prehead):
    z = np.asarray(prehead, np.float32)
    return z.mean(axis=(1, 2))


def predict_raw_batched(adapter, ck, x_np, tod_np, dow_np, device, batch_size=64):
    x = np.asarray(x_np, np.float32)
    tod = np.asarray(tod_np, np.int64)
    dow = np.asarray(dow_np, np.int64)
    if x.ndim == 2:
        return predict_raw(adapter, ck, x, tod, dow, device)
    outs = []
    bs = max(1, int(batch_size))
    for s in range(0, len(x), bs):
        outs.append(predict_raw(adapter, ck, x[s:s+bs], tod[s:s+bs], dow[s:s+bs], device))
    return np.concatenate(outs, axis=0) if outs else np.empty((0,), np.float32)


def predict_raw_prehead_batched(adapter, ck, x_np, tod_np, dow_np, device, batch_size=64):
    """Return denormalized predictions and PatchSTG prehead states in bounded batches."""
    x = np.asarray(x_np, np.float32)
    tod = np.asarray(tod_np, np.int64)
    dow = np.asarray(dow_np, np.int64)
    if x.ndim == 2:
        pred, pre = predict_raw(adapter, ck, x, tod, dow, device, return_prehead=True)
        return pred, pre
    preds, pres = [], []
    bs = max(1, int(batch_size))
    for s in range(0, len(x), bs):
        pred, pre = predict_raw(adapter, ck, x[s:s+bs], tod[s:s+bs], dow[s:s+bs], device, return_prehead=True)
        preds.append(pred); pres.append(pre)
    if not preds:
        return np.empty((0,), np.float32), np.empty((0,), np.float32)
    return np.concatenate(preds, axis=0), np.concatenate(pres, axis=0)
