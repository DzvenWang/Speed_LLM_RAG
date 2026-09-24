from __future__ import annotations
import numpy as np
import torch


def make_inputs(x, tod_hist, dow_hist, device):
    x = x.to(device, non_blocking=True).float().unsqueeze(-1)
    B, T, N, _ = x.shape
    tod = tod_hist.to(device, non_blocking=True).long().unsqueeze(-1).expand(B, T, N)
    dow = dow_hist.to(device, non_blocking=True).long().unsqueeze(-1).expand(B, T, N)
    te = torch.stack([tod, dow], dim=-1)
    return x, te


def forward_norm(model, batch, device):
    x, te = make_inputs(batch['x_obs'], batch['tod_hist'], batch['dow_hist'], device)
    pred = model(x, te)
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    return pred


def masked_mae(pred, target, mask):
    m = mask.bool() & torch.isfinite(target) & torch.isfinite(pred)
    if not torch.any(m):
        return pred.sum() * 0.0
    return torch.abs(pred - target)[m].mean()


def per_sample_masked_mae(pred, target, mask):
    m = mask.bool() & torch.isfinite(target) & torch.isfinite(pred)
    ae = torch.abs(pred - target)
    out = []
    for b in range(pred.shape[0]):
        mb = m[b]
        out.append(ae[b][mb].mean() if torch.any(mb) else pred[b].sum() * 0.0)
    return torch.stack(out, dim=0)


def restore_original_nodes(model, rex):
    B, Tp, _, D = rex.shape
    out = rex.new_zeros((B, Tp, int(model.node_num), D))
    out[:, :, model.ori_parts_idx, :] = rex[:, :, model.reo_parts_idx, :]
    return out


def decode_prehead(model, prehead_4d):
    B, Tp, N, D = prehead_4d.shape
    pred = model.regression_conv(prehead_4d.transpose(2, 3).reshape(B, Tp * D, N, 1))
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    return pred


def forward_with_prehead(model, x, te):
    embedded = model.embedding(x, te)
    rex = embedded[:, :, model.reo_all_idx, :]
    for block in model.spa_encoder:
        rex = block(rex)
    prehead = restore_original_nodes(model, rex)
    return decode_prehead(model, prehead), prehead


@torch.no_grad()
def predict_raw(model, item, device, mean, std):
    batch = {k: item[k].unsqueeze(0) for k in ['x_obs','y_raw','y_norm','future_valid','direct_mask','support_mask','tod_hist','dow_hist']}
    model.eval()
    pn = forward_norm(model, batch, device)
    return (pn * float(std) + float(mean)).detach().cpu().numpy()[0].astype(np.float32)


@torch.no_grad()
def predict_raw_and_key(model, item, device, mean, std):
    x = item['x_obs'].unsqueeze(0)
    tod = item['tod_hist'].unsqueeze(0)
    dow = item['dow_hist'].unsqueeze(0)
    tx, te = make_inputs(x, tod, dow, device)
    model.eval()
    pn, pre = forward_with_prehead(model, tx, te)
    raw = (pn * float(std) + float(mean)).detach().cpu().numpy()[0].astype(np.float32)
    z = pre.mean(dim=(1, 2)).detach().cpu().numpy()[0].astype(np.float32)
    z = z / (np.linalg.norm(z) + 1e-8)
    return raw, z

@torch.no_grad()
def predict_raw_from_x_batch(model, item, x_batch_np, device, mean, std):
    """Predict a batch of alternative histories for one query without using its future label."""
    xb=torch.from_numpy(np.asarray(x_batch_np,np.float32))
    B=xb.shape[0]
    tod=item['tod_hist'].unsqueeze(0).expand(B,-1)
    dow=item['dow_hist'].unsqueeze(0).expand(B,-1)
    batch={'x_obs':xb,'tod_hist':tod,'dow_hist':dow}
    model.eval(); pn=forward_norm(model,batch,device)
    return (pn*float(std)+float(mean)).detach().cpu().numpy().astype(np.float32)
