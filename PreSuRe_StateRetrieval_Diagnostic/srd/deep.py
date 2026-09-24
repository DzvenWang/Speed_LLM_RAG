from __future__ import annotations
import torch


def make_inputs(x, tod_hist, dow_hist, device):
    x = x.to(device, non_blocking=True).float().unsqueeze(-1)
    B, T, N, _ = x.shape
    tod = tod_hist.to(device, non_blocking=True).long().unsqueeze(-1).expand(B, T, N)
    dow = dow_hist.to(device, non_blocking=True).long().unsqueeze(-1).expand(B, T, N)
    te = torch.stack([tod, dow], dim=-1)
    return x, te


def restore_original_nodes(model, rex):
    B, Tp, _, D = rex.shape
    out = rex.new_zeros((B, Tp, int(model.node_num), D))
    out[:, :, model.ori_parts_idx, :] = rex[:, :, model.reo_parts_idx, :]
    return out


def decode_prehead(model, prehead_4d):
    B, Tp, N, D = prehead_4d.shape
    pred = model.regression_conv(prehead_4d.transpose(2, 3).reshape(B, Tp * D, N, 1))
    if pred.ndim == 4 and pred.shape[-1] == 1: pred = pred.squeeze(-1)
    return pred


def forward_with_prehead(model, x, te):
    embedded = model.embedding(x, te); rex = embedded[:, :, model.reo_all_idx, :]
    for block in model.spa_encoder: rex = block(rex)
    prehead = restore_original_nodes(model, rex)
    return decode_prehead(model, prehead), prehead
