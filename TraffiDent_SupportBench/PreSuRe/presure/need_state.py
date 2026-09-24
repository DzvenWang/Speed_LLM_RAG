from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .models import NodeSensingEncoder
from .probes import binary_auroc


class PredictiveNeedStateNet(nn.Module):
    """Task-specific predictor for coarse predictive-need states.

    The model is intentionally instantiated and trained separately for each task
    (need detection, direction, magnitude band, signed need state) so that the
    v0.7.2 diagnostic is not confounded by multi-task gradient conflict.

    Inputs are deployment-safe node features only.  The feature tensor already
    contains incomplete history, history observation masks, neighbour context,
    time covariates, node identity and the frozen PatchSTG forecast trajectory.
    No current-query future value or evaluation mask is an input.
    """

    def __init__(
        self,
        din: int,
        N: int,
        P: int,
        base_start: int,
        num_classes: int,
        hidden: int = 192,
        demb: int = 96,
        node_dim: int = 24,
        horizon_dim: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.P = int(P)
        self.base_start = int(base_start)
        self.num_classes = int(num_classes)
        self.encoder = NodeSensingEncoder(
            int(din), int(N), int(hidden), int(demb), int(node_dim), float(dropout)
        )
        self.h_emb = nn.Embedding(self.P, int(horizon_dim))
        self.head = nn.Sequential(
            nn.Linear(int(demb) + int(horizon_dim) + 1, int(hidden)),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 96),
            nn.GELU(),
            nn.Linear(96, self.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [B,N,F] -> logits [B,P,N,C]
        B, N, _ = x.shape
        z = self.encoder(x)
        h = self.h_emb.weight[None, None, :, :].expand(B, N, -1, -1)
        ze = z[:, :, None, :].expand(-1, -1, self.P, -1)
        base = x[:, :, self.base_start : self.base_start + self.P]
        if base.shape[-1] != self.P:
            raise RuntimeError(
                f"PredictiveNeedStateNet cannot recover PatchSTG base slice from "
                f"{tuple(x.shape)} start={self.base_start} P={self.P}"
            )
        o = self.head(torch.cat([ze, h, base[:, :, :, None]], dim=-1))
        return o.permute(0, 2, 1, 3).contiguous()


def class_weights_from_counts(counts, eps: float = 1.0):
    counts = np.asarray(counts, np.float64)
    if counts.ndim != 1 or len(counts) < 2:
        raise ValueError("counts must be a 1-D vector with >=2 classes")
    counts = np.maximum(counts, float(eps))
    w = counts.sum() / (len(counts) * counts)
    w = w / np.mean(w)
    return w.astype(np.float32)


def confusion_matrix_np(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true, np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, np.int64).reshape(-1)
    m = (y_true >= 0) & (y_true < int(num_classes)) & (y_pred >= 0) & (y_pred < int(num_classes))
    out = np.zeros((int(num_classes), int(num_classes)), np.int64)
    if np.any(m):
        np.add.at(out, (y_true[m], y_pred[m]), 1)
    return out


def macro_f1_from_cm(cm):
    cm = np.asarray(cm, np.float64)
    vals = []
    for k in range(cm.shape[0]):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        den = 2.0 * tp + fp + fn
        vals.append(0.0 if den <= 0 else (2.0 * tp / den))
    return float(np.mean(vals)) if vals else 0.0


def balanced_accuracy_from_cm(cm):
    cm = np.asarray(cm, np.float64)
    vals = []
    for k in range(cm.shape[0]):
        den = cm[k, :].sum()
        if den > 0:
            vals.append(cm[k, k] / den)
    return float(np.mean(vals)) if vals else 0.0


def accuracy_from_cm(cm):
    cm = np.asarray(cm, np.float64)
    den = cm.sum()
    return float(np.trace(cm) / den) if den > 0 else 0.0


def binary_precision_recall_f1(scores, labels, threshold=0.5):
    s = np.asarray(scores, np.float64).reshape(-1)
    y = np.asarray(labels, bool).reshape(-1)
    m = np.isfinite(s)
    s = s[m]
    y = y[m]
    p = s >= float(threshold)
    tp = int(np.sum(p & y))
    fp = int(np.sum(p & ~y))
    fn = int(np.sum(~p & y))
    tn = int(np.sum(~p & ~y))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    tpr = recall
    tnr = tn / max(tn + fp, 1)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(0.5 * (tpr + tnr)),
        "accuracy": float((tp + tn) / max(tp + fp + fn + tn, 1)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def binary_auprc(scores, labels, max_points=500000, seed=10):
    s = np.asarray(scores, np.float64).reshape(-1)
    y = np.asarray(labels, bool).reshape(-1)
    m = np.isfinite(s)
    s = s[m]
    y = y[m]
    if len(s) == 0 or int(y.sum()) == 0:
        return None
    if len(s) > int(max_points):
        rng = np.random.default_rng(seed)
        ids = rng.choice(len(s), size=int(max_points), replace=False)
        s = s[ids]
        y = y[ids]
    order = np.argsort(-s, kind="mergesort")
    yy = y[order].astype(np.float64)
    tp = np.cumsum(yy)
    fp = np.cumsum(1.0 - yy)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / max(float(yy.sum()), 1.0)
    # Step-wise AP, equivalent to mean precision at positive ranks.
    return float(np.sum(precision * yy) / max(float(yy.sum()), 1.0))


def binary_metrics(scores, labels, threshold=0.5, max_points=500000, seed=10):
    out = binary_precision_recall_f1(scores, labels, threshold)
    out["auroc"] = binary_auroc(scores, labels, max_points=max_points, seed=seed)
    out["auprc"] = binary_auprc(scores, labels, max_points=max_points, seed=seed + 1)
    y = np.asarray(labels, np.float64).reshape(-1)
    s = np.asarray(scores, np.float64).reshape(-1)
    m = np.isfinite(s)
    if np.any(m):
        out["prevalence"] = float(np.mean(y[m]))
        out["brier"] = float(np.mean((s[m] - y[m]) ** 2))
    else:
        out["prevalence"] = None
        out["brier"] = None
    return out


def top_fraction_recall_lift(scores, labels, fraction=0.10):
    s = np.asarray(scores, np.float64).reshape(-1)
    y = np.asarray(labels, bool).reshape(-1)
    m = np.isfinite(s)
    s = s[m]
    y = y[m]
    if len(s) == 0 or int(y.sum()) == 0:
        return {"fraction": float(fraction), "recall": None, "precision": None, "lift": None}
    k = max(1, int(round(len(s) * float(fraction))))
    ids = np.argpartition(s, -k)[-k:]
    recall = float(y[ids].sum() / max(y.sum(), 1))
    precision = float(y[ids].mean())
    prevalence = float(y.mean())
    lift = precision / prevalence if prevalence > 0 else None
    return {"fraction": float(fraction), "recall": recall, "precision": precision, "lift": None if lift is None else float(lift)}
