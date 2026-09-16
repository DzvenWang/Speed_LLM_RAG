from __future__ import annotations

import gc
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


AUGMENTATIONS = (
    "none",
    "smoothed_output",
    "overestimated_output",
    "underestimated_output",
    "upward_trend_output",
    "downward_trend_output",
)

AUGMENTATION_DESCRIPTIONS = {
    "none": "original prediction",
    "smoothed_output": "3-step smoothed prediction",
    "overestimated_output": "prediction multiplied by 1.05",
    "underestimated_output": "prediction multiplied by 0.95",
    "upward_trend_output": "prediction with +1% additional multiplicative trend per future step",
    "downward_trend_output": "prediction with -1% additional multiplicative trend per future step",
}

CHOICE_META = tuple(
    (branch, aug)
    for branch in ("GraphBranch", "HypergraphBranch")
    for aug in AUGMENTATIONS
)

_FORBIDDEN_PROMPT_KEYS = (
    "ground_truth",
    "ground truth",
    "optimal_idx",
    "oracle_choice",
    "future_label",
    "graph_error",
    "hyper_error",
)


def _smooth_output(raw: torch.Tensor, window_size: int = 3) -> torch.Tensor:
    """Exact tensor form used by vendor/LEAF/adapter/aug_adapter.py::smooth_output.

    raw: [B, P, N, 1] in physical units.
    """
    if raw.ndim != 4 or raw.shape[-1] != 1:
        raise ValueError(f"expected [B,P,N,1], got {tuple(raw.shape)}")
    window = torch.ones(window_size, dtype=raw.dtype, device=raw.device) / float(window_size)
    window = window.view(1, 1, -1, 1)
    b, p, n, d = raw.shape
    z = raw.permute(0, 2, 1, 3).contiguous().view(b * n, 1, p, d)
    z = F.pad(z, (0, 0, window_size // 2, window_size // 2), mode="replicate")
    z = F.conv2d(z, window, padding=0)
    return z.view(b, n, p, d).permute(0, 2, 1, 3).contiguous()


def augment_prediction(raw: torch.Tensor, aug_type: str) -> torch.Tensor:
    if aug_type == "none":
        return raw
    if aug_type == "smoothed_output":
        return _smooth_output(raw)
    if aug_type == "overestimated_output":
        return raw * 1.05
    if aug_type == "underestimated_output":
        return raw * 0.95
    if aug_type == "upward_trend_output":
        factor = torch.arange(raw.shape[1], dtype=raw.dtype, device=raw.device).view(1, -1, 1, 1) * 0.01 + 1.0
        return raw * factor
    if aug_type == "downward_trend_output":
        factor = torch.arange(raw.shape[1], dtype=raw.dtype, device=raw.device).view(1, -1, 1, 1) * -0.01 + 1.0
        return raw * factor
    raise ValueError(f"unknown LEAF augmentation: {aug_type}")


def build_choice_tensor(graph_raw: torch.Tensor, hyper_raw: torch.Tensor) -> torch.Tensor:
    """Return the official LEAF 12-choice set as [12,P,N,1] in physical units."""
    if graph_raw.shape != hyper_raw.shape or graph_raw.shape[0] != 1:
        raise ValueError(f"branch shapes must match with B=1: {tuple(graph_raw.shape)} vs {tuple(hyper_raw.shape)}")
    choices = []
    for branch_raw in (graph_raw, hyper_raw):
        for aug in AUGMENTATIONS:
            choices.append(augment_prediction(branch_raw, aug)[0])
    return torch.stack(choices, dim=0)


def selected_positive_and_negatives(
    choices_raw: torch.Tensor,
    selected_1based: np.ndarray,
    mean: float,
    std: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Match vendor LLMSelectorFromJson semantics for per-node selected choices.

    choices_raw: [12,P,N,1], selected_1based: [N] in 1..12.
    Returns normalized positive [1,P,N,1] and 11 normalized negative tensors.
    """
    if choices_raw.ndim != 4 or choices_raw.shape[0] != 12 or choices_raw.shape[-1] != 1:
        raise ValueError(f"invalid LEAF choice tensor {tuple(choices_raw.shape)}")
    p, n = int(choices_raw.shape[1]), int(choices_raw.shape[2])
    sel = np.asarray(selected_1based, dtype=np.int64)
    if sel.shape != (n,) or np.any(sel < 1) or np.any(sel > 12):
        raise ValueError("selected choices must be shape [N] with values 1..12")
    positive = torch.empty((1, p, n, 1), dtype=choices_raw.dtype, device=choices_raw.device)
    negatives = [torch.empty_like(positive) for _ in range(11)]
    for node in range(n):
        idx = int(sel[node]) - 1
        positive[0, :, node, 0] = choices_raw[idx, :, node, 0]
        neg_idx = [j for j in range(12) if j != idx]
        for k, j in enumerate(neg_idx):
            negatives[k][0, :, node, 0] = choices_raw[j, :, node, 0]
    inv = 1.0 / max(float(std), 1e-6)
    positive = ((positive - float(mean)) * inv).detach()
    negatives = [((z - float(mean)) * inv).detach() for z in negatives]
    return positive, negatives


def multi_ranking_loss(output: torch.Tensor, positive: torch.Tensor, negatives: list[torch.Tensor], margin: float = 0.0) -> torch.Tensor:
    """Exact Huber ranking objective used by vendor/LEAF/adapter/aug_adapter.py."""
    negative = torch.cat(negatives, dim=0)
    # Vendor LEAF intentionally broadcasts one predictor output against all negatives.
    def huber(a: torch.Tensor, b: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        diff = torch.abs(a - b)
        return torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    pos_loss = huber(output, positive).mean(dim=(1, 3))
    neg_loss = huber(output, negative).mean(dim=(1, 3))
    largest_positive = pos_loss.max(dim=0)[0]
    smallest_negative = neg_loss.min(dim=0)[0]
    return torch.relu(largest_positive - smallest_negative + float(margin)).mean()


def _int_series(x: np.ndarray) -> str:
    # The official dump path casts histories and candidates to integer lists.
    vals = np.asarray(x).astype(np.int64, copy=False).tolist()
    return "[" + ", ".join(str(int(v)) for v in vals) + "]"


def build_selector_prompt(
    *,
    node_idx: int,
    num_nodes: int,
    graph_degree: int,
    history_raw: np.ndarray,
    history_observed: np.ndarray,
    history_tod: np.ndarray,
    history_dow: np.ndarray,
    future_tod: np.ndarray,
    future_dow: np.ndarray,
    choices_raw: np.ndarray,
) -> str:
    """LEAF-style prompt adapted to TraffiDent incomplete sensing.

    The paper specifies general data information, spatial information, temporal
    information, historical data, task instruction, and the full choice set.
    No future labels or candidate errors are accepted here.
    """
    choices_raw = np.asarray(choices_raw)
    if choices_raw.shape[0] != 12:
        raise ValueError(f"expected 12 LEAF choices, got {choices_raw.shape}")
    obs = np.asarray(history_observed).astype(bool)
    hist = np.asarray(history_raw, dtype=np.float32)
    visible = [str(int(hist[i])) if bool(obs[i]) else "?" for i in range(len(hist))]
    candidate_lines = []
    for i, ((branch, aug), candidate) in enumerate(zip(CHOICE_META, choices_raw), start=1):
        candidate_lines.append(
            f"Choice {i}: {branch}; {AUGMENTATION_DESCRIPTIONS[aug]}; values={_int_series(candidate)}"
        )
    text = (
        "You are the frozen prediction selector in the LEAF traffic forecasting framework.\n"
        "The data are 5-minute traffic-state/speed measurements. Some historical sensor values are unavailable and are shown as '?'. "
        "Use only the supplied history, sensing pattern, road-network context, time context, and candidate forecasts.\n"
        f"Sensor: node {node_idx} of {num_nodes}; road-graph degree={graph_degree}.\n"
        f"Historical values (oldest to newest): [{', '.join(visible)}]\n"
        f"Historical observed mask (1=observed, 0=missing): {_int_series(obs.astype(np.int64))}\n"
        f"History time-of-day slots (0..287): {_int_series(history_tod)}; day-of-week (0..6): {_int_series(history_dow)}\n"
        f"Forecast time-of-day slots (0..287): {_int_series(future_tod)}; day-of-week (0..6): {_int_series(future_dow)}\n"
        "No external incident or special-event metadata is supplied; do not invent any.\n"
        "Choose the single most plausible 12-step forecast from the LEAF choice set.\n"
        + "\n".join(candidate_lines)
        + "\nReturn ONLY one integer from 1 to 12."
    )
    low = text.lower()
    for key in _FORBIDDEN_PROMPT_KEYS:
        if key in low:
            raise RuntimeError(f"forbidden label-bearing prompt token: {key}")
    return text


def parse_choice(text: str, fallback: int = 1) -> tuple[int, bool]:
    m = re.search(r"(?<!\d)(1[0-2]|[1-9])(?!\d)", str(text))
    if m is None:
        return int(fallback), True
    value = int(m.group(1))
    if not (1 <= value <= 12):
        return int(fallback), True
    return value, False


def resolve_snapshot(model_cache_root: str | Path, preferred_snapshot: str | None = None) -> Path:
    root = Path(model_cache_root)
    if preferred_snapshot:
        p = root / "snapshots" / preferred_snapshot
        if not p.is_dir():
            raise FileNotFoundError(f"configured Qwen snapshot does not exist: {p}")
        return p
    snaps = sorted(p for p in (root / "snapshots").iterdir() if p.is_dir())
    if len(snaps) != 1:
        raise RuntimeError(f"expected exactly one local Qwen snapshot under {root/'snapshots'}, found {len(snaps)}")
    return snaps[0]


@dataclass
class SelectorStats:
    total: int = 0
    invalid: int = 0
    oom_reductions: int = 0

    def as_dict(self) -> dict:
        return {
            "total": int(self.total),
            "invalid": int(self.invalid),
            "invalid_fraction": float(self.invalid / max(self.total, 1)),
            "oom_reductions": int(self.oom_reductions),
        }


class LocalQwenLeafSelector:
    """Frozen local Qwen selector; the only model substitution in LEAF-Qwen1.5B."""

    def __init__(
        self,
        model_cache_root: str | Path,
        snapshot: str | None,
        device: torch.device,
        batch_size: int = 32,
        max_input_tokens: int = 2048,
        max_new_tokens: int = 4,
        fallback_choice: int = 1,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.snapshot = resolve_snapshot(model_cache_root, snapshot)
        self.device = device
        self.batch_size = int(batch_size)
        self.max_input_tokens = int(max_input_tokens)
        self.max_new_tokens = int(max_new_tokens)
        self.fallback_choice = int(fallback_choice)
        self.stats = SelectorStats()

        self.tokenizer = AutoTokenizer.from_pretrained(self.snapshot, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            self.snapshot,
            local_files_only=True,
            torch_dtype=torch.float16,
        ).to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    def _chat(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are the LEAF frozen LLM selector. Follow the user's multiple-choice instruction exactly."},
            {"role": "user", "content": prompt},
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.inference_mode()
    def select_prompts(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        rendered = [self._chat(p) for p in prompts]
        selections = np.empty(len(rendered), dtype=np.uint8)
        invalid = np.zeros(len(rendered), dtype=np.uint8)
        start = 0
        active_bs = max(1, self.batch_size)
        while start < len(rendered):
            end = min(len(rendered), start + active_bs)
            chunk = rendered[start:end]
            try:
                tok = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_tokens,
                )
                tok = {k: v.to(self.device, non_blocking=True) for k, v in tok.items()}
                input_len = int(tok["input_ids"].shape[1])
                out = self.model.generate(
                    **tok,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
                decoded = self.tokenizer.batch_decode(out[:, input_len:], skip_special_tokens=True)
                for local_i, text in enumerate(decoded):
                    choice, bad = parse_choice(text, self.fallback_choice)
                    selections[start + local_i] = choice
                    invalid[start + local_i] = 1 if bad else 0
                self.stats.total += len(chunk)
                self.stats.invalid += int(invalid[start:end].sum())
                start = end
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                if active_bs <= 1:
                    raise
                active_bs = max(1, active_bs // 2)
                self.stats.oom_reductions += 1
                print(f"LEAF-QWEN selector CUDA OOM: reducing selector_batch_size to {active_bs}", flush=True)
        return selections, invalid

    def metadata(self) -> dict:
        return {
            "selector_model": "Qwen2.5-1.5B-Instruct",
            "snapshot": str(self.snapshot),
            "dtype": "float16",
            "local_files_only": True,
            "do_sample": False,
            "max_new_tokens": self.max_new_tokens,
            "configured_batch_size": self.batch_size,
            "stats": self.stats.as_dict(),
        }
