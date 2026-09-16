from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class LeafSelectorOracle:
    """Query-node oracle for the current LEAF selector contract.

    A single branch is selected for each (query, node) pair and that same
    branch supplies all forecast horizons for the node.  This intentionally
    matches ``leaf_runner.selector_apply``; it is *not* a point-wise oracle.

    The oracle uses validation future labels and is diagnostic-only.
    """

    choices: np.ndarray  # uint8 [B, N], 1=Graph, 2=Hyper
    eligible: np.ndarray  # bool [B, N], at least one valid future target
    graph_mae: np.ndarray  # float32 [B, N]
    hyper_mae: np.ndarray  # float32 [B, N]


def _check_shapes(
    graph: np.ndarray,
    hyper: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> tuple[int, int, int]:
    shape = tuple(int(x) for x in graph.shape)
    if len(shape) != 3:
        raise ValueError(f"LEAF predictions must have [B,P,N] shape, got {shape}")
    for name, arr in (("hyper", hyper), ("target", target), ("valid", valid)):
        if tuple(arr.shape) != shape:
            raise ValueError(f"{name} shape {tuple(arr.shape)} != graph shape {shape}")
    return shape


def query_node_mae(
    pred: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return valid-future MAE and eligibility for each query-node pair."""
    if tuple(pred.shape) != tuple(target.shape) or tuple(pred.shape) != tuple(valid.shape):
        raise ValueError("pred/target/valid shapes must match")
    m = np.asarray(valid, dtype=bool)
    count = m.sum(axis=1, dtype=np.int32)
    eligible = count > 0
    # Accumulate in float64 for deterministic, stable branch comparisons.
    err_sum = np.where(
        m,
        np.abs(np.asarray(pred, np.float64) - np.asarray(target, np.float64)),
        0.0,
    ).sum(axis=1, dtype=np.float64)
    mae = np.full(count.shape, np.inf, dtype=np.float32)
    mae[eligible] = (err_sum[eligible] / count[eligible]).astype(np.float32)
    return mae, eligible


def build_query_node_oracle(
    graph: np.ndarray,
    hyper: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    tie_break: str = "hyper",
    tie_atol: float = 1e-12,
) -> LeafSelectorOracle:
    """Build the future-label oracle at LEAF's query-node selector granularity.

    Strictly lower 12-step valid-future MAE wins.  Exact/numerical ties are
    resolved toward Hyper by default because Hyper is the stronger fixed branch
    in the current Development run.  Query-node pairs with no valid future
    target also fall back to Hyper and are excluded from win-rate denominators.
    """
    _check_shapes(graph, hyper, target, valid)
    if tie_break not in {"graph", "hyper"}:
        raise ValueError("tie_break must be 'graph' or 'hyper'")

    g_mae, g_ok = query_node_mae(graph, target, valid)
    h_mae, h_ok = query_node_mae(hyper, target, valid)
    if not np.array_equal(g_ok, h_ok):
        raise AssertionError("Graph/Hyper eligibility mismatch")
    eligible = g_ok

    diff = np.zeros(g_mae.shape, dtype=np.float64)
    diff[eligible] = (
        np.asarray(g_mae[eligible], np.float64)
        - np.asarray(h_mae[eligible], np.float64)
    )
    graph_win = eligible & (diff < -float(tie_atol))
    hyper_win = eligible & (diff > float(tie_atol))
    ties = eligible & ~(graph_win | hyper_win)

    default = np.uint8(1 if tie_break == "graph" else 2)
    choices = np.full(g_mae.shape, default, dtype=np.uint8)
    choices[graph_win] = np.uint8(1)
    choices[hyper_win] = np.uint8(2)
    # no-valid pairs intentionally retain deterministic fixed-branch fallback.

    return LeafSelectorOracle(
        choices=choices,
        eligible=eligible,
        graph_mae=g_mae,
        hyper_mae=h_mae,
    )


def apply_query_node_choices(
    graph: np.ndarray,
    hyper: np.ndarray,
    choices: np.ndarray,
    *,
    out_path: Path | None = None,
) -> np.ndarray:
    """Apply [B,N] LEAF branch choices to [B,P,N] predictions.

    When ``out_path`` is supplied the oracle prediction is written as an NPY
    memmap so the diagnostic does not require a second full prediction tensor
    in RAM.
    """
    if tuple(graph.shape) != tuple(hyper.shape):
        raise ValueError("graph and hyper shapes must match")
    if graph.ndim != 3 or tuple(choices.shape) != (graph.shape[0], graph.shape[2]):
        raise ValueError(
            f"choices shape {tuple(choices.shape)} must be [B,N]="
            f"{(graph.shape[0], graph.shape[2])}"
        )
    if np.any((choices != 1) & (choices != 2)):
        bad = np.unique(choices[(choices != 1) & (choices != 2)])
        raise ValueError(f"invalid LEAF choices {bad.tolist()}; expected only 1/2")

    if out_path is None:
        out = np.empty(graph.shape, dtype=np.float32)
    else:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32, shape=graph.shape)

    # Node-wise choice is broadcast across the forecast horizon.
    pick_graph = choices == 1
    for q in range(graph.shape[0]):
        gm = pick_graph[q]
        out[q, :, gm] = np.asarray(graph[q, :, gm], dtype=np.float32)
        out[q, :, ~gm] = np.asarray(hyper[q, :, ~gm], dtype=np.float32)
    if hasattr(out, "flush"):
        out.flush()
    return out


def choice_statistics(oracle: LeafSelectorOracle) -> dict[str, Any]:
    eligible = oracle.eligible
    n = int(eligible.sum())
    graph_win = eligible & (oracle.choices == 1)
    # With Hyper tie-break, choices==2 contains both strict Hyper wins and ties.
    strict_g = eligible & (oracle.graph_mae < oracle.hyper_mae)
    strict_h = eligible & (oracle.hyper_mae < oracle.graph_mae)
    tie = eligible & ~(strict_g | strict_h)

    # Positive value means an oracle switch away from always-Hyper can help.
    hyper_regret = np.zeros(oracle.graph_mae.shape, dtype=np.float64)
    hyper_regret[eligible] = np.maximum(
        np.asarray(oracle.hyper_mae[eligible], np.float64)
        - np.minimum(
            np.asarray(oracle.graph_mae[eligible], np.float64),
            np.asarray(oracle.hyper_mae[eligible], np.float64),
        ),
        0.0,
    )

    return {
        "eligible_query_nodes": n,
        "total_query_nodes": int(eligible.size),
        "eligible_fraction": float(n / max(int(eligible.size), 1)),
        "graph_strict_win_count": int(strict_g.sum()),
        "hyper_strict_win_count": int(strict_h.sum()),
        "tie_count": int(tie.sum()),
        "graph_strict_win_fraction": float(strict_g.sum() / max(n, 1)),
        "hyper_strict_win_fraction": float(strict_h.sum() / max(n, 1)),
        "tie_fraction": float(tie.sum() / max(n, 1)),
        "oracle_choice_graph_fraction": float(graph_win.sum() / max(n, 1)),
        "oracle_choice_hyper_fraction": float((eligible & (oracle.choices == 2)).sum() / max(n, 1)),
        "mean_query_node_hyper_regret_if_always_hyper": float(hyper_regret[eligible].mean()) if n else 0.0,
        "p95_query_node_hyper_regret_if_always_hyper": float(np.quantile(hyper_regret[eligible], 0.95)) if n else 0.0,
    }


def relative_gain(baseline_value: float, improved_value: float) -> float:
    baseline_value = float(baseline_value)
    improved_value = float(improved_value)
    if not np.isfinite(baseline_value) or baseline_value == 0.0:
        return float("nan")
    return float((baseline_value - improved_value) / baseline_value)
