from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


ACTION_USE = 'USE'
ACTION_REJECT = 'REJECT'
ACTION_ABSTAIN = 'ABSTAIN'
VALID_ACTIONS = (ACTION_USE, ACTION_REJECT, ACTION_ABSTAIN)


def resolve_hf_snapshot(cache_root: str | Path) -> Path:
    """Resolve either a HF snapshot directory or the models--org--name cache root.

    The user-provided Qwen paths are HuggingFace cache roots.  No network access is
    attempted: this function only inspects local ``snapshots/*`` directories.
    """
    root = Path(cache_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f'local HuggingFace cache does not exist: {root}')
    if (root / 'config.json').exists():
        return root
    snaps = root / 'snapshots'
    if not snaps.exists():
        raise FileNotFoundError(f'no config.json and no snapshots/ under {root}')
    candidates = [p for p in snaps.iterdir() if p.is_dir() and (p / 'config.json').exists()]
    if not candidates:
        raise FileNotFoundError(f'no local snapshot containing config.json under {snaps}')
    # Prefer newest local snapshot; ties are deterministic by name.
    candidates.sort(key=lambda p: (p.stat().st_mtime_ns, p.name))
    return candidates[-1]


def decision_label(
    utilities: Sequence[float],
    valid: Sequence[bool],
    beneficial_threshold: float = 0.02,
    reject_threshold: float = -0.02,
) -> Dict[str, object]:
    """Privileged Train/evaluation label with an explicit no-support option.

    USE     : at least one proposal has material positive utility.
    REJECT  : even the best available proposal is clearly harmful.
    ABSTAIN : best proposal is near-neutral / not materially beneficial.

    The current-query future may be used to build these labels on Train and for
    Validation reporting, but never appears in the deployment prompt.
    """
    u = np.asarray(utilities, np.float64)
    m = np.asarray(valid, bool)
    if u.shape != m.shape or u.ndim != 1:
        raise ValueError(f'utility/mask mismatch: {u.shape} vs {m.shape}')
    if not np.any(m):
        return {'action': ACTION_ABSTAIN, 'proposal_index': -1, 'best_utility': 0.0, 'oracle_value': 0.0}
    vv = np.where(m, u, -np.inf)
    best = int(np.argmax(vv))
    best_u = float(vv[best])
    if best_u >= float(beneficial_threshold):
        action, pidx = ACTION_USE, best
    elif best_u <= float(reject_threshold):
        action, pidx = ACTION_REJECT, -1
    else:
        action, pidx = ACTION_ABSTAIN, -1
    return {
        'action': action,
        'proposal_index': int(pidx),
        'best_utility': best_u,
        'oracle_value': max(0.0, best_u),
    }


def _fmt(v, digits=3):
    if v is None:
        return 'NA'
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(x):
        return 'NA'
    return f'{x:.{digits}f}'


def reasoning_system_prompt() -> str:
    return (
        'You are a traffic forecasting support-value reasoner. '
        'The numerical forecaster has already produced a base forecast. '
        'Do NOT predict traffic speed and do NOT invent future observations. '
        'Judge only whether one of the retrieved historical support proposals is useful enough to apply. '
        'Choose exactly one action: USE one proposal, REJECT when the proposals appear harmful, '
        'or ABSTAIN when evidence is weak or conflicting. '
        'Return JSON only with keys action and proposal. '
        'proposal must be P0, P1, P2, P3, or NONE.'
    )


def format_reasoning_user(record: Dict[str, object]) -> str:
    props = record.get('proposals', [])
    lines = [
        'Current decision context:',
        f"- predicted severe-error need probability: {_fmt(record.get('need_probability'))}",
        f"- forecast horizon index: {int(record.get('horizon_index', -1)) + 1}",
        f"- unique support proposals: {int(record.get('unique_proposals', len(props)))}",
        f"- expert disagreement indicator: {_fmt(record.get('expert_disagreement'))}",
        'Retrieved support proposals:',
    ]
    for p in props:
        if not p.get('valid', True):
            continue
        ep = p.get('expert_probabilities', {})
        proposed = ','.join(p.get('proposed_by', [])) or 'none'
        lines.extend([
            f"{p['id']}:",
            f"  proposed_by={proposed}; agreement={_fmt(p.get('agreement_ratio'))}",
            f"  M3_confidence={_fmt(p.get('m3_confidence'))}; M3_margin={_fmt(p.get('m3_margin'))}",
            '  expert_value_scores=' + ', '.join(
                f'{k}:{_fmt(ep.get(k))}' for k in ['generic', 'hard', 'pred10', 'pred20']
            ),
            f"  local_historical_residual={_fmt(p.get('local_residual'))}; "
            f"residual_mean={_fmt(p.get('residual_mean'))}; residual_std={_fmt(p.get('residual_std'))}; "
            f"residual_maxabs={_fmt(p.get('residual_maxabs'))}",
            f"  query_support_cosine={_fmt(p.get('context_cosine'))}; "
            f"context_absdiff_mean={_fmt(p.get('context_absdiff_mean'))}",
        ])
    lines.extend([
        'Decision rule:',
        '- USE only when one proposal has sufficiently coherent evidence to justify correction.',
        '- REJECT when available proposals look actively harmful or contradictory.',
        '- ABSTAIN when no proposal is clearly worth using.',
        'Return JSON only, for example {"action":"USE","proposal":"P1"}.',
    ])
    return '\n'.join(lines)


def target_response(record: Dict[str, object]) -> str:
    action = str(record['label_action']).upper()
    proposal = 'NONE'
    if action == ACTION_USE:
        proposal = f"P{int(record['label_proposal_index'])}"
    return json.dumps({'action': action, 'proposal': proposal}, separators=(',', ':'))


def parse_reasoner_output(text: str, valid_proposals: Sequence[str]) -> Dict[str, object]:
    raw = (text or '').strip()
    valid = {str(x).upper() for x in valid_proposals}
    obj = None
    # Prefer the first JSON object; Qwen sometimes wraps it in markdown fences.
    match = re.search(r'\{[^{}]*\}', raw, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
        except Exception:
            obj = None
    if isinstance(obj, dict):
        action = str(obj.get('action', '')).upper().strip()
        proposal = str(obj.get('proposal', 'NONE')).upper().strip()
    else:
        up = raw.upper()
        action = next((a for a in VALID_ACTIONS if re.search(rf'\b{a}\b', up)), '')
        pm = re.search(r'\bP([0-3])\b', up)
        proposal = f'P{pm.group(1)}' if pm else 'NONE'
    if action not in VALID_ACTIONS:
        return {'action': ACTION_ABSTAIN, 'proposal': 'NONE', 'parse_ok': False, 'raw': raw}
    if action == ACTION_USE:
        if proposal not in valid:
            return {'action': ACTION_ABSTAIN, 'proposal': 'NONE', 'parse_ok': False, 'raw': raw}
    else:
        proposal = 'NONE'
    return {'action': action, 'proposal': proposal, 'parse_ok': True, 'raw': raw}


def proposal_prompt_features(feature_row: Sequence[float], expert_order=('generic','hard','pred10','pred20')) -> Dict[str, object]:
    """Decode the stable v0.7.6 proposal feature contract into prompt-safe summaries."""
    x = np.asarray(feature_row, np.float32).reshape(-1)
    E = len(expert_order)
    base = 4 * E + 13  # v0.7.6 fixed scalar prefix = 29 for E=4
    if len(x) < base + 1:
        raise ValueError(f'proposal feature vector too short: {len(x)}')
    cand_prob = x[0:E]
    proposer = x[E:2*E]
    top1_prob = x[2*E:3*E]
    top1_margin = x[3*E:4*E]
    o = 4 * E
    need, m3cand, m31, m32, m3margin, m3mean, agree = [float(v) for v in x[o:o+7]]
    local_rr, abs_rr, rr_mean, rr_std, rr_maxabs, horizon = [float(v) for v in x[o+7:o+13]]
    # If q/support/difference contexts are present, they occupy 3 equally-sized blocks
    # before the final unique-proposal fraction.
    ctx_len = len(x) - (base + 1)
    cosine, diff_mean = None, None
    if ctx_len > 0 and ctx_len % 3 == 0:
        F = ctx_len // 3
        q = x[base:base+F]
        s = x[base+F:base+2*F]
        d = x[base+2*F:base+3*F]
        den = float(np.linalg.norm(q) * np.linalg.norm(s))
        cosine = float(np.dot(q, s) / den) if den > 1e-8 else 0.0
        diff_mean = float(np.mean(np.abs(d)))
    return {
        'need_probability': need,
        'm3_confidence': m3cand,
        'm3_top1': m31,
        'm3_top2': m32,
        'm3_margin': m3margin,
        'm3_mean': m3mean,
        'agreement_ratio': agree,
        'local_residual': local_rr,
        'residual_abs': abs_rr,
        'residual_mean': rr_mean,
        'residual_std': rr_std,
        'residual_maxabs': rr_maxabs,
        'horizon_fraction': horizon,
        'context_cosine': cosine,
        'context_absdiff_mean': diff_mean,
        'expert_probabilities': {k: float(cand_prob[i]) for i, k in enumerate(expert_order)},
        'proposed_by': [k for i, k in enumerate(expert_order) if proposer[i] > 0.5],
        'expert_top1_probabilities': {k: float(top1_prob[i]) for i, k in enumerate(expert_order)},
        'expert_top1_margins': {k: float(top1_margin[i]) for i, k in enumerate(expert_order)},
        'unique_fraction': float(x[-1]),
    }


def load_jsonl(path: str | Path) -> List[dict]:
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl(path: str | Path, rows: Iterable[dict]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + '\n')


@dataclass
class QwenDecisionMetrics:
    beneficial_threshold: float = 0.02
    harmful_threshold: float = -0.02
    total: int = 0
    parse_ok: int = 0
    action_correct: int = 0
    use_tp: int = 0
    use_pred: int = 0
    use_true: int = 0
    reject_tp: int = 0
    reject_pred: int = 0
    reject_true: int = 0
    abstain_tp: int = 0
    abstain_pred: int = 0
    abstain_true: int = 0
    selected_beneficial: int = 0
    selected_harmful: int = 0
    selected_neutral: int = 0
    harmful_regret: int = 0
    oracle_beneficial: int = 0
    sum_value: float = 0.0
    sum_oracle_value: float = 0.0
    sum_regret: float = 0.0
    regrets: List[float] = field(default_factory=list)

    def update(self, record: dict, pred_action: str, pred_proposal: str, parse_ok: bool = True):
        self.total += 1
        self.parse_ok += int(bool(parse_ok))
        true_action = str(record['label_action'])
        self.action_correct += int(pred_action == true_action)
        for name in ('use', 'reject', 'abstain'):
            a = name.upper()
            setattr(self, f'{name}_true', getattr(self, f'{name}_true') + int(true_action == a))
            setattr(self, f'{name}_pred', getattr(self, f'{name}_pred') + int(pred_action == a))
            setattr(self, f'{name}_tp', getattr(self, f'{name}_tp') + int(pred_action == a and true_action == a))

        props = record.get('proposals', [])
        utils = {str(p['id']).upper(): float(p['target_utility']) for p in props if p.get('valid', True)}
        if pred_action == ACTION_USE and pred_proposal in utils:
            value = float(utils[pred_proposal])
        else:
            value = 0.0  # keep base forecast / no support correction
        oracle = max(0.0, max(utils.values()) if utils else 0.0)
        regret = max(0.0, oracle - value)
        self.sum_value += value
        self.sum_oracle_value += oracle
        self.sum_regret += regret
        self.regrets.append(regret)
        self.oracle_beneficial += int(oracle >= self.beneficial_threshold)
        self.selected_beneficial += int(value >= self.beneficial_threshold)
        self.selected_harmful += int(value <= self.harmful_threshold)
        self.selected_neutral += int(self.harmful_threshold < value < self.beneficial_threshold)
        self.harmful_regret += int(value <= self.harmful_threshold and oracle >= self.beneficial_threshold)

    def summary(self):
        n = max(self.total, 1)
        r = np.asarray(self.regrets, np.float64) if self.regrets else np.zeros(1)
        def pr(tp, pred): return None if pred == 0 else tp / pred
        def rc(tp, true): return None if true == 0 else tp / true
        return {
            'count': self.total,
            'parse_rate': self.parse_ok / n,
            'action_accuracy': self.action_correct / n,
            'use_precision': pr(self.use_tp, self.use_pred),
            'use_recall': rc(self.use_tp, self.use_true),
            'reject_precision': pr(self.reject_tp, self.reject_pred),
            'reject_recall': rc(self.reject_tp, self.reject_true),
            'abstain_precision': pr(self.abstain_tp, self.abstain_pred),
            'abstain_recall': rc(self.abstain_tp, self.abstain_true),
            'selected_beneficial_rate': self.selected_beneficial / n,
            'selected_harmful_rate': self.selected_harmful / n,
            'selected_neutral_rate': self.selected_neutral / n,
            'oracle_beneficial_rate': self.oracle_beneficial / n,
            'harmful_regret_rate': self.harmful_regret / n,
            'mean_decision_utility': self.sum_value / n,
            'mean_oracle_decision_utility': self.sum_oracle_value / n,
            'mean_selection_regret': self.sum_regret / n,
            'median_selection_regret': float(np.median(r)),
            'p90_selection_regret': float(np.quantile(r, 0.90)),
        }
