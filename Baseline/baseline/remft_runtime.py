from __future__ import annotations

import os
from pathlib import Path


def project_root(baseline_root: Path) -> Path:
    env = os.getenv('SPEED_LLM_RAG_ROOT', '').strip()
    return Path(env).expanduser().resolve() if env else baseline_root.resolve().parent


def resolve_remft_bert_path(baseline_root: Path, *, require: bool = True) -> Path:
    """Resolve the single local BERT checkpoint used by ReMFT.

    Priority:
      1) REMFT_BERT_PATH (explicit override)
      2) <SPEED_LLM_RAG_ROOT>/model_cache/bert-base-uncased (shared project cache)
      3) <Baseline>/vendor/ReMFT/LLM/BERT (legacy compatibility fallback)
    """
    explicit = os.getenv('REMFT_BERT_PATH', '').strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend([
        project_root(baseline_root) / 'model_cache' / 'bert-base-uncased',
        baseline_root / 'vendor' / 'ReMFT' / 'LLM' / 'BERT',
    ])
    for path in candidates:
        path = path.resolve()
        if (path / 'config.json').is_file() and any(
            (path / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin')
        ):
            return path
    if require:
        rendered = '\n  - '.join(str(p.resolve()) for p in candidates)
        raise FileNotFoundError(
            'ReMFT local BERT checkpoint not found. Checked:\n  - ' + rendered
            + '\nSet REMFT_BERT_PATH or place bert-base-uncased under '
              '<SPEED_LLM_RAG_ROOT>/model_cache/bert-base-uncased.'
        )
    return candidates[0].resolve()


def bind_remft_bert_env(baseline_root: Path) -> Path:
    path = resolve_remft_bert_path(baseline_root, require=True)
    os.environ['REMFT_BERT_PATH'] = str(path)
    return path
