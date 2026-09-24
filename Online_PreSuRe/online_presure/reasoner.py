from __future__ import annotations
import numpy as np


class NoOpReasoner:
    name = 'none'
    def rerank(self, query_meta, retrieved):
        if retrieved is None:
            return None
        out = dict(retrieved)
        out['reasoner_weight'] = np.ones(len(retrieved['case_index']), dtype=np.float32)
        return out


def build_reasoner(cfg):
    mode = str(cfg.get('mode', 'none')).lower()
    if mode == 'none':
        return NoOpReasoner()
    if mode == 'qwen':
        raise RuntimeError(
            'Qwen arbitration is intentionally disabled in Online PreSuRe v2.0. '
            'First establish a positive numerical RAG fast-path increment over slow online adaptation.'
        )
    raise ValueError(f'Unknown reasoner mode: {mode}')
