from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class MatureEvidenceCase:
    index: int
    origin: int
    key: np.ndarray
    tod: int
    dow: int
    x_obs: np.ndarray
    history_observed: np.ndarray
    y_raw: np.ndarray
    future_valid: np.ndarray


class MatureEvidenceMemory:
    """Causal explicit memory of already-matured state -> future transitions."""
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.cases = deque(maxlen=self.capacity)

    def __len__(self): return len(self.cases)
    def add(self, case: MatureEvidenceCase): self.cases.append(case)

    def recent_indices(self, n: int):
        if not self.cases: return []
        rows = list(self.cases)[-int(n):]
        return [int(x.index) for x in rows]

    def rows(self): return list(self.cases)

    def arrays(self):
        if not self.cases: return None
        rows = list(self.cases)
        return {
            'index': np.asarray([x.index for x in rows], np.int64),
            'origin': np.asarray([x.origin for x in rows], np.int64),
            'keys': np.stack([x.key for x in rows]).astype(np.float32),
            'tod': np.asarray([x.tod for x in rows], np.int16),
            'dow': np.asarray([x.dow for x in rows], np.int8),
        }
