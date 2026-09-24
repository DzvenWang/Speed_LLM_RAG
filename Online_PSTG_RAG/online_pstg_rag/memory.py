from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class MemoryCase:
    index: int
    origin: int
    key: np.ndarray
    residual: np.ndarray
    valid: np.ndarray
    tod: int
    dow: int


class RecentResidualMemory:
    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self.cases = deque(maxlen=self.capacity)

    def __len__(self): return len(self.cases)

    def add(self, case: MemoryCase):
        self.cases.append(case)

    def arrays(self):
        if not self.cases:
            return None
        rows = list(self.cases)
        return {
            "index": np.asarray([x.index for x in rows], np.int64),
            "origin": np.asarray([x.origin for x in rows], np.int64),
            "keys": np.stack([x.key for x in rows]).astype(np.float32),
            "residual": np.stack([x.residual for x in rows]).astype(np.float32),
            "valid": np.stack([x.valid for x in rows]).astype(bool),
            "tod": np.asarray([x.tod for x in rows], np.int16),
            "dow": np.asarray([x.dow for x in rows], np.int8),
        }
