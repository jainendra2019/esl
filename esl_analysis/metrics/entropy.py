from __future__ import annotations

import numpy as np

def belief_entropy(belief: np.ndarray) -> float:
    b = np.asarray(belief, dtype=np.float64)
    b = np.clip(b, 0.0, None)
    b = b / b.sum()
    positive = b[b > 0.0]
    return float(-np.sum(positive * np.log(positive)))
