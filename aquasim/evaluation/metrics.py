from __future__ import annotations

import numpy as np


def rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    if mask is not None:
        diff = diff[np.asarray(mask).astype(bool)]
    return float(np.sqrt(np.mean(diff * diff)))


def mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))
    if mask is not None:
        diff = diff[np.asarray(mask).astype(bool)]
    return float(np.mean(diff))
