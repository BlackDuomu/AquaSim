#!/usr/bin/env python
"""Step6 utility functions."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj: Dict[str, object]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics_from_numpy(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {"count": 0, "rmse": np.nan, "mae": np.nan, "r2": np.nan, "mape": np.nan}

    y_true = y_true.astype(np.float64)
    y_pred = y_pred.astype(np.float64)

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))

    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 1e-12 else np.nan

    mape = float(np.mean(np.abs((y_pred - y_true) / np.maximum(np.abs(y_true), 1e-6))) * 100.0)

    return {"count": int(y_true.size), "rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

