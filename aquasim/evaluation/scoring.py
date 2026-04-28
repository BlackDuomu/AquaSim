from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping


def _safe_float(v: object, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _inv_score(x: float) -> float:
    if not math.isfinite(x):
        return 0.0
    return float(max(0.0, 1.0 / (1.0 + max(0.0, x))))


def compute_topo_score_from_metrics(
    metrics: Mapping[str, object],
    near_weight: float = 0.35,
    slope_weight: float = 0.35,
    consistency_weight: float = 0.30,
) -> float:
    near_rmse = _safe_float(metrics.get("near_bottom_rmse"), float("nan"))
    slope_rmse = _safe_float(metrics.get("slope_area_rmse"), float("nan"))
    consistency = _safe_float(metrics.get("topo_consistency_score"), float("nan"))

    near_score = _inv_score(near_rmse)
    slope_score = _inv_score(slope_rmse)
    consistency_score = float(consistency) if math.isfinite(consistency) else 0.0
    consistency_score = float(min(1.0, max(0.0, consistency_score)))

    wsum = max(1e-8, float(near_weight + slope_weight + consistency_weight))
    topo_score = (
        float(near_weight) * near_score
        + float(slope_weight) * slope_score
        + float(consistency_weight) * consistency_score
    ) / wsum
    return float(max(0.0, topo_score))


def compute_final_score_bundle(
    test_rmse: float,
    physics_score: float = 0.0,
    topo_score: float = 0.0,
) -> Dict[str, float]:
    return {
        "test_rmse": float(test_rmse),
        "physics_score": float(physics_score),
        "topo_score": float(topo_score),
        "final_score": float(max(0.0, 1.0 / (1.0 + float(test_rmse))) + physics_score + topo_score),
    }


def save_score_bundle(path: Path, bundle: Dict[str, float]) -> None:
    path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
