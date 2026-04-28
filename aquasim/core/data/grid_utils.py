from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def recover_lat_lon(lat_lon_meta: Dict[str, float], h: int, w: int) -> Tuple[np.ndarray, np.ndarray]:
    lat = np.linspace(float(lat_lon_meta['lat_min']), float(lat_lon_meta['lat_max']), int(h), dtype=np.float32)
    lon = np.linspace(float(lat_lon_meta['lon_min']), float(lat_lon_meta['lon_max']), int(w), dtype=np.float32)
    return lat, lon


def get_starts(length: int, window: int, stride: int, include_tail: bool) -> list[int]:
    if window <= 0 or stride <= 0:
        raise ValueError('window and stride must be positive')
    if window > length:
        return [0]
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if include_tail and starts[-1] != last:
        starts.append(last)
    return starts


def crop_bbox_indices(lat: np.ndarray, lon: np.ndarray, lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> tuple[np.ndarray, np.ndarray]:
    lat_idx = np.where((lat >= float(lat_min) - 1e-6) & (lat <= float(lat_max) + 1e-6))[0]
    lon_idx = np.where((lon >= float(lon_min) - 1e-6) & (lon <= float(lon_max) + 1e-6))[0]
    if lat_idx.size == 0 or lon_idx.size == 0:
        raise ValueError('bbox does not intersect grid')
    return lat_idx, lon_idx


def intersects(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def is_window_overlap(h0: int, h1: int, w0: int, w1: int, box: Dict[str, int]) -> bool:
    lat_overlap = intersects(h0, h1 - 1, int(box['lat_start']), int(box['lat_end']))
    lon_overlap = intersects(w0, w1 - 1, int(box['lon_start']), int(box['lon_end']))
    return bool(lat_overlap and lon_overlap)


def split_code_to_name(code: int) -> str:
    return {1: 'train', 2: 'val', 3: 'test'}.get(int(code), 'none')



