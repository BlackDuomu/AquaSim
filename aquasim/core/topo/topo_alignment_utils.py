from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class RegularGridTopo:
    lon: np.ndarray
    lat: np.ndarray
    elevation_m: np.ndarray
    raw_meta: Dict[str, Any]


def _normalize_lon_to_minus180_180(lon: np.ndarray) -> np.ndarray:
    out = ((lon + 180.0) % 360.0) - 180.0
    out = np.where(np.isclose(out, -180.0), 180.0, out)
    return out


def canonicalize_target_lon(lon: np.ndarray, src_lon_min: float, src_lon_max: float) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    if src_lon_min >= 0.0 and src_lon_max > 180.0:
        return np.mod(lon, 360.0)
    return _normalize_lon_to_minus180_180(lon)


def _axis_from_range(vmin: float, vmax: float, count: int) -> np.ndarray:
    return np.linspace(float(vmin), float(vmax), int(count), dtype=np.float64)


def load_etopo1_grd(path: Path) -> RegularGridTopo:
    from netCDF4 import Dataset

    with Dataset(str(path), mode="r") as ds:
        x_range = ds.variables["x_range"][:]
        y_range = ds.variables["y_range"][:]
        dims = ds.variables["dimension"][:]
        spacing = ds.variables["spacing"][:]
        z = ds.variables["z"][:]
        fill = getattr(ds.variables["z"], "_FillValue", None)

    nx = int(dims[0])
    ny = int(dims[1])
    lon = _axis_from_range(float(x_range[0]), float(x_range[1]), nx)
    lat = _axis_from_range(float(y_range[0]), float(y_range[1]), ny)

    elev = np.asarray(z, dtype=np.float32).reshape((ny, nx))
    if fill is not None:
        elev = np.where(elev == float(fill), np.nan, elev)

    # The ETOPO1 GDAL/GMT grid used by this project stores rows north-to-south
    # even though y_range is reported as ascending. Flip the data so row order
    # matches the ascending latitude axis constructed from y_range.
    elev = elev[::-1, :]

    # Ensure ascending axes; most interpolation assumes monotonic ascending grids.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        elev = elev[::-1, :]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        elev = elev[:, ::-1]

    raw_meta: Dict[str, Any] = {
        "x_range": [float(x_range[0]), float(x_range[1])],
        "y_range": [float(y_range[0]), float(y_range[1])],
        "spacing": [float(spacing[0]), float(spacing[1])],
        "dimension": [nx, ny],
    }
    return RegularGridTopo(lon=lon, lat=lat, elevation_m=elev, raw_meta=raw_meta)


def bilinear_resample_to_grid(
    src_lon: np.ndarray,
    src_lat: np.ndarray,
    src_field: np.ndarray,
    target_lon: np.ndarray,
    target_lat: np.ndarray,
) -> np.ndarray:
    src_lon = np.asarray(src_lon, dtype=np.float64)
    src_lat = np.asarray(src_lat, dtype=np.float64)
    src_field = np.asarray(src_field, dtype=np.float32)
    target_lon = np.asarray(target_lon, dtype=np.float64)
    target_lat = np.asarray(target_lat, dtype=np.float64)

    if src_lat.ndim != 1 or src_lon.ndim != 1:
        raise ValueError("src_lat/src_lon must be 1D arrays")
    if src_field.shape != (src_lat.size, src_lon.size):
        raise ValueError("src_field shape must equal (len(src_lat), len(src_lon))")
    if target_lat.ndim != 1 or target_lon.ndim != 1:
        raise ValueError("target_lat/target_lon must be 1D arrays")

    lon_t = canonicalize_target_lon(
        lon=target_lon, src_lon_min=float(src_lon.min()), src_lon_max=float(src_lon.max())
    )
    lat_t = np.clip(target_lat, float(src_lat[0]), float(src_lat[-1]))
    lon_t = np.clip(lon_t, float(src_lon[0]), float(src_lon[-1]))

    dlat = float(src_lat[1] - src_lat[0])
    dlon = float(src_lon[1] - src_lon[0])
    if dlat <= 0.0 or dlon <= 0.0:
        raise ValueError("src grid spacing must be positive (ascending axes required)")

    y = (lat_t - src_lat[0]) / dlat
    x = (lon_t - src_lon[0]) / dlon

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.clip(y0, 0, src_lat.size - 2)
    x0 = np.clip(x0, 0, src_lon.size - 2)
    y1 = y0 + 1
    x1 = x0 + 1

    wy = (y - y0).astype(np.float32)
    wx = (x - x0).astype(np.float32)
    wy = np.clip(wy, 0.0, 1.0)
    wx = np.clip(wx, 0.0, 1.0)

    out = np.empty((target_lat.size, target_lon.size), dtype=np.float32)
    for i in range(target_lat.size):
        iy0 = y0[i]
        iy1 = y1[i]
        fy = wy[i]

        v00 = src_field[iy0, x0]
        v01 = src_field[iy0, x1]
        v10 = src_field[iy1, x0]
        v11 = src_field[iy1, x1]

        top = v00 * (1.0 - wx) + v01 * wx
        bot = v10 * (1.0 - wx) + v11 * wx
        out[i, :] = top * (1.0 - fy) + bot * fy
    return out


def get_training_grid(unified_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    import json

    meta = json.loads((unified_dir / "lat_lon_range.json").read_text(encoding="utf-8"))
    lat = np.linspace(float(meta["lat_min"]), float(meta["lat_max"]), int(meta["lat_count"]), dtype=np.float32)
    lon = np.linspace(float(meta["lon_min"]), float(meta["lon_max"]), int(meta["lon_count"]), dtype=np.float32)
    return lat, lon




