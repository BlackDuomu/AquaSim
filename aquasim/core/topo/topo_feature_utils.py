from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


EARTH_METERS_PER_DEGREE = 111_120.0


def bottom_depth_from_elevation(elevation_m: np.ndarray) -> np.ndarray:
    elev = np.asarray(elevation_m, dtype=np.float32)
    return np.where(np.isfinite(elev) & (elev < 0.0), -elev, 0.0).astype(np.float32)


def _rolling_nan_std2d(arr: np.ndarray, valid_mask: np.ndarray, radius: int) -> np.ndarray:
    out = np.zeros_like(arr, dtype=np.float32)
    denom = np.zeros_like(arr, dtype=np.float32)
    s1 = np.zeros_like(arr, dtype=np.float32)
    s2 = np.zeros_like(arr, dtype=np.float32)

    pad = radius
    a = np.pad(arr, ((pad, pad), (pad, pad)), mode="edge")
    m = np.pad(valid_mask.astype(np.float32), ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    a = np.where(m > 0, a, 0.0)

    k = 2 * radius + 1
    for di in range(k):
        for dj in range(k):
            w = m[di : di + arr.shape[0], dj : dj + arr.shape[1]]
            v = a[di : di + arr.shape[0], dj : dj + arr.shape[1]]
            denom += w
            s1 += v
            s2 += v * v

    good = denom > 0.0
    mean = np.zeros_like(arr, dtype=np.float32)
    mean[good] = s1[good] / denom[good]
    var = np.zeros_like(arr, dtype=np.float32)
    var[good] = s2[good] / denom[good] - mean[good] * mean[good]
    var = np.maximum(var, 0.0)
    out[good] = np.sqrt(var[good], dtype=np.float32)
    return out.astype(np.float32)


def _rolling_nan_range2d(arr: np.ndarray, valid_mask: np.ndarray, radius: int) -> np.ndarray:
    pad = radius
    arr_nan = np.where(valid_mask, arr, np.nan).astype(np.float32)
    a = np.pad(arr_nan, ((pad, pad), (pad, pad)), mode="constant", constant_values=np.nan)
    k = 2 * radius + 1
    max_map = np.full(arr.shape, np.nan, dtype=np.float32)
    min_map = np.full(arr.shape, np.nan, dtype=np.float32)
    for di in range(k):
        for dj in range(k):
            v = a[di : di + arr.shape[0], dj : dj + arr.shape[1]]
            max_map = np.fmax(max_map, v)
            min_map = np.fmin(min_map, v)
    relief = max_map - min_map
    relief = np.where(np.isfinite(relief), relief, 0.0)
    return relief.astype(np.float32)


def slope_from_bottom_depth(bottom_depth_m: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    depth = np.asarray(bottom_depth_m, dtype=np.float32)
    lat = np.asarray(lat, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    dlat = float(np.abs(lat[1] - lat[0]))
    dlon = float(np.abs(lon[1] - lon[0]))

    dy = EARTH_METERS_PER_DEGREE * dlat
    cos_lat = np.cos(np.deg2rad(lat)).astype(np.float64)
    cos_lat = np.maximum(cos_lat, 1e-6)
    dx_per_row = (EARTH_METERS_PER_DEGREE * dlon * cos_lat).astype(np.float32)

    dz_dlat = np.gradient(depth, axis=0).astype(np.float32) / np.float32(max(dy, 1e-6))
    dz_dlon_raw = np.gradient(depth, axis=1).astype(np.float32)
    dz_dlon = dz_dlon_raw / dx_per_row[:, None]
    slope = np.sqrt(dz_dlat * dz_dlat + dz_dlon * dz_dlon, dtype=np.float32)
    slope = np.where(depth > 0.0, slope, 0.0)
    return slope.astype(np.float32)


def compute_topo_derivatives(
    bottom_depth_m: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    roughness_radius: int = 1,
    relief_radius: int = 2,
) -> Dict[str, np.ndarray]:
    depth = np.asarray(bottom_depth_m, dtype=np.float32)
    ocean = depth > 0.0
    slope = slope_from_bottom_depth(depth, lat=lat, lon=lon)
    roughness = _rolling_nan_std2d(depth, valid_mask=ocean, radius=int(roughness_radius))
    relief = _rolling_nan_range2d(depth, valid_mask=ocean, radius=int(relief_radius))

    roughness = np.where(ocean, roughness, 0.0).astype(np.float32)
    relief = np.where(ocean, relief, 0.0).astype(np.float32)

    return {
        "bottom_depth": depth.astype(np.float32),
        "slope": slope.astype(np.float32),
        "roughness": roughness.astype(np.float32),
        "relief": relief.astype(np.float32),
    }


def classify_topo(
    bottom_depth_m: np.ndarray,
    slope: np.ndarray,
    shelf_max_depth_m: float = 200.0,
    basin_min_depth_m: float = 2000.0,
    slope_min_gradient: float = 0.002,
) -> Dict[str, np.ndarray]:
    depth = np.asarray(bottom_depth_m, dtype=np.float32)
    slope = np.asarray(slope, dtype=np.float32)
    ocean = depth > 0.0

    shelf = ocean & (depth <= float(shelf_max_depth_m))
    basin = ocean & (depth >= float(basin_min_depth_m))
    slope_zone = ocean & (~shelf) & (~basin)

    # Tighten slope-zone mask with gradient information but do not erase all mid-depth cells.
    slope_mask = slope_zone & (slope >= float(slope_min_gradient))
    slope_mask = slope_mask | (slope_zone & (~slope_mask))

    topo_class = np.zeros_like(depth, dtype=np.uint8)
    topo_class[shelf] = 1
    topo_class[slope_mask] = 2
    topo_class[basin] = 3

    return {
        "shelf_mask": shelf.astype(np.uint8),
        "slope_mask": slope_mask.astype(np.uint8),
        "basin_mask": basin.astype(np.uint8),
        "topo_class": topo_class,
    }


def build_wet_and_near_bottom_masks(
    depth_levels_m: np.ndarray,
    bottom_depth_m: np.ndarray,
    thresholds_m: Tuple[float, ...] = (100.0, 200.0),
) -> Dict[str, np.ndarray]:
    depth_levels = np.asarray(depth_levels_m, dtype=np.float32)
    bottom = np.asarray(bottom_depth_m, dtype=np.float32)
    ocean2d = bottom > 0.0

    d3 = depth_levels[:, None, None]
    bottom3 = bottom[None, :, :]
    wet = (d3 <= bottom3) & ocean2d[None, :, :]

    out: Dict[str, np.ndarray] = {"wet_mask": wet.astype(np.uint8)}
    for th in thresholds_m:
        near = wet & ((bottom3 - d3) <= float(th))
        out[f"near_bottom_mask_{int(th)}m"] = near.astype(np.uint8)
    return out



