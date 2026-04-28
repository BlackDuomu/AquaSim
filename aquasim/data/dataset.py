#!/usr/bin/env python
"""Patch dataset utilities.

Loads patch files and provides Dataset/DataLoader helpers.
Supports optional topo runtime tensors for formal topo-constraint training.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


REQUIRED_FIELDS = [
    "input",
    "target",
    "real_mask",
    "artificial_mask",
    "visible_mask",
    "supervise_mask",
    "patch_id",
]


@dataclass
class PatchSplit:
    split: str
    input: np.ndarray
    target: np.ndarray
    real_mask: np.ndarray
    artificial_mask: np.ndarray
    visible_mask: np.ndarray
    supervise_mask: np.ndarray
    patch_id: np.ndarray


@dataclass
class ChannelNormStats:
    mean: np.ndarray  # [C]
    std: np.ndarray   # [C]


class Step6PatchDataset(Dataset):
    """Minimal dataset wrapper over patch patch arrays.

    Returns dict with:
    - input: [C,D,H,W] float32 (already masked visible input from patch)
    - target: [C,D,H,W] float32
    - visible_mask: [C,D,H,W] float32
    - supervise_mask: [C,D,H,W] float32
    - patch_id: int
    """

    def __init__(
        self,
        patch_split: PatchSplit,
        topo_runtime_assets: Optional["TopoRuntimeAssets"] = None,
        return_topo_runtime: bool = True,
    ):
        self.data = patch_split
        self.n = int(self.data.input.shape[0])
        self.topo_runtime_assets = topo_runtime_assets
        self.return_topo_runtime = bool(return_topo_runtime)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        out = {
            "input": self.data.input[idx],
            "target": self.data.target[idx],
            "visible_mask": self.data.visible_mask[idx].astype(np.float32, copy=False),
            "supervise_mask": self.data.supervise_mask[idx].astype(np.float32, copy=False),
            "patch_id": np.int64(self.data.patch_id[idx]),
        }
        if self.return_topo_runtime and self.topo_runtime_assets is not None:
            patch_id = int(self.data.patch_id[idx])
            patch_shape = tuple(int(v) for v in self.data.input[idx].shape[-3:])
            topo = self.topo_runtime_assets.extract_patch_runtime(patch_id=patch_id, patch_shape=patch_shape)
            out.update(topo)
        return out


def _first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _safe_load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def _slice_1d(arr: np.ndarray, start: int, end: int, out_len: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((out_len,), fill_value=np.float32(fill), dtype=np.float32)
    s = max(0, int(start))
    e = min(int(end), int(arr.shape[0]))
    if e > s:
        seg = arr[s:e].astype(np.float32)
        out[: min(out_len, seg.shape[0])] = seg[:out_len]
    return out


def _slice_2d(arr: np.ndarray, h0: int, h1: int, w0: int, w1: int, out_hw: Tuple[int, int], fill: float = 0.0) -> np.ndarray:
    oh, ow = out_hw
    out = np.full((oh, ow), fill_value=np.float32(fill), dtype=np.float32)
    hs = max(0, int(h0))
    he = min(int(h1), int(arr.shape[0]))
    ws = max(0, int(w0))
    we = min(int(w1), int(arr.shape[1]))
    if he > hs and we > ws:
        seg = arr[hs:he, ws:we].astype(np.float32)
        out[: min(oh, seg.shape[0]), : min(ow, seg.shape[1])] = seg[:oh, :ow]
    return out


def _slice_3d(
    arr: np.ndarray,
    d0: int,
    d1: int,
    h0: int,
    h1: int,
    w0: int,
    w1: int,
    out_dhw: Tuple[int, int, int],
    fill: float = 0.0,
) -> np.ndarray:
    od, oh, ow = out_dhw
    out = np.full((od, oh, ow), fill_value=np.float32(fill), dtype=np.float32)
    ds = max(0, int(d0))
    de = min(int(d1), int(arr.shape[0]))
    hs = max(0, int(h0))
    he = min(int(h1), int(arr.shape[1]))
    ws = max(0, int(w0))
    we = min(int(w1), int(arr.shape[2]))
    if de > ds and he > hs and we > ws:
        seg = arr[ds:de, hs:he, ws:we].astype(np.float32)
        out[: min(od, seg.shape[0]), : min(oh, seg.shape[1]), : min(ow, seg.shape[2])] = seg[:od, :oh, :ow]
    return out


def _to_steepness_weight(slope: np.ndarray, ocean_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    s = np.asarray(slope, dtype=np.float32)
    ocean = np.asarray(ocean_mask).astype(bool)
    if bool(ocean.any()):
        q95 = float(np.percentile(s[ocean], 95.0))
    else:
        q95 = float(np.percentile(s, 95.0))
    scale = max(float(q95), float(eps))
    steepness = np.clip(s / scale, 0.0, 1.0)
    steepness = np.where(ocean, steepness, 0.0).astype(np.float32)
    return steepness


def _to_roughness_weight(roughness: np.ndarray, ocean_mask: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = np.asarray(roughness, dtype=np.float32)
    ocean = np.asarray(ocean_mask).astype(bool)
    if bool(ocean.any()):
        q95 = float(np.percentile(r[ocean], 95.0))
    else:
        q95 = float(np.percentile(r, 95.0))
    scale = max(float(q95), float(eps))
    x = np.clip(r / scale, 0.0, 1.0)
    x = np.where(ocean, x, 0.0).astype(np.float32)
    return x


@dataclass
class TopoRuntimeAssets:
    depth_levels_m: np.ndarray
    wet_mask: np.ndarray
    near_bottom_mask: np.ndarray
    slope_steepness: np.ndarray
    roughness_weight: Optional[np.ndarray]
    topo_class: Optional[np.ndarray]
    patch_bounds: Dict[int, Tuple[int, int, int, int, int, int]]

    @classmethod
    def from_dirs(cls, topo_dir: Path, data_dir: Path, unified_dir: Path) -> "TopoRuntimeAssets":
        arrays_dir = topo_dir / "arrays"
        wet_path = _first_existing(topo_dir / "wet_mask_from_topo.npy", arrays_dir / "wet_mask_from_topo.npy")
        near_path = _first_existing(topo_dir / "near_bottom_mask.npy", arrays_dir / "near_bottom_mask_200m.npy")
        slope_path = _first_existing(arrays_dir / "slope.npy")
        bottom_depth_path = _first_existing(arrays_dir / "bottom_depth.npy")
        rough_path = _first_existing(arrays_dir / "roughness.npy")
        topo_class_path = _first_existing(arrays_dir / "topo_class.npy")
        patch_meta_path = data_dir / "patch_metadata.csv"
        depth_levels_path = unified_dir / "depth_levels.npy"

        for p, name in [
            (wet_path, "wet_mask_from_topo"),
            (near_path, "near_bottom_mask"),
            (slope_path, "slope"),
            (patch_meta_path, "patch_metadata.csv"),
            (depth_levels_path, "depth_levels.npy"),
        ]:
            if p is None or (isinstance(p, Path) and not p.exists()):
                raise FileNotFoundError(f"Missing topo runtime dependency: {name}")

        wet = _safe_load(wet_path).astype(np.float32)  # type: ignore[arg-type]
        near = _safe_load(near_path).astype(np.float32)  # type: ignore[arg-type]
        slope = _safe_load(slope_path).astype(np.float32)  # type: ignore[arg-type]
        if wet.shape != near.shape:
            raise ValueError(f"wet_mask shape {wet.shape} != near_bottom_mask shape {near.shape}")
        if wet.ndim != 3:
            raise ValueError(f"wet_mask must be [D,H,W], got {wet.shape}")
        if slope.shape != wet.shape[1:]:
            raise ValueError(f"slope shape {slope.shape} incompatible with wet_mask spatial {wet.shape[1:]}")

        if bottom_depth_path is not None and bottom_depth_path.exists():
            bottom_depth = _safe_load(bottom_depth_path).astype(np.float32)
            ocean = bottom_depth > 0.0
        else:
            ocean = slope > 0.0

        slope_steepness = _to_steepness_weight(slope=slope, ocean_mask=ocean)
        roughness_weight: Optional[np.ndarray] = None
        if rough_path is not None and rough_path.exists():
            roughness_weight = _to_roughness_weight(roughness=_safe_load(rough_path).astype(np.float32), ocean_mask=ocean)
        topo_class: Optional[np.ndarray] = None
        if topo_class_path is not None and topo_class_path.exists():
            topo_class = _safe_load(topo_class_path).astype(np.int64)

        patch_df = pd.read_csv(patch_meta_path)
        req = {"patch_id", "d0", "d1", "h0", "h1", "w0", "w1"}
        if not req.issubset(set(patch_df.columns)):
            raise ValueError(f"{patch_meta_path} missing required columns: {sorted(req)}")
        patch_bounds: Dict[int, Tuple[int, int, int, int, int, int]] = {}
        for _, row in patch_df.iterrows():
            patch_bounds[int(row["patch_id"])] = (
                int(row["d0"]),
                int(row["d1"]),
                int(row["h0"]),
                int(row["h1"]),
                int(row["w0"]),
                int(row["w1"]),
            )

        depth_levels = _safe_load(depth_levels_path).astype(np.float32)
        return cls(
            depth_levels_m=depth_levels,
            wet_mask=wet,
            near_bottom_mask=near,
            slope_steepness=slope_steepness,
            roughness_weight=roughness_weight,
            topo_class=topo_class,
            patch_bounds=patch_bounds,
        )

    def extract_patch_runtime(self, patch_id: int, patch_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
        d, h, w = (int(v) for v in patch_shape)
        b = self.patch_bounds.get(int(patch_id))
        if b is None:
            patch_depth = _slice_1d(self.depth_levels_m, 0, d, d, fill=0.0)
            wet = np.ones((d, h, w), dtype=np.float32)
            near = np.zeros((d, h, w), dtype=np.float32)
            slope = np.zeros((d, h, w), dtype=np.float32)
            rough = np.zeros((d, h, w), dtype=np.float32)
            topo_class = np.zeros((h, w), dtype=np.int64)
            return {
                "patch_depth_m": patch_depth,
                "wet_mask": wet,
                "near_bottom_mask": near,
                "slope_weight": slope,
                "roughness_weight": rough,
                "topo_class": topo_class,
            }

        d0, d1, h0, h1, w0, w1 = b
        patch_depth = _slice_1d(self.depth_levels_m, d0, d1, d, fill=0.0)
        wet = _slice_3d(self.wet_mask, d0, d1, h0, h1, w0, w1, (d, h, w), fill=0.0)
        near = _slice_3d(self.near_bottom_mask, d0, d1, h0, h1, w0, w1, (d, h, w), fill=0.0)
        steep2d = _slice_2d(self.slope_steepness, h0, h1, w0, w1, (h, w), fill=0.0)
        slope = np.repeat(steep2d[None, :, :], d, axis=0).astype(np.float32)
        if self.roughness_weight is None:
            rough2d = np.zeros((h, w), dtype=np.float32)
        else:
            rough2d = _slice_2d(self.roughness_weight, h0, h1, w0, w1, (h, w), fill=0.0)
        rough = np.repeat(rough2d[None, :, :], d, axis=0).astype(np.float32)
        if self.topo_class is None:
            topo_class = np.zeros((h, w), dtype=np.int64)
        else:
            topo_class = _slice_2d(self.topo_class.astype(np.float32), h0, h1, w0, w1, (h, w), fill=0.0).astype(np.int64)
        return {
            "patch_depth_m": patch_depth.astype(np.float32),
            "wet_mask": wet.astype(np.float32),
            "near_bottom_mask": near.astype(np.float32),
            "slope_weight": slope.astype(np.float32),
            "roughness_weight": rough.astype(np.float32),
            "topo_class": topo_class.astype(np.int64),
        }


def load_topo_runtime_assets(
    topo_dir: Optional[Path],
    data_dir: Path,
    unified_dir: Path,
    required: bool = False,
) -> Optional[TopoRuntimeAssets]:
    if topo_dir is None:
        if required:
            raise FileNotFoundError("topo_dir is required but not provided")
        return None
    try:
        return TopoRuntimeAssets.from_dirs(topo_dir=topo_dir, data_dir=data_dir, unified_dir=unified_dir)
    except Exception:
        if required:
            raise
        return None


def _assert_npz_schema(path: Path, arrs: Dict[str, np.ndarray]) -> None:
    for k in REQUIRED_FIELDS:
        if k not in arrs:
            raise KeyError(f"{path} missing field `{k}`")

    n = int(arrs["input"].shape[0])
    for k in ["target", "real_mask", "artificial_mask", "visible_mask", "supervise_mask"]:
        if tuple(arrs[k].shape) != tuple(arrs["input"].shape):
            raise ValueError(f"{path} field `{k}` shape mismatch: {arrs[k].shape} vs {arrs['input'].shape}")

    if tuple(arrs["patch_id"].shape) != (n,):
        raise ValueError(f"{path} patch_id shape mismatch: {arrs['patch_id'].shape} vs ({n},)")


def load_patch_split(patch_dir: Path, split: str) -> PatchSplit:
    path = patch_dir / "patch_dataset" / f"{split}_patches.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing patch split file: {path}")

    with np.load(path, allow_pickle=False) as raw:
        arrs = {k: raw[k] for k in raw.files}
    _assert_npz_schema(path, arrs)

    return PatchSplit(
        split=split,
        input=arrs["input"].astype(np.float32, copy=False),
        target=arrs["target"].astype(np.float32, copy=False),
        real_mask=arrs["real_mask"].astype(np.uint8, copy=False),
        artificial_mask=arrs["artificial_mask"].astype(np.uint8, copy=False),
        visible_mask=arrs["visible_mask"].astype(np.uint8, copy=False),
        supervise_mask=arrs["supervise_mask"].astype(np.uint8, copy=False),
        patch_id=arrs["patch_id"].astype(np.int64, copy=False),
    )


def load_all_splits(patch_dir: Path) -> Tuple[PatchSplit, PatchSplit, PatchSplit]:
    train = load_patch_split(patch_dir, "train")
    val = load_patch_split(patch_dir, "val")
    test = load_patch_split(patch_dir, "test")
    return train, val, test


def compute_channel_norm_stats(train_split: PatchSplit, eps: float = 1e-6) -> ChannelNormStats:
    """Compute per-channel mean/std from train target on real observed points only."""
    target = train_split.target  # [N,C,D,H,W]
    real = train_split.real_mask

    c = target.shape[1]
    mean = np.zeros((c,), dtype=np.float32)
    std = np.ones((c,), dtype=np.float32)

    for ci in range(c):
        values = target[:, ci]
        mask = (real[:, ci] > 0) & np.isfinite(values)
        count = int(mask.sum())
        if count == 0:
            mean[ci] = 0.0
            std[ci] = 1.0
            continue
        sum_v = float(np.sum(values, where=mask, dtype=np.float64))
        sum_sq = float(np.sum(values * values, where=mask, dtype=np.float64))
        m = sum_v / float(count)
        var = max(0.0, sum_sq / float(count) - m * m)
        mean[ci] = float(m)
        s = float(np.sqrt(var))
        std[ci] = s if s > eps else 1.0

    return ChannelNormStats(mean=mean, std=std)


def default_channel_norm_cache_path(patch_dir: Path) -> Path:
    return patch_dir / "patch_dataset" / "channel_norm_stats.npz"


def _channel_norm_cache_metadata(patch_dir: Path, train_split: PatchSplit) -> Dict[str, object]:
    train_path = patch_dir / "patch_dataset" / "train_patches.npz"
    st = train_path.stat()
    return {
        "train_file_size": int(st.st_size),
        "train_file_mtime_ns": int(st.st_mtime_ns),
        "target_shape": tuple(int(v) for v in train_split.target.shape),
    }


def _load_channel_norm_stats_cache(
    cache_path: Path,
    patch_dir: Path,
    train_split: PatchSplit,
) -> Optional[ChannelNormStats]:
    if not cache_path.exists():
        return None
    expected = _channel_norm_cache_metadata(patch_dir=patch_dir, train_split=train_split)
    try:
        with np.load(cache_path, allow_pickle=False) as payload:
            mean = payload["mean"].astype(np.float32)
            std = payload["std"].astype(np.float32)
            cached_size = int(payload["train_file_size"])
            cached_mtime_ns = int(payload["train_file_mtime_ns"])
            cached_shape = tuple(int(v) for v in payload["target_shape"].tolist())
    except Exception:
        return None

    channels = int(train_split.target.shape[1])
    valid = (
        mean.shape == (channels,)
        and std.shape == (channels,)
        and bool(np.all(np.isfinite(mean)))
        and bool(np.all(np.isfinite(std)))
        and bool(np.all(std > 0.0))
        and cached_size == expected["train_file_size"]
        and cached_mtime_ns == expected["train_file_mtime_ns"]
        and cached_shape == expected["target_shape"]
    )
    if not valid:
        return None
    return ChannelNormStats(mean=mean, std=std)


def _save_channel_norm_stats_cache(
    cache_path: Path,
    patch_dir: Path,
    train_split: PatchSplit,
    stats: ChannelNormStats,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta = _channel_norm_cache_metadata(patch_dir=patch_dir, train_split=train_split)
    tmp_path = cache_path.with_name(f"{cache_path.name}.tmp.{os.getpid()}.npz")
    np.savez(
        tmp_path,
        mean=stats.mean.astype(np.float32),
        std=stats.std.astype(np.float32),
        train_file_size=np.array(meta["train_file_size"], dtype=np.int64),
        train_file_mtime_ns=np.array(meta["train_file_mtime_ns"], dtype=np.int64),
        target_shape=np.array(meta["target_shape"], dtype=np.int64),
    )
    os.replace(tmp_path, cache_path)


def load_or_compute_channel_norm_stats(
    patch_dir: Path,
    train_split: PatchSplit,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
) -> ChannelNormStats:
    if not use_cache:
        return compute_channel_norm_stats(train_split)

    resolved_cache_path = cache_path if cache_path is not None else default_channel_norm_cache_path(patch_dir)
    cached = _load_channel_norm_stats_cache(
        cache_path=resolved_cache_path,
        patch_dir=patch_dir,
        train_split=train_split,
    )
    if cached is not None:
        print(f"[NORM-CACHE] hit: {resolved_cache_path}")
        return cached

    print(f"[NORM-CACHE] miss: {resolved_cache_path}")
    stats = compute_channel_norm_stats(train_split)
    try:
        _save_channel_norm_stats_cache(
            cache_path=resolved_cache_path,
            patch_dir=patch_dir,
            train_split=train_split,
            stats=stats,
        )
        print(f"[NORM-CACHE] saved: {resolved_cache_path}")
    except Exception as exc:
        print(f"[NORM-CACHE] save skipped: {resolved_cache_path} ({exc})")
    return stats


def apply_channel_norm(split: PatchSplit, stats: ChannelNormStats) -> PatchSplit:
    """Normalize input/target with channel stats and keep hidden positions in input as 0."""
    mean = stats.mean.reshape(1, -1, 1, 1, 1)
    std = stats.std.reshape(1, -1, 1, 1, 1)

    target_n = split.target.astype(np.float32, copy=False)
    input_n = split.input.astype(np.float32, copy=False)
    np.subtract(target_n, mean, out=target_n)
    np.divide(target_n, std, out=target_n)
    np.subtract(input_n, mean, out=input_n)
    np.divide(input_n, std, out=input_n)
    # Keep invisible points as 0 in normalized input.
    np.multiply(input_n, split.visible_mask > 0, out=input_n)

    return PatchSplit(
        split=split.split,
        input=input_n,
        target=target_n,
        real_mask=split.real_mask,
        artificial_mask=split.artificial_mask,
        visible_mask=split.visible_mask,
        supervise_mask=split.supervise_mask,
        patch_id=split.patch_id,
    )


def denormalize_tensor(x: np.ndarray, stats: ChannelNormStats) -> np.ndarray:
    mean = stats.mean.reshape(1, -1, 1, 1, 1)
    std = stats.std.reshape(1, -1, 1, 1, 1)
    return x * std + mean



