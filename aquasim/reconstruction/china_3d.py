from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from aquasim.core.data.grid_utils import get_starts, recover_lat_lon


SPLITS: Tuple[str, ...] = ("train", "val", "test")


@dataclass(frozen=True)
class ReconstructionPaths:
    china_patch_dir: Path
    unified_dir: Path
    topo_runtime_dir: Path
    checkpoint_path: Path
    norm_stats_path: Path
    out_dir: Path


@dataclass(frozen=True)
class _NormStats:
    mean: np.ndarray
    std: np.ndarray


def _read_variable_names(unified_dir: Path, channels: int) -> List[str]:
    path = unified_dir / "variable_dimension_mapping.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing variable mapping: {path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]
    if "var_name" not in df.columns:
        df = df.rename(columns={df.columns[0]: "var_name"})
    names = [str(v) for v in df["var_name"].tolist()[:channels]]
    if len(names) != int(channels):
        raise RuntimeError(f"variable mapping has {len(names)} names for {channels} channels")
    return names


def _load_norm_stats(path: Path, channels: int) -> _NormStats:
    if not path.exists():
        raise FileNotFoundError(f"missing normalization stats: {path}")
    with np.load(path, allow_pickle=False) as payload:
        mean = payload["mean"].astype(np.float32)
        std = payload["std"].astype(np.float32)
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError(f"normalization stats shape mismatch: mean={mean.shape}, std={std.shape}, channels={channels}")
    if not bool(np.all(np.isfinite(mean))) or not bool(np.all(np.isfinite(std))) or not bool(np.all(std > 0.0)):
        raise ValueError(f"invalid normalization stats: {path}")
    return _NormStats(mean=mean, std=std)


def _load_china_grid(paths: ReconstructionPaths) -> Dict[str, np.ndarray]:
    x_path = paths.china_patch_dir / "dataset_8vars_china.npy"
    m_path = paths.china_patch_dir / "mask_8vars_china.npy"
    info_path = paths.china_patch_dir / "china_region_info.json"
    depth_path = paths.unified_dir / "depth_levels.npy"
    lat_lon_path = paths.unified_dir / "lat_lon_range.json"
    wet_path = paths.topo_runtime_dir / "wet_mask_from_topo.npy"

    for path in [x_path, m_path, info_path, depth_path, lat_lon_path, wet_path]:
        if not path.exists():
            raise FileNotFoundError(f"missing reconstruction dependency: {path}")

    original = np.load(x_path, allow_pickle=False).astype(np.float32)
    original_mask = np.load(m_path, allow_pickle=False).astype(bool)
    if original.shape != original_mask.shape:
        raise ValueError(f"original/mask shape mismatch: {original.shape} vs {original_mask.shape}")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    idx = info["index_range_global"]
    lat0, lat1 = int(idx["lat_start"]), int(idx["lat_end"]) + 1
    lon0, lon1 = int(idx["lon_start"]), int(idx["lon_end"]) + 1

    wet_global = np.load(wet_path, allow_pickle=False).astype(bool)
    wet = wet_global[:, lat0:lat1, lon0:lon1]
    if wet.shape != original.shape[1:]:
        raise ValueError(f"wet mask shape {wet.shape} incompatible with China grid {original.shape[1:]}")

    depth = np.load(depth_path, allow_pickle=False).astype(np.float32)
    lat_lon_meta = json.loads(lat_lon_path.read_text(encoding="utf-8"))
    lat_global, lon_global = recover_lat_lon(lat_lon_meta, h=wet_global.shape[1], w=wet_global.shape[2])
    lat = lat_global[lat0:lat1].astype(np.float32)
    lon = lon_global[lon0:lon1].astype(np.float32)

    return {
        "original": original,
        "original_mask": original_mask,
        "wet_mask": wet,
        "depth": depth,
        "lat": lat,
        "lon": lon,
    }


def _load_patch_bounds(china_patch_dir: Path) -> Dict[int, Tuple[int, int, int, int, int, int]]:
    path = china_patch_dir / "patch_metadata.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing patch metadata: {path}")
    df = pd.read_csv(path)
    required = {"patch_id", "d0", "d1", "h0", "h1", "w0", "w1"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{path} missing columns: {sorted(required)}")
    bounds: Dict[int, Tuple[int, int, int, int, int, int]] = {}
    for _, row in df.iterrows():
        bounds[int(row["patch_id"])] = (
            int(row["d0"]),
            int(row["d1"]),
            int(row["h0"]),
            int(row["h1"]),
            int(row["w0"]),
            int(row["w1"]),
        )
    return bounds


def _load_patch_protocol(china_patch_dir: Path) -> Dict[str, int]:
    path = china_patch_dir / "china_strict_protocol.json"
    if not path.exists():
        raise FileNotFoundError(f"missing China patch protocol: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    patch = payload.get("patch", {})
    stride = payload.get("stride", {})
    return {
        "patch_d": int(patch.get("d", 24)),
        "patch_h": int(patch.get("h", 12)),
        "patch_w": int(patch.get("w", 12)),
        "stride_d": int(stride.get("d", 12)),
        "stride_h": int(stride.get("h", 1)),
        "stride_w": int(stride.get("w", 1)),
    }


def _load_model(
    checkpoint_path: Path,
    model_name: str,
    in_channels: int,
    out_channels: int,
    device: "torch.device",
) -> "torch.nn.Module":
    import torch

    from aquasim.models.backbone3d import build_marine_model

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint_path}")
    model = build_marine_model(
        model_name=model_name,
        in_channels=int(in_channels),
        out_channels=int(out_channels),
        base_channels=32,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _iter_splits(requested: Iterable[str]) -> List[str]:
    splits = [str(s).strip().lower() for s in requested if str(s).strip()]
    invalid = sorted(set(splits) - set(SPLITS))
    if invalid:
        raise ValueError(f"invalid splits {invalid}; expected subset of {SPLITS}")
    return splits or list(SPLITS)


def _accumulate_prediction(
    pred_sum: np.ndarray,
    pred_count: np.ndarray,
    pred_patch: np.ndarray,
    bounds: Tuple[int, int, int, int, int, int],
) -> None:
    d0, d1, h0, h1, w0, w1 = bounds
    d_len, h_len, w_len = d1 - d0, h1 - h0, w1 - w0
    pred_sum[:, d0:d1, h0:h1, w0:w1] += pred_patch[:, :d_len, :h_len, :w_len].astype(np.float64)
    pred_count[d0:d1, h0:h1, w0:w1] += np.uint16(1)


def _predict_batch(
    model: "torch.nn.Module",
    device: "torch.device",
    inputs: List[np.ndarray],
    visible_masks: List[np.ndarray],
    stats: _NormStats,
) -> np.ndarray:
    import torch

    inp = torch.from_numpy(np.stack(inputs).astype(np.float32)).to(device=device)
    visible = torch.from_numpy(np.stack(visible_masks).astype(np.float32)).to(device=device)
    pred_norm = model(inp, visible)
    pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6)
    pred = pred_norm.detach().cpu().numpy().astype(np.float32)
    return pred * stats.std.reshape(1, -1, 1, 1, 1) + stats.mean.reshape(1, -1, 1, 1, 1)


def _run_full_grid_prediction(
    model: "torch.nn.Module",
    device: "torch.device",
    original: np.ndarray,
    original_mask: np.ndarray,
    wet_mask: np.ndarray,
    stats: _NormStats,
    pred_sum: np.ndarray,
    pred_count: np.ndarray,
    batch_size: int,
    include_tail: bool,
    china_patch_dir: Path,
) -> Dict[str, int]:
    protocol = _load_patch_protocol(china_patch_dir)
    _, depth_len, lat_len, lon_len = original.shape
    d_starts = get_starts(depth_len, protocol["patch_d"], protocol["stride_d"], include_tail=include_tail)
    h_starts = get_starts(lat_len, protocol["patch_h"], protocol["stride_h"], include_tail=include_tail)
    w_starts = get_starts(lon_len, protocol["patch_w"], protocol["stride_w"], include_tail=include_tail)

    inputs: List[np.ndarray] = []
    visibles: List[np.ndarray] = []
    bounds_batch: List[Tuple[int, int, int, int, int, int]] = []
    total_windows = 0
    skipped_dry_windows = 0

    def flush() -> None:
        if not inputs:
            return
        pred = _predict_batch(model=model, device=device, inputs=inputs, visible_masks=visibles, stats=stats)
        for bi, bounds in enumerate(bounds_batch):
            _accumulate_prediction(pred_sum=pred_sum, pred_count=pred_count, pred_patch=pred[bi], bounds=bounds)
        inputs.clear()
        visibles.clear()
        bounds_batch.clear()

    for d0 in d_starts:
        d1 = min(d0 + protocol["patch_d"], depth_len)
        for h0 in h_starts:
            h1 = min(h0 + protocol["patch_h"], lat_len)
            for w0 in w_starts:
                w1 = min(w0 + protocol["patch_w"], lon_len)
                total_windows += 1
                if not bool(np.any(wet_mask[d0:d1, h0:h1, w0:w1])):
                    skipped_dry_windows += 1
                    continue

                patch = original[:, d0:d1, h0:h1, w0:w1]
                visible = original_mask[:, d0:d1, h0:h1, w0:w1]
                norm = (patch - stats.mean.reshape(-1, 1, 1, 1)) / stats.std.reshape(-1, 1, 1, 1)
                norm = np.where(visible, norm, 0.0)
                norm = np.nan_to_num(norm, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)

                inputs.append(norm)
                visibles.append(visible.astype(np.float32))
                bounds_batch.append((d0, d1, h0, h1, w0, w1))
                if len(inputs) >= int(batch_size):
                    flush()
    flush()
    return {
        "total_windows": int(total_windows),
        "skipped_dry_windows": int(skipped_dry_windows),
        "predicted_windows": int(total_windows - skipped_dry_windows),
    }


def _run_patch_split_prediction(
    model: "torch.nn.Module",
    device: "torch.device",
    paths: ReconstructionPaths,
    stats: _NormStats,
    pred_sum: np.ndarray,
    pred_count: np.ndarray,
    splits: Iterable[str],
    batch_size: int,
    num_workers: int,
) -> Dict[str, int]:
    import torch
    from torch.utils.data import DataLoader

    from aquasim.data.dataset import Step6PatchDataset, apply_channel_norm, load_patch_split

    patch_bounds = _load_patch_bounds(paths.china_patch_dir)
    requested_splits = _iter_splits(splits)
    loader_kwargs = {"batch_size": int(batch_size), "shuffle": False, "num_workers": int(num_workers)}
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True

    predicted_patches = 0
    for split_name in requested_splits:
        split_raw = load_patch_split(paths.china_patch_dir, split_name)
        split_norm = apply_channel_norm(split_raw, stats)
        dataset = Step6PatchDataset(split_norm, topo_runtime_assets=None, return_topo_runtime=False)
        loader = DataLoader(dataset, **loader_kwargs)
        for batch in loader:
            inp = batch["input"].to(device=device, dtype=torch.float32)
            visible = batch["visible_mask"].to(device=device, dtype=torch.float32)
            pred_norm = model(inp, visible)
            pred_norm = torch.nan_to_num(pred_norm, nan=0.0, posinf=1e6, neginf=-1e6)
            pred = pred_norm.detach().cpu().numpy().astype(np.float32)
            pred = pred * stats.std.reshape(1, -1, 1, 1, 1) + stats.mean.reshape(1, -1, 1, 1, 1)
            patch_ids = batch["patch_id"].detach().cpu().numpy().astype(np.int64)

            for bi, patch_id in enumerate(patch_ids):
                if int(patch_id) not in patch_bounds:
                    raise KeyError(f"patch_id={int(patch_id)} missing from patch metadata")
                _accumulate_prediction(
                    pred_sum=pred_sum,
                    pred_count=pred_count,
                    pred_patch=pred[bi],
                    bounds=patch_bounds[int(patch_id)],
                )
                predicted_patches += 1

    return {
        "requested_splits": len(requested_splits),
        "predicted_windows": int(predicted_patches),
    }


def run_china_3d_reconstruction(
    paths: ReconstructionPaths,
    model_name: str = "marine3d_transformer",
    batch_size: int = 8,
    num_workers: int = 0,
    splits: Iterable[str] = SPLITS,
    source_mode: str = "full_grid",
    include_tail: bool = True,
    force_cpu: bool = False,
) -> Path:
    import torch

    grid = _load_china_grid(paths)
    original = grid["original"]
    original_mask = grid["original_mask"]
    wet_mask = grid["wet_mask"]
    channels = int(original.shape[0])

    stats = _load_norm_stats(paths.norm_stats_path, channels=channels)
    variable_names = _read_variable_names(paths.unified_dir, channels=channels)

    device = torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")
    model = _load_model(
        checkpoint_path=paths.checkpoint_path,
        model_name=model_name,
        in_channels=channels * 2,
        out_channels=channels,
        device=device,
    )

    pred_sum = np.zeros_like(original, dtype=np.float64)
    pred_count = np.zeros(original.shape[1:], dtype=np.uint16)

    with torch.no_grad():
        mode = str(source_mode).strip().lower()
        if mode == "full_grid":
            prediction_summary = _run_full_grid_prediction(
                model=model,
                device=device,
                original=original,
                original_mask=original_mask,
                wet_mask=wet_mask,
                stats=stats,
                pred_sum=pred_sum,
                pred_count=pred_count,
                batch_size=batch_size,
                include_tail=include_tail,
                china_patch_dir=paths.china_patch_dir,
            )
            requested_splits = []
        elif mode == "patch_splits":
            requested_splits = _iter_splits(splits)
            prediction_summary = _run_patch_split_prediction(
                model=model,
                device=device,
                paths=paths,
                stats=stats,
                pred_sum=pred_sum,
                pred_count=pred_count,
                splits=requested_splits,
                batch_size=batch_size,
                num_workers=num_workers,
            )
        else:
            raise ValueError("source_mode must be either 'full_grid' or 'patch_splits'")

    model_prediction = np.full_like(original, np.nan, dtype=np.float32)
    covered = pred_count > 0
    model_prediction[:, covered] = (pred_sum[:, covered] / pred_count[covered].astype(np.float64)).astype(np.float32)

    wet4 = wet_mask[None, :, :, :]
    model_prediction = np.where(wet4 & covered[None, :, :, :], model_prediction, np.nan).astype(np.float32)
    reconstructed = np.where(original_mask, original, model_prediction).astype(np.float32)
    reconstructed = np.where(wet4, reconstructed, np.nan).astype(np.float32)

    paths.out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = paths.out_dir / "china_3d_reconstruction.npz"
    np.savez_compressed(
        out_npz,
        reconstructed_field=reconstructed,
        model_prediction=model_prediction,
        original_data=original,
        original_mask=original_mask.astype(np.uint8),
        wet_mask=wet_mask.astype(np.uint8),
        prediction_count=pred_count,
        depth=grid["depth"],
        lat=grid["lat"],
        lon=grid["lon"],
        variable_names=np.asarray(variable_names),
    )

    manifest = {
        "artifact": str(out_npz),
        "model_name": model_name,
        "checkpoint_path": str(paths.checkpoint_path),
        "normalization_stats": str(paths.norm_stats_path),
        "china_patch_dir": str(paths.china_patch_dir),
        "unified_dir": str(paths.unified_dir),
        "topo_runtime_dir": str(paths.topo_runtime_dir),
        "source_mode": mode,
        "splits": requested_splits,
        "include_tail": bool(include_tail),
        "prediction_summary": prediction_summary,
        "field_shape_c_d_h_w": [int(v) for v in reconstructed.shape],
        "variable_names": variable_names,
        "fusion_rule": "observed values are retained where original_mask=1; model predictions fill original missing cells",
        "ocean_mask_rule": "cells outside wet_mask_from_topo are set to NaN",
        "covered_wet_cells": int(np.sum(covered & wet_mask)),
        "total_wet_cells": int(np.sum(wet_mask)),
        "observed_wet_values": int(np.sum(original_mask & wet4)),
        "missing_wet_values_filled": int(np.sum((~original_mask) & np.isfinite(model_prediction) & wet4)),
    }
    (paths.out_dir / "china_3d_reconstruction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_npz
