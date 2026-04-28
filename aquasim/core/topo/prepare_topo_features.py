#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aquasim.utils.common import ensure_dir, parse_simple_yaml, pick_existing, save_json
from aquasim.core.topo.topo_alignment_utils import bilinear_resample_to_grid, get_training_grid, load_etopo1_grd
from aquasim.core.topo.topo_feature_utils import (
    bottom_depth_from_elevation,
    build_wet_and_near_bottom_masks,
    classify_topo,
    compute_topo_derivatives,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare aligned and derived topo products for training/eval/topo-aware downstream usage."
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data_topo_prepare.yaml"),
        help="YAML config path",
    )
    p.add_argument("--topo-file", type=Path, default=None, help="Override raw topo .grd file")
    p.add_argument("--unified-dir", type=Path, default=None, help="Override unified field output directory")
    p.add_argument("--step2-dir", type=Path, default=None, help="Override Step2 output directory")
    p.add_argument("--patch-metadata", type=Path, default=None, help="Override patch metadata CSV")
    p.add_argument("--out-dir", type=Path, default=None, help="Override output directory")
    p.add_argument("--no-figures", action="store_true", help="Disable figure generation")
    return p.parse_args()


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return parse_simple_yaml(path)


def _nested_get(d: Dict[str, Any], keys: Iterable[str], default: Any) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _save_array(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(path, arr, allow_pickle=False)


def _class_label(class_id: int) -> str:
    if class_id == 1:
        return "shelf"
    if class_id == 2:
        return "slope"
    if class_id == 3:
        return "basin"
    return "unknown"


def _to_idx_bounds(row: pd.Series, h: int, w: int, d: int) -> Tuple[int, int, int, int, int, int]:
    d0 = int(row["d0"])
    d1 = int(row["d1"])
    h0 = int(row["h0"])
    h1 = int(row["h1"])
    w0 = int(row["w0"])
    w1 = int(row["w1"])
    if d1 <= d0:
        d1 = d0 + 1
    if h1 <= h0:
        h1 = h0 + 1
    if w1 <= w0:
        w1 = w0 + 1
    d0, d1 = max(0, d0), min(d, d1)
    h0, h1 = max(0, h0), min(h, h1)
    w0, w1 = max(0, w0), min(w, w1)
    return d0, d1, h0, h1, w0, w1


def _safe_nanmean(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.nanmean(x))


def _safe_nanstd(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.nanstd(x))


def _build_patch_topo_metadata(
    patch_meta: pd.DataFrame,
    bottom_depth: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    relief: np.ndarray,
    shelf_mask: np.ndarray,
    slope_mask: np.ndarray,
    basin_mask: np.ndarray,
    topo_class: np.ndarray,
    near_bottom_200m: np.ndarray,
    wet_mask: np.ndarray,
) -> pd.DataFrame:
    h, w = bottom_depth.shape
    d = near_bottom_200m.shape[0]
    rows: List[Dict[str, Any]] = []

    for _, r in patch_meta.iterrows():
        d0, d1, h0, h1, w0, w1 = _to_idx_bounds(r, h=h, w=w, d=d)
        ocean2d = bottom_depth[h0:h1, w0:w1] > 0.0
        ocean_count = int(ocean2d.sum())

        bd = bottom_depth[h0:h1, w0:w1][ocean2d]
        sl = slope[h0:h1, w0:w1][ocean2d]
        rg = roughness[h0:h1, w0:w1][ocean2d]
        rf = relief[h0:h1, w0:w1][ocean2d]
        cls = topo_class[h0:h1, w0:w1][ocean2d]

        wet_win = wet_mask[d0:d1, h0:h1, w0:w1]
        nb_win = near_bottom_200m[d0:d1, h0:h1, w0:w1]
        wet_count = int(wet_win.sum())
        near_bottom_fraction = float(nb_win.sum() / wet_count) if wet_count > 0 else 0.0

        if cls.size > 0:
            vals, cnt = np.unique(cls, return_counts=True)
            majority_id = int(vals[int(np.argmax(cnt))])
        else:
            majority_id = 0

        row: Dict[str, Any] = {
            "patch_id": int(r["patch_id"]),
            "bottom_depth_mean": _safe_nanmean(bd),
            "bottom_depth_std": _safe_nanstd(bd),
            "slope_mean": _safe_nanmean(sl),
            "slope_std": _safe_nanstd(sl),
            "roughness_mean": _safe_nanmean(rg),
            "relief_mean": _safe_nanmean(rf),
            "shelf_fraction": float(shelf_mask[h0:h1, w0:w1][ocean2d].mean()) if ocean_count > 0 else 0.0,
            "slope_fraction": float(slope_mask[h0:h1, w0:w1][ocean2d].mean()) if ocean_count > 0 else 0.0,
            "basin_fraction": float(basin_mask[h0:h1, w0:w1][ocean2d].mean()) if ocean_count > 0 else 0.0,
            "near_bottom_fraction": near_bottom_fraction,
            "topo_class_majority": _class_label(majority_id),
        }
        for keep_col in ("split", "center_lat", "center_lon", "center_depth_m", "d0", "d1", "h0", "h1", "w0", "w1"):
            if keep_col in patch_meta.columns:
                row[keep_col] = r[keep_col]
        rows.append(row)
    return pd.DataFrame(rows)


def _build_patch_runtime_artifacts(
    patch_meta: pd.DataFrame,
    depth_levels: np.ndarray,
    wet_mask: np.ndarray,
    near_bottom_mask: np.ndarray,
    slope: np.ndarray,
    roughness: np.ndarray,
    topo_class: np.ndarray,
    out_runtime_dir: Path,
) -> Dict[str, Any]:
    ensure_dir(out_runtime_dir)
    req = {"patch_id", "d0", "d1", "h0", "h1", "w0", "w1"}
    if not req.issubset(set(patch_meta.columns)):
        raise ValueError(f"patch metadata missing required runtime columns: {sorted(req)}")

    patch_ids = patch_meta["patch_id"].astype(np.int64).to_numpy()
    bounds = patch_meta[["patch_id", "d0", "d1", "h0", "h1", "w0", "w1"]].copy()
    bounds.to_csv(out_runtime_dir / "patch_runtime_index.csv", index=False)

    d_sizes = (bounds["d1"] - bounds["d0"]).astype(np.int64).to_numpy()
    h_sizes = (bounds["h1"] - bounds["h0"]).astype(np.int64).to_numpy()
    w_sizes = (bounds["w1"] - bounds["w0"]).astype(np.int64).to_numpy()
    fixed_shape = bool(np.all(d_sizes == d_sizes[0]) and np.all(h_sizes == h_sizes[0]) and np.all(w_sizes == w_sizes[0]))

    runtime_manifest: Dict[str, Any] = {
        "runtime_mode": "global_arrays_plus_patch_index",
        "patch_count": int(patch_ids.shape[0]),
        "fixed_patch_shape": fixed_shape,
        "required_fields": ["patch_depth_m", "wet_mask", "near_bottom_mask", "slope_weight"],
        "optional_fields": ["roughness_weight", "topo_class"],
        "depth_levels_path": "unified_field/depth_levels.npy",
        "patch_runtime_index": str(out_runtime_dir / "patch_runtime_index.csv"),
    }

    if not fixed_shape:
        return runtime_manifest

    d = int(d_sizes[0])
    h = int(h_sizes[0])
    w = int(w_sizes[0])
    n = int(patch_ids.shape[0])

    patch_depth = np.zeros((n, d), dtype=np.float32)
    wet = np.zeros((n, d, h, w), dtype=np.uint8)
    near = np.zeros((n, d, h, w), dtype=np.uint8)
    slope_w = np.zeros((n, d, h, w), dtype=np.float32)
    rough_w = np.zeros((n, d, h, w), dtype=np.float32)
    topo_cls = np.zeros((n, h, w), dtype=np.uint8)

    ocean = (wet_mask.sum(axis=0) > 0)
    slope_scale = max(float(np.percentile(slope[ocean], 95.0)) if bool(ocean.any()) else float(np.percentile(slope, 95.0)), 1e-6)
    rough_scale = max(float(np.percentile(roughness[ocean], 95.0)) if bool(ocean.any()) else float(np.percentile(roughness, 95.0)), 1e-6)

    for i, row in enumerate(bounds.itertuples(index=False)):
        d0, d1 = int(row.d0), int(row.d1)
        h0, h1 = int(row.h0), int(row.h1)
        w0, w1 = int(row.w0), int(row.w1)
        patch_depth[i] = depth_levels[d0:d1][:d]
        wet[i] = wet_mask[d0:d1, h0:h1, w0:w1][:d, :h, :w]
        near[i] = near_bottom_mask[d0:d1, h0:h1, w0:w1][:d, :h, :w]
        s2 = np.clip(slope[h0:h1, w0:w1][:h, :w] / slope_scale, 0.0, 1.0).astype(np.float32)
        r2 = np.clip(roughness[h0:h1, w0:w1][:h, :w] / rough_scale, 0.0, 1.0).astype(np.float32)
        slope_w[i] = np.repeat(s2[None, :, :], d, axis=0)
        rough_w[i] = np.repeat(r2[None, :, :], d, axis=0)
        topo_cls[i] = topo_class[h0:h1, w0:w1][:h, :w].astype(np.uint8)

    np.save(out_runtime_dir / "patch_ids.npy", patch_ids, allow_pickle=False)
    np.save(out_runtime_dir / "patch_depth_m.npy", patch_depth, allow_pickle=False)
    np.save(out_runtime_dir / "wet_mask.npy", wet, allow_pickle=False)
    np.save(out_runtime_dir / "near_bottom_mask.npy", near, allow_pickle=False)
    np.save(out_runtime_dir / "slope_weight.npy", slope_w, allow_pickle=False)
    np.save(out_runtime_dir / "roughness_weight.npy", rough_w, allow_pickle=False)
    np.save(out_runtime_dir / "topo_class.npy", topo_cls, allow_pickle=False)

    runtime_manifest.update(
        {
            "runtime_mode": "fixed_patch_arrays_and_global_arrays",
            "patch_shape_dhw": [d, h, w],
            "paths": {
                "patch_ids": str(out_runtime_dir / "patch_ids.npy"),
                "patch_depth_m": str(out_runtime_dir / "patch_depth_m.npy"),
                "wet_mask": str(out_runtime_dir / "wet_mask.npy"),
                "near_bottom_mask": str(out_runtime_dir / "near_bottom_mask.npy"),
                "slope_weight": str(out_runtime_dir / "slope_weight.npy"),
                "roughness_weight": str(out_runtime_dir / "roughness_weight.npy"),
                "topo_class": str(out_runtime_dir / "topo_class.npy"),
            },
        }
    )
    return runtime_manifest


def _quality_checks(
    bottom_depth: np.ndarray,
    slope: np.ndarray,
    shelf_mask: np.ndarray,
    slope_mask: np.ndarray,
    basin_mask: np.ndarray,
    near_bottom_100m: np.ndarray,
    near_bottom_200m: np.ndarray,
    wet_mask: np.ndarray,
    depth_levels: np.ndarray,
    step2_info: Dict[str, Any],
    unified_mask_path: Path,
    patch_topo_df: pd.DataFrame,
) -> Dict[str, Any]:
    data_mask = np.load(unified_mask_path, allow_pickle=False)
    # data_mask: [var, depth, lat, lon], 1 means observed.
    observed_any = (data_mask > 0).any(axis=(0, 1))
    topo_ocean = bottom_depth > 0.0

    agreement = float((observed_any == topo_ocean).mean())
    topo_only_ratio = float((topo_ocean & (~observed_any)).mean())
    observed_only_ratio = float((observed_any & (~topo_ocean)).mean())

    ar = step2_info["actual_range"]
    lat = np.linspace(-89.5, 89.5, bottom_depth.shape[0], dtype=np.float32)
    lon = np.linspace(-179.5, 179.5, bottom_depth.shape[1], dtype=np.float32)
    lat_mask = (lat >= float(ar["lat_min"])) & (lat <= float(ar["lat_max"]))
    lon_mask = (lon >= float(ar["lon_min"])) & (lon <= float(ar["lon_max"]))
    china_ocean = topo_ocean[np.ix_(lat_mask, lon_mask)]

    wet_count_by_depth = wet_mask.sum(axis=(1, 2)).astype(np.float64)
    nb100_count_by_depth = near_bottom_100m.sum(axis=(1, 2)).astype(np.float64)
    nb200_count_by_depth = near_bottom_200m.sum(axis=(1, 2)).astype(np.float64)
    near100_ratio = np.divide(nb100_count_by_depth, np.maximum(wet_count_by_depth, 1.0))
    near200_ratio = np.divide(nb200_count_by_depth, np.maximum(wet_count_by_depth, 1.0))
    corr_100 = float(np.corrcoef(depth_levels, near100_ratio)[0, 1])
    corr_200 = float(np.corrcoef(depth_levels, near200_ratio)[0, 1])

    ocean = topo_ocean
    slope_ocean = slope[ocean]
    slope_p99 = float(np.percentile(slope_ocean, 99.0)) if slope_ocean.size > 0 else 0.0
    high_slope = (slope >= slope_p99) & ocean
    high_slope_lat = np.where(high_slope)[0]
    high_slope_lon = np.where(high_slope)[1]
    lat_v = lat[high_slope_lat] if high_slope_lat.size > 0 else np.array([], dtype=np.float32)
    lon_v = lon[high_slope_lon] if high_slope_lon.size > 0 else np.array([], dtype=np.float32)

    class_sum = (shelf_mask + slope_mask + basin_mask).astype(np.int16)
    class_ok = bool(np.all((class_sum == 1) | (~ocean)))

    numeric_cols = [
        "bottom_depth_mean",
        "bottom_depth_std",
        "slope_mean",
        "slope_std",
        "roughness_mean",
        "relief_mean",
        "shelf_fraction",
        "slope_fraction",
        "basin_fraction",
        "near_bottom_fraction",
    ]
    nan_counts = {c: int(patch_topo_df[c].isna().sum()) for c in numeric_cols}
    negative_depth_rows = int((patch_topo_df["bottom_depth_mean"] < 0).sum())
    frac_outside_01 = {
        c: int(((patch_topo_df[c] < 0.0) | (patch_topo_df[c] > 1.0)).sum())
        for c in ("shelf_fraction", "slope_fraction", "basin_fraction", "near_bottom_fraction")
    }

    return {
        "alignment_check": {
            "observed_vs_topo_ocean_agreement_ratio": agreement,
            "topo_ocean_without_observation_ratio": topo_only_ratio,
            "observed_but_topo_land_ratio": observed_only_ratio,
            "note": "Observed mask is not a strict ocean mask; mismatches may include data sparsity.",
        },
        "china_coverage_check": {
            "china_ocean_fraction_in_bbox": float(china_ocean.mean()) if china_ocean.size > 0 else 0.0,
            "china_bbox_shape_hw": [int(china_ocean.shape[0]), int(china_ocean.shape[1])],
        },
        "near_bottom_sanity": {
            "corr_depth_vs_near_bottom_ratio_100m": corr_100,
            "corr_depth_vs_near_bottom_ratio_200m": corr_200,
            "near_bottom_100m_ratio_min_max": [float(near100_ratio.min()), float(near100_ratio.max())],
            "near_bottom_200m_ratio_min_max": [float(near200_ratio.min()), float(near200_ratio.max())],
        },
        "high_slope_check": {
            "p99_threshold": slope_p99,
            "high_slope_cell_count": int(high_slope.sum()),
            "lat_min_max": [float(lat_v.min()) if lat_v.size > 0 else None, float(lat_v.max()) if lat_v.size > 0 else None],
            "lon_min_max": [float(lon_v.min()) if lon_v.size > 0 else None, float(lon_v.max()) if lon_v.size > 0 else None],
        },
        "class_consistency": {
            "one_class_per_ocean_cell": class_ok,
            "global_shelf_fraction_over_ocean": float(shelf_mask[ocean].mean()) if ocean.any() else 0.0,
            "global_slope_fraction_over_ocean": float(slope_mask[ocean].mean()) if ocean.any() else 0.0,
            "global_basin_fraction_over_ocean": float(basin_mask[ocean].mean()) if ocean.any() else 0.0,
        },
        "patch_stats_anomaly_check": {
            "nan_counts": nan_counts,
            "negative_bottom_depth_mean_rows": negative_depth_rows,
            "fraction_outside_0_1_counts": frac_outside_01,
        },
    }


def _build_markdown_report(
    report_path: Path,
    runtime_meta: Dict[str, Any],
    alignment_summary: Dict[str, Any],
    quality: Dict[str, Any],
    patch_rows: int,
) -> None:
    lines: List[str] = []
    lines.append("# TOPO Preparation Auto Report")
    lines.append("")
    lines.append("## Runtime")
    lines.append(f"- Python executable: `{runtime_meta['python_executable']}`")
    lines.append(f"- Config file: `{runtime_meta['config_path']}`")
    lines.append(f"- Output root: `{runtime_meta['out_dir']}`")
    lines.append("")
    lines.append("## Alignment Summary")
    lines.append(f"- Raw topo file: `{alignment_summary['raw_topo_file']}`")
    lines.append(f"- Raw topo shape (ny,nx): `{alignment_summary['raw_topo_shape_ny_nx']}`")
    lines.append(f"- Training grid shape (h,w): `{alignment_summary['training_grid_shape_h_w']}`")
    lines.append(f"- Longitude convention: `{alignment_summary['lon_convention']}`")
    lines.append(f"- Depth sign convention: `{alignment_summary['depth_sign_convention']}`")
    lines.append("")
    lines.append("## Patch Metadata")
    lines.append(f"- Patch rows generated: `{patch_rows}`")
    lines.append("")
    lines.append("## Quality Checks")
    lines.append("```json")
    lines.append(json.dumps(quality, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Manual Confirmation Required")
    lines.append("- Verify high-slope map and shelf/slope/basin partition near complex coastlines.")
    lines.append("- Confirm whether patch metadata source should be strict-China strict dataset or global pretrain dataset for downstream tasks.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_figures(
    figures_dir: Path,
    bottom_depth: np.ndarray,
    slope: np.ndarray,
    relief: np.ndarray,
    near_bottom_200m: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    ensure_dir(figures_dir)

    plt.figure(figsize=(12, 5))
    plt.imshow(bottom_depth, cmap="Blues", origin="lower")
    plt.colorbar(label="m")
    plt.title("Bottom Depth on Training Grid (m)")
    plt.tight_layout()
    plt.savefig(figures_dir / "bottom_depth_map.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.imshow(np.log10(np.maximum(slope, 1e-6)), cmap="magma", origin="lower")
    plt.colorbar(label="log10(slope)")
    plt.title("Seafloor Slope")
    plt.tight_layout()
    plt.savefig(figures_dir / "slope_map.png", dpi=180)
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.imshow(relief, cmap="viridis", origin="lower")
    plt.colorbar(label="m")
    plt.title("Local Relief (5x5)")
    plt.tight_layout()
    plt.savefig(figures_dir / "relief_map.png", dpi=180)
    plt.close()

    near_ratio = near_bottom_200m.mean(axis=0)
    plt.figure(figsize=(12, 5))
    plt.imshow(near_ratio, cmap="plasma", origin="lower", vmin=0.0, vmax=1.0)
    plt.colorbar(label="fraction over depth levels")
    plt.title("Near-Bottom Ratio (200m)")
    plt.tight_layout()
    plt.savefig(figures_dir / "near_bottom_ratio_200m_map.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = _load_yaml(args.config)

    topo_file = Path(args.topo_file or _nested_get(cfg, ("inputs", "topo_file"), "data/ETOPO1_Bed_g_gdal.grd"))
    unified_dir = Path(args.unified_dir or _nested_get(cfg, ("inputs", "unified_dir"), "data/processed/unified_field"))
    step2_dir = Path(args.step2_dir or _nested_get(cfg, ("inputs", "step2_dir"), "outputs/china_strict"))
    patch_meta_override = args.patch_metadata
    out_dir = Path(args.out_dir or _nested_get(cfg, ("outputs", "out_dir"), "outputs/enhancement/topo_prepared"))

    arrays_dir = out_dir / "arrays"
    metadata_dir = out_dir / "metadata"
    reports_dir = out_dir / "reports"
    figures_dir = out_dir / "figures"
    logs_dir = out_dir / "logs"
    for d in (arrays_dir, metadata_dir, reports_dir, figures_dir, logs_dir):
        ensure_dir(d)

    patch_meta_path = patch_meta_override or pick_existing(
        Path(
            _nested_get(
                cfg,
                ("inputs", "patch_metadata_preferred"),
                _nested_get(cfg, ("inputs", "patch_metadata_primary"), "outputs/china_strict/patch_metadata.csv"),
            )
        ),
        Path(_nested_get(cfg, ("inputs", "patch_metadata_fallback"), "outputs/global_pretrain/patch_metadata.csv")),
    )

    depth_levels = np.load(unified_dir / "depth_levels.npy", allow_pickle=False).astype(np.float32)
    lat, lon = get_training_grid(unified_dir=unified_dir)
    step2_info = json.loads((step2_dir / "china_region_info.json").read_text(encoding="utf-8"))

    topo = load_etopo1_grd(path=topo_file)
    topo_on_grid = bilinear_resample_to_grid(
        src_lon=topo.lon, src_lat=topo.lat, src_field=topo.elevation_m, target_lon=lon, target_lat=lat
    ).astype(np.float32)
    _save_array(arrays_dir / "topo_on_training_grid.npy", topo_on_grid)
    _save_array(out_dir / "topo_on_training_grid.npy", topo_on_grid)

    bottom_depth = bottom_depth_from_elevation(topo_on_grid)
    topo_features = compute_topo_derivatives(
        bottom_depth_m=bottom_depth,
        lat=lat,
        lon=lon,
        roughness_radius=int(_nested_get(cfg, ("derivation", "roughness_radius"), 1)),
        relief_radius=int(_nested_get(cfg, ("derivation", "relief_radius"), 2)),
    )
    classes = classify_topo(
        bottom_depth_m=topo_features["bottom_depth"],
        slope=topo_features["slope"],
        shelf_max_depth_m=float(_nested_get(cfg, ("classification", "shelf_max_depth_m"), 200.0)),
        basin_min_depth_m=float(_nested_get(cfg, ("classification", "basin_min_depth_m"), 2000.0)),
        slope_min_gradient=float(_nested_get(cfg, ("classification", "slope_min_gradient"), 0.002)),
    )

    threshold_cfg = _nested_get(cfg, ("near_bottom", "thresholds_m"), [100.0, 200.0])
    if isinstance(threshold_cfg, (int, float, str)):
        threshold_cfg = [float(threshold_cfg)]
    if not isinstance(threshold_cfg, list):
        threshold_cfg = [100.0, 200.0]
    near_bottom_thresholds = tuple(float(v) for v in threshold_cfg)
    if len(near_bottom_thresholds) == 0:
        near_bottom_thresholds = (100.0, 200.0)
    masks = build_wet_and_near_bottom_masks(
        depth_levels_m=depth_levels, bottom_depth_m=topo_features["bottom_depth"], thresholds_m=near_bottom_thresholds
    )

    _save_array(arrays_dir / "bottom_depth.npy", topo_features["bottom_depth"])
    _save_array(arrays_dir / "slope.npy", topo_features["slope"])
    _save_array(arrays_dir / "roughness.npy", topo_features["roughness"])
    _save_array(arrays_dir / "relief.npy", topo_features["relief"])
    _save_array(arrays_dir / "shelf_mask.npy", classes["shelf_mask"])
    _save_array(arrays_dir / "slope_mask.npy", classes["slope_mask"])
    _save_array(arrays_dir / "basin_mask.npy", classes["basin_mask"])
    _save_array(arrays_dir / "topo_class.npy", classes["topo_class"])
    _save_array(arrays_dir / "wet_mask_from_topo.npy", masks["wet_mask"])
    _save_array(out_dir / "wet_mask_from_topo.npy", masks["wet_mask"])

    for th in near_bottom_thresholds:
        k = f"near_bottom_mask_{int(th)}m"
        _save_array(arrays_dir / f"{k}.npy", masks[k])
    if "near_bottom_mask_200m" in masks:
        _save_array(out_dir / "near_bottom_mask.npy", masks["near_bottom_mask_200m"])
    else:
        first = f"near_bottom_mask_{int(near_bottom_thresholds[0])}m"
        _save_array(out_dir / "near_bottom_mask.npy", masks[first])

    alignment_summary = {
        "raw_topo_file": str(topo_file),
        "raw_topo_shape_ny_nx": [int(topo.elevation_m.shape[0]), int(topo.elevation_m.shape[1])],
        "raw_topo_meta": topo.raw_meta,
        "training_grid_shape_h_w": [int(lat.size), int(lon.size)],
        "training_grid_lat_range": [float(lat.min()), float(lat.max())],
        "training_grid_lon_range": [float(lon.min()), float(lon.max())],
        "lon_convention": "-180_to_180",
        "lat_direction": "ascending",
        "depth_sign_convention": "bottom_depth_positive_down_meters",
        "land_or_invalid_encoding": "0_in_bottom_depth_and_masks",
    }
    save_json(metadata_dir / "topo_alignment_summary.json", alignment_summary)
    save_json(out_dir / "topo_alignment_summary.json", alignment_summary)

    patch_df = pd.read_csv(patch_meta_path)
    if not {"patch_id", "d0", "d1", "h0", "h1", "w0", "w1"}.issubset(set(patch_df.columns)):
        raise ValueError("patch metadata must include patch_id,d0,d1,h0,h1,w0,w1")

    near_bottom_for_patch = masks["near_bottom_mask_200m"] if "near_bottom_mask_200m" in masks else masks[
        f"near_bottom_mask_{int(near_bottom_thresholds[0])}m"
    ]
    patch_topo = _build_patch_topo_metadata(
        patch_meta=patch_df,
        bottom_depth=topo_features["bottom_depth"],
        slope=topo_features["slope"],
        roughness=topo_features["roughness"],
        relief=topo_features["relief"],
        shelf_mask=classes["shelf_mask"],
        slope_mask=classes["slope_mask"],
        basin_mask=classes["basin_mask"],
        topo_class=classes["topo_class"],
        near_bottom_200m=near_bottom_for_patch,
        wet_mask=masks["wet_mask"],
    )
    patch_out = metadata_dir / "patch_topo_metadata.csv"
    patch_topo.to_csv(patch_out, index=False)
    patch_topo.to_csv(out_dir / "patch_topo_metadata.csv", index=False)
    runtime_manifest = _build_patch_runtime_artifacts(
        patch_meta=patch_df,
        depth_levels=depth_levels,
        wet_mask=masks["wet_mask"],
        near_bottom_mask=near_bottom_for_patch,
        slope=topo_features["slope"],
        roughness=topo_features["roughness"],
        topo_class=classes["topo_class"],
        out_runtime_dir=out_dir / "runtime",
    )
    save_json(metadata_dir / "topo_runtime_manifest.json", runtime_manifest)
    save_json(out_dir / "topo_runtime_manifest.json", runtime_manifest)

    quality = _quality_checks(
        bottom_depth=topo_features["bottom_depth"],
        slope=topo_features["slope"],
        shelf_mask=classes["shelf_mask"],
        slope_mask=classes["slope_mask"],
        basin_mask=classes["basin_mask"],
        near_bottom_100m=masks["near_bottom_mask_100m"]
        if "near_bottom_mask_100m" in masks
        else masks[f"near_bottom_mask_{int(near_bottom_thresholds[0])}m"],
        near_bottom_200m=near_bottom_for_patch,
        wet_mask=masks["wet_mask"],
        depth_levels=depth_levels,
        step2_info=step2_info,
        unified_mask_path=unified_dir / "mask_8vars.npy",
        patch_topo_df=patch_topo,
    )
    save_json(reports_dir / "topo_quality_summary.json", quality)
    save_json(out_dir / "topo_quality_summary.json", quality)

    runtime_meta = {
        "python_executable": sys.executable,
        "config_path": str(args.config),
        "out_dir": str(out_dir),
        "patch_metadata_used": str(patch_meta_path),
        "topo_runtime_manifest": str(out_dir / "topo_runtime_manifest.json"),
    }
    save_json(logs_dir / "run_manifest.json", runtime_meta)
    save_json(out_dir / "topo_preparation_manifest.json", runtime_meta)
    _build_markdown_report(
        report_path=reports_dir / "topo_quality_report.md",
        runtime_meta=runtime_meta,
        alignment_summary=alignment_summary,
        quality=quality,
        patch_rows=int(len(patch_topo)),
    )

    figures_enabled_cfg = bool(_nested_get(cfg, ("figures", "enabled"), True))
    if figures_enabled_cfg and (not args.no_figures):
        _save_figures(
            figures_dir=figures_dir,
            bottom_depth=topo_features["bottom_depth"],
            slope=topo_features["slope"],
            relief=topo_features["relief"],
            near_bottom_200m=near_bottom_for_patch,
        )

    print("Topo preparation completed.")
    print(f"Output root: {out_dir}")
    print(f"Patch metadata rows: {len(patch_topo)}")


if __name__ == "__main__":
    main()






