#!/usr/bin/env python
"""Build the unified WOA23 8-variable field and validate data readiness for modeling.

This script:
1) scans NetCDF/CSV sources in a data directory,
2) selects each NetCDF main field with a strict preference for '*_an',
3) parses oxygen saturation CSV into a regular [depth, lat, lon] grid,
4) validates coordinate/depth/missing-value/type consistency,
5) stacks 8 variables into X in R^(8 x 102 x H x W),
6) generates mask M in {0,1}^(8 x 102 x H x W),
7) writes outputs and a markdown validation report.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import xarray as xr


LAT_CANDIDATES = ("lat", "latitude", "nav_lat", "y")
LON_CANDIDATES = ("lon", "longitude", "nav_lon", "x")
DEPTH_CANDIDATES = ("depth", "lev", "level", "deptht", "z")
TIME_CANDIDATES = ("time", "t")


def _pick_name(names: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower_map = {n.lower(): n for n in names}
    for c in candidates:
        if c in lower_map:
            return lower_map[c]
    return None


def _safe_float(token: str) -> float:
    t = token.strip()
    if t == "":
        return np.nan
    return float(t)


@dataclass
class VarRecord:
    var_name: str
    source_file: str
    source_field: str
    standard_name: str
    units: str
    source_dtype: str
    output_dtype: str
    fill_values_found: str
    shape_depth_lat_lon: Tuple[int, int, int]
    missing_rate: float
    finite_min: float
    finite_max: float
    empty_layers: int
    duplicates_detected: int


def detect_primary_field(ds: xr.Dataset) -> Tuple[str, List[str]]:
    """Return chosen field name and ordered candidates with explicit scoring."""
    dims = set(ds.dims)
    lat_name = _pick_name(ds.coords, LAT_CANDIDATES) or _pick_name(ds.variables, LAT_CANDIDATES)
    lon_name = _pick_name(ds.coords, LON_CANDIDATES) or _pick_name(ds.variables, LON_CANDIDATES)
    depth_name = _pick_name(ds.coords, DEPTH_CANDIDATES) or _pick_name(ds.variables, DEPTH_CANDIDATES)
    time_name = _pick_name(ds.coords, TIME_CANDIDATES) or _pick_name(ds.variables, TIME_CANDIDATES)
    if lat_name is None or lon_name is None or depth_name is None:
        raise ValueError("Cannot detect lat/lon/depth coordinate names in NetCDF file")

    scored: List[Tuple[int, str]] = []
    for var_name_raw, da in ds.data_vars.items():
        var_name = str(var_name_raw)
        var_dims = set(da.dims)
        has_core = {lat_name, lon_name, depth_name}.issubset(var_dims)
        if not has_core:
            continue
        if not (len(da.dims) in (3, 4)):
            continue
        score = 0
        if var_name.endswith("_an"):
            score += 100
        if "analy" in str(da.attrs.get("long_name", "")).lower():
            score += 20
        if time_name and time_name in da.dims and da.sizes.get(time_name, 0) == 1:
            score += 5
        if np.issubdtype(da.dtype, np.floating):
            score += 2
        scored.append((score, var_name))

    if not scored:
        raise ValueError("No candidate data variable has [depth, lat, lon] core dimensions")

    scored.sort(key=lambda x: (-x[0], x[1]))
    ordered_candidates = [name for _, name in scored]
    return ordered_candidates[0], ordered_candidates


def load_netcdf_main_field(nc_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    """Load one NetCDF file and return [depth, lat, lon] data plus coords and metadata."""
    with xr.open_dataset(nc_path, decode_times=False) as ds:
        lat_name = _pick_name(ds.coords, LAT_CANDIDATES) or _pick_name(ds.variables, LAT_CANDIDATES)
        lon_name = _pick_name(ds.coords, LON_CANDIDATES) or _pick_name(ds.variables, LON_CANDIDATES)
        depth_name = _pick_name(ds.coords, DEPTH_CANDIDATES) or _pick_name(ds.variables, DEPTH_CANDIDATES)
        time_name = _pick_name(ds.coords, TIME_CANDIDATES) or _pick_name(ds.variables, TIME_CANDIDATES)

        if lat_name is None or lon_name is None or depth_name is None:
            raise ValueError(f"Missing required coordinates in {nc_path.name}")

        chosen, candidates = detect_primary_field(ds)
        da = ds[chosen]

        if time_name in da.dims:
            if da.sizes[time_name] < 1:
                raise ValueError(f"Time dimension is empty in {nc_path.name}:{chosen}")
            da = da.isel({time_name: 0})

        da = da.transpose(depth_name, lat_name, lon_name)
        arr = da.values

        # Normalize explicit missing codes to NaN before validation.
        fill_values = set()
        for key in ("_FillValue", "missing_value"):
            if key in da.attrs:
                try:
                    fill_values.add(float(da.attrs[key]))
                except (TypeError, ValueError):
                    pass
            if key in da.encoding:
                try:
                    fill_values.add(float(da.encoding[key]))
                except (TypeError, ValueError):
                    pass

        arr = np.asarray(arr, dtype=np.float32)
        for fv in fill_values:
            arr[np.isclose(arr, fv, equal_nan=False)] = np.nan

        meta = {
            "source_file": nc_path.name,
            "source_field": chosen,
            "candidates": candidates,
            "standard_name": str(da.attrs.get("standard_name", "")),
            "units": str(da.attrs.get("units", "")),
            "source_dtype": str(ds[chosen].dtype),
            "fill_values": sorted(fill_values),
            "lat_name": lat_name,
            "lon_name": lon_name,
            "depth_name": depth_name,
        }
        lat = np.asarray(ds[lat_name].values, dtype=np.float32)
        lon = np.asarray(ds[lon_name].values, dtype=np.float32)
        depth = np.asarray(ds[depth_name].values, dtype=np.float32)

    return arr, lat, lon, depth, meta


def parse_csv_depths(csv_path: Path) -> List[float]:
    with csv_path.open("r", encoding="utf-8") as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()
    if not line1.startswith("#") or not line2.startswith("#"):
        raise ValueError(f"CSV {csv_path.name} does not start with expected metadata comment lines")

    match = re.search(r"DEPTHS \(M\):(.*)$", line2)
    if not match:
        raise ValueError(f"Cannot parse depth metadata from second line in {csv_path.name}")

    tokens = [tok.strip() for tok in match.group(1).split(",") if tok.strip() != ""]
    depths = [float(tok) for tok in tokens]
    return depths


def load_oxygen_csv_to_grid(
    csv_path: Path,
    ref_depth: np.ndarray,
    ref_lat: np.ndarray,
    ref_lon: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, object]]:
    """Read oxygen saturation CSV and map ragged rows to [depth, lat, lon] on reference grid."""
    csv_depths = np.asarray(parse_csv_depths(csv_path), dtype=np.float32)
    if csv_depths.size == 0:
        raise ValueError(f"No depth values parsed from {csv_path.name}")

    # Depth alignment is by value, not by index, to avoid hidden order mismatches.
    depth_map: Dict[int, int] = {}
    used_ref_idx = set()
    for i, d in enumerate(csv_depths):
        idx = int(np.argmin(np.abs(ref_depth - d)))
        if not np.isclose(ref_depth[idx], d, atol=1e-4):
            raise ValueError(
                f"CSV depth {d} m cannot be matched to NetCDF depth levels (nearest={ref_depth[idx]})"
            )
        if idx in used_ref_idx:
            raise ValueError("CSV depths map to duplicate NetCDF depth indices")
        depth_map[i] = idx
        used_ref_idx.add(idx)

    arr = np.full((ref_depth.size, ref_lat.size, ref_lon.size), np.nan, dtype=np.float32)
    seen_points = np.zeros((ref_lat.size, ref_lon.size), dtype=np.uint16)

    lat_idx = {round(float(v), 6): i for i, v in enumerate(ref_lat)}
    lon_idx = {round(float(v), 6): i for i, v in enumerate(ref_lon)}

    row_count = 0
    bad_coord_rows = 0
    malformed_rows = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        for row in reader:
            if not row:
                continue
            row_count += 1
            if len(row) < 2:
                malformed_rows += 1
                continue
            try:
                latv = round(float(row[0]), 6)
                lonv = round(float(row[1]), 6)
            except ValueError:
                malformed_rows += 1
                continue

            if latv not in lat_idx or lonv not in lon_idx:
                bad_coord_rows += 1
                continue

            i_lat = lat_idx[latv]
            i_lon = lon_idx[lonv]
            seen_points[i_lat, i_lon] += 1

            values = row[2:]
            if len(values) < csv_depths.size:
                values = values + [""] * (csv_depths.size - len(values))
            elif len(values) > csv_depths.size:
                values = values[: csv_depths.size]

            for i_csv, token in enumerate(values):
                i_ref = depth_map[i_csv]
                try:
                    v = _safe_float(token)
                except ValueError:
                    v = np.nan
                arr[i_ref, i_lat, i_lon] = v

    duplicate_points = int(np.sum(seen_points > 1))

    meta = {
        "source_file": csv_path.name,
        "source_field": "csv_depth_values",
        "candidates": ["csv_depth_values"],
        "standard_name": "percent_oxygen_saturation",
        "units": "percent",
        "source_dtype": "csv_float",
        "fill_values": [],
        "csv_depth_count": int(csv_depths.size),
        "csv_row_count": row_count,
        "csv_malformed_rows": malformed_rows,
        "csv_bad_coord_rows": bad_coord_rows,
        "csv_duplicate_points": duplicate_points,
    }
    return arr, meta


def check_coords_unique(coord: np.ndarray, name: str) -> None:
    uniq = np.unique(coord)
    if uniq.size != coord.size:
        raise ValueError(f"Coordinate {name} has duplicate values ({coord.size - uniq.size} duplicates)")


def summarize_array(arr: np.ndarray) -> Dict[str, object]:
    finite = np.isfinite(arr)
    total = arr.size
    valid = int(finite.sum())
    missing = total - valid
    missing_rate = missing / total if total else np.nan

    if valid > 0:
        vmin = float(np.nanmin(arr))
        vmax = float(np.nanmax(arr))
    else:
        vmin = np.nan
        vmax = np.nan

    empty_layers = int(np.sum(np.all(~finite, axis=(1, 2))))
    inf_count = int(np.isinf(arr).sum())

    return {
        "total": int(total),
        "valid": valid,
        "missing": int(missing),
        "missing_rate": float(missing_rate),
        "finite_min": vmin,
        "finite_max": vmax,
        "empty_layers": empty_layers,
        "inf_count": inf_count,
    }


def dataframe_to_markdown_table(df: pd.DataFrame) -> str:
    """Render a simple markdown table without optional third-party dependencies."""
    cols = [str(c) for c in df.columns]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows: List[str] = []
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep] + rows)


def write_markdown_report(
    report_path: Path,
    data_dir: Path,
    variable_map_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    depth: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    X_shape: Tuple[int, int, int, int],
    M_shape: Tuple[int, int, int, int],
    coord_checks: Dict[str, bool],
    dtype_checks: Dict[str, object],
    issues: List[str],
) -> None:
    lines: List[str] = []
    lines.append("# Unified Field Data Validation Report")
    lines.append("")
    lines.append(f"- Data directory: `{data_dir}`")
    lines.append(f"- Unified tensor shape `X`: `{X_shape}`")
    lines.append(f"- Mask shape `M`: `{M_shape}`")
    lines.append("")
    lines.append("## Coordinate Summary")
    lines.append("")
    lines.append(f"- Depth levels: `{depth.size}`")
    lines.append(f"- Latitude count: `{lat.size}`; range: `{float(lat.min())}` to `{float(lat.max())}`")
    lines.append(f"- Longitude count: `{lon.size}`; range: `{float(lon.min())}` to `{float(lon.max())}`")
    lines.append(f"- Lat equal across NetCDF vars: `{coord_checks['lat_equal_all_nc']}`")
    lines.append(f"- Lon equal across NetCDF vars: `{coord_checks['lon_equal_all_nc']}`")
    lines.append(f"- Depth equal across NetCDF vars: `{coord_checks['depth_equal_all_nc']}`")
    lines.append(f"- CSV depth mappable to NetCDF depth: `{coord_checks['csv_depth_mappable']}`")
    lines.append("")
    lines.append("## Missing / Type Checks")
    lines.append("")
    lines.append(f"- Source dtypes: `{dtype_checks['source_dtypes']}`")
    lines.append(f"- Output dtype: `{dtype_checks['output_dtype']}`")
    lines.append(f"- Fill values found: `{dtype_checks['fill_values_found']}`")
    lines.append("")

    lines.append("## Variable-File-Field Mapping")
    lines.append("")
    lines.append(dataframe_to_markdown_table(variable_map_df))
    lines.append("")
    lines.append("## Missing Rate Table")
    lines.append("")
    lines.append(dataframe_to_markdown_table(missing_df))
    lines.append("")
    lines.append("## Depth List (m)")
    lines.append("")
    lines.append(", ".join(str(float(d)) for d in depth))
    lines.append("")
    lines.append("## Findings")
    lines.append("")
    if issues:
        for it in issues:
            lines.append(f"- {it}")
    else:
        lines.append("- No blocking inconsistencies found in this run.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build AquaSim unified WOA23 variables and validate")
    parser.add_argument(
        "--data-dir",
        default="data/raw",
        help="Directory containing NetCDF and CSV sources (default: data)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/unified_field",
        help="Output directory for unified tensors and reports",
    )
    parser.add_argument(
        "--save-nc",
        action="store_true",
        help="Also save unified tensor and mask to NetCDF",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    nc_files = sorted(data_dir.glob("*.nc"))
    csv_files = sorted(data_dir.glob("*.csv"))

    if len(nc_files) != 7:
        raise RuntimeError(f"Expected 7 NetCDF files, found {len(nc_files)} in {data_dir}")
    if len(csv_files) < 1:
        raise RuntimeError(f"Expected at least 1 CSV file, found {len(csv_files)} in {data_dir}")

    oxygen_csv = None
    for c in csv_files:
        if "O00mn" in c.name or "o00mn" in c.name:
            oxygen_csv = c
            break
    if oxygen_csv is None:
        oxygen_csv = csv_files[0]

    print(f"[INFO] Data directory: {data_dir}")
    print(f"[INFO] NetCDF files ({len(nc_files)}): {[p.name for p in nc_files]}")
    print(f"[INFO] Oxygen CSV: {oxygen_csv.name}")

    arrays: List[np.ndarray] = []
    var_records: List[VarRecord] = []
    source_inventory: List[Dict[str, object]] = []

    ref_lat = None
    ref_lon = None
    ref_depth = None

    lat_equal_all_nc = True
    lon_equal_all_nc = True
    depth_equal_all_nc = True

    source_dtypes = []
    all_fill_values = set()

    for nc_path in nc_files:
        print(f"[INFO] Reading NetCDF: {nc_path.name}")
        arr, lat, lon, depth, meta = load_netcdf_main_field(nc_path)

        check_coords_unique(lat, f"{nc_path.name}:lat")
        check_coords_unique(lon, f"{nc_path.name}:lon")
        check_coords_unique(depth, f"{nc_path.name}:depth")

        if ref_lat is None:
            ref_lat, ref_lon, ref_depth = lat, lon, depth
        else:
            lat_equal_all_nc = lat_equal_all_nc and bool(np.allclose(lat, ref_lat, atol=1e-6))
            lon_equal_all_nc = lon_equal_all_nc and bool(np.allclose(lon, ref_lon, atol=1e-6))
            depth_equal_all_nc = depth_equal_all_nc and bool(np.allclose(depth, ref_depth, atol=1e-6))

        stats = summarize_array(arr)

        arrays.append(arr)
        source_dtypes.append(meta["source_dtype"])
        for fv in meta["fill_values"]:
            all_fill_values.add(float(fv))

        var_records.append(
            VarRecord(
                var_name=str(meta["source_field"]),
                source_file=str(meta["source_file"]),
                source_field=str(meta["source_field"]),
                standard_name=str(meta["standard_name"]),
                units=str(meta["units"]),
                source_dtype=str(meta["source_dtype"]),
                output_dtype="float32",
                fill_values_found=str(meta["fill_values"]),
                shape_depth_lat_lon=arr.shape,
                missing_rate=float(stats["missing_rate"]),
                finite_min=float(stats["finite_min"]),
                finite_max=float(stats["finite_max"]),
                empty_layers=int(stats["empty_layers"]),
                duplicates_detected=0,
            )
        )

        source_inventory.append(
            {
                "file": meta["source_file"],
                "selected_field": meta["source_field"],
                "candidate_fields": meta["candidates"],
                "shape_depth_lat_lon": arr.shape,
                "standard_name": meta["standard_name"],
                "units": meta["units"],
                "source_dtype": meta["source_dtype"],
                "fill_values": meta["fill_values"],
            }
        )

    assert ref_lat is not None and ref_lon is not None and ref_depth is not None

    print(f"[INFO] Reading oxygen CSV: {oxygen_csv.name}")
    ox_arr, ox_meta = load_oxygen_csv_to_grid(
        oxygen_csv,
        ref_depth=ref_depth,
        ref_lat=ref_lat,
        ref_lon=ref_lon,
    )
    ox_stats = summarize_array(ox_arr)

    arrays.append(ox_arr)
    source_dtypes.append(ox_meta["source_dtype"])
    var_records.append(
        VarRecord(
            var_name="O2sat_csv",
            source_file=str(ox_meta["source_file"]),
            source_field=str(ox_meta["source_field"]),
            standard_name=str(ox_meta["standard_name"]),
            units=str(ox_meta["units"]),
            source_dtype=str(ox_meta["source_dtype"]),
            output_dtype="float32",
            fill_values_found="[]",
            shape_depth_lat_lon=ox_arr.shape,
            missing_rate=float(ox_stats["missing_rate"]),
            finite_min=float(ox_stats["finite_min"]),
            finite_max=float(ox_stats["finite_max"]),
            empty_layers=int(ox_stats["empty_layers"]),
            duplicates_detected=int(ox_meta["csv_duplicate_points"]),
        )
    )
    source_inventory.append(
        {
            "file": ox_meta["source_file"],
            "selected_field": "O2sat_csv",
            "candidate_fields": ["csv_depth_values"],
            "shape_depth_lat_lon": ox_arr.shape,
            "standard_name": ox_meta["standard_name"],
            "units": ox_meta["units"],
            "source_dtype": ox_meta["source_dtype"],
            "fill_values": [],
            "csv_depth_count": ox_meta["csv_depth_count"],
            "csv_row_count": ox_meta["csv_row_count"],
            "csv_malformed_rows": ox_meta["csv_malformed_rows"],
            "csv_bad_coord_rows": ox_meta["csv_bad_coord_rows"],
            "csv_duplicate_points": ox_meta["csv_duplicate_points"],
        }
    )

    X = np.stack(arrays, axis=0).astype(np.float32)
    M = np.isfinite(X).astype(np.uint8)

    # Global anomaly checks avoid variable-specific assumptions.
    issues: List[str] = []
    if not lat_equal_all_nc:
        issues.append("NetCDF latitude arrays are not identical across files.")
    if not lon_equal_all_nc:
        issues.append("NetCDF longitude arrays are not identical across files.")
    if not depth_equal_all_nc:
        issues.append("NetCDF depth arrays are not identical across files.")

    if int(ox_meta["csv_bad_coord_rows"]) > 0:
        issues.append(f"CSV rows outside NetCDF grid: {ox_meta['csv_bad_coord_rows']}")
    if int(ox_meta["csv_malformed_rows"]) > 0:
        issues.append(f"CSV malformed rows: {ox_meta['csv_malformed_rows']}")
    if int(ox_meta["csv_duplicate_points"]) > 0:
        issues.append(f"CSV duplicated (lat,lon) points: {ox_meta['csv_duplicate_points']}")

    inf_count_total = int(np.isinf(X).sum())
    if inf_count_total > 0:
        issues.append(f"Infinite values found in unified tensor: {inf_count_total}")

    extreme_count = int(np.sum(np.abs(X[np.isfinite(X)]) > 1e10))
    if extreme_count > 0:
        issues.append(f"Values with |x| > 1e10 found: {extreme_count}")

    # Save core artifacts.
    np.save(out_dir / "dataset_8vars.npy", X)
    np.save(out_dir / "mask_8vars.npy", M)
    np.save(out_dir / "depth_levels.npy", ref_depth)

    with (out_dir / "lat_lon_range.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "lat_min": float(ref_lat.min()),
                "lat_max": float(ref_lat.max()),
                "lon_min": float(ref_lon.min()),
                "lon_max": float(ref_lon.max()),
                "lat_count": int(ref_lat.size),
                "lon_count": int(ref_lon.size),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    variable_map_df = pd.DataFrame(
        [
            {
                "var_name": r.var_name,
                "source_file": r.source_file,
                "source_field": r.source_field,
                "shape": str(r.shape_depth_lat_lon),
                "source_dtype": r.source_dtype,
                "output_dtype": r.output_dtype,
                "units": r.units,
                "standard_name": r.standard_name,
                "fill_values_found": r.fill_values_found,
                "duplicates_detected": r.duplicates_detected,
            }
            for r in var_records
        ]
    )
    missing_df = pd.DataFrame(
        [
            {
                "var_name": r.var_name,
                "missing_rate": r.missing_rate,
                "finite_min": r.finite_min,
                "finite_max": r.finite_max,
                "empty_layers": r.empty_layers,
            }
            for r in var_records
        ]
    )

    variable_map_df.to_csv(out_dir / "variable_dimension_mapping.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(out_dir / "missing_rate_stats.csv", index=False, encoding="utf-8-sig")

    with (out_dir / "source_inventory.json").open("w", encoding="utf-8") as f:
        json.dump(source_inventory, f, indent=2, ensure_ascii=False)

    coord_checks = {
        "lat_equal_all_nc": lat_equal_all_nc,
        "lon_equal_all_nc": lon_equal_all_nc,
        "depth_equal_all_nc": depth_equal_all_nc,
        "csv_depth_mappable": True,
    }
    dtype_checks = {
        "source_dtypes": sorted(set(source_dtypes)),
        "output_dtype": "float32",
        "fill_values_found": sorted(all_fill_values),
    }

    write_markdown_report(
        report_path=out_dir / "unified_field_data_validation_report.md",
        data_dir=data_dir,
        variable_map_df=variable_map_df,
        missing_df=missing_df,
        depth=ref_depth,
        lat=ref_lat,
        lon=ref_lon,
        X_shape=tuple(X.shape),
        M_shape=tuple(M.shape),
        coord_checks=coord_checks,
        dtype_checks=dtype_checks,
        issues=issues,
    )

    if args.save_nc:
        ds_out = xr.Dataset(
            data_vars={
                "dataset_8vars": (("var", "depth", "lat", "lon"), X),
                "mask_8vars": (("var", "depth", "lat", "lon"), M),
            },
            coords={
                "var": np.arange(X.shape[0], dtype=np.int16),
                "depth": ref_depth,
                "lat": ref_lat,
                "lon": ref_lon,
            },
            attrs={
                "description": "Unified 8-variable tensor and valid-data mask for WOA23 unified field",
                "var_order": ",".join(variable_map_df["var_name"].tolist()),
            },
        )
        ds_out.to_netcdf(out_dir / "dataset_8vars.nc")

    print("[DONE] Saved outputs:")
    print(f"  - {out_dir / 'dataset_8vars.npy'}")
    print(f"  - {out_dir / 'mask_8vars.npy'}")
    print(f"  - {out_dir / 'variable_dimension_mapping.csv'}")
    print(f"  - {out_dir / 'missing_rate_stats.csv'}")
    print(f"  - {out_dir / 'depth_levels.npy'}")
    print(f"  - {out_dir / 'lat_lon_range.json'}")
    print(f"  - {out_dir / 'unified_field_data_validation_report.md'}")
    print(f"[DONE] Unified tensor shape: {X.shape}; mask shape: {M.shape}")


if __name__ == "__main__":
    main()





