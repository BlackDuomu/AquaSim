from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


REGION_NAMES = {
    1: "Nearshore",
    2: "Continental shelf",
    3: "Offshore",
}


VARIABLE_LABELS = {
    "A_an": "AOU",
    "o_an": "DO",
    "n_an": "NO3",
    "p_an": "PO4",
}


@dataclass(frozen=True)
class ReconstructionFigurePaths:
    reconstruction_npz: Path
    topo_runtime_dir: Path
    china_patch_dir: Path
    out_dir: Path


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 11,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
        }
    )
    return plt


def _load_reconstruction(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"missing reconstruction artifact: {path}")
    with np.load(path, allow_pickle=False) as payload:
        return {k: payload[k] for k in payload.files}


def _china_global_indices(china_patch_dir: Path) -> Tuple[slice, slice]:
    info_path = china_patch_dir / "china_region_info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"missing China region info: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    idx = info["index_range_global"]
    return slice(int(idx["lat_start"]), int(idx["lat_end"]) + 1), slice(int(idx["lon_start"]), int(idx["lon_end"]) + 1)


def _load_topo_china(paths: ReconstructionFigurePaths) -> Dict[str, np.ndarray]:
    arrays = paths.topo_runtime_dir / "arrays"
    lat_slice, lon_slice = _china_global_indices(paths.china_patch_dir)
    required = {
        "bottom_depth": arrays / "bottom_depth.npy",
        "slope": arrays / "slope.npy",
        "roughness": arrays / "roughness.npy",
    }
    out: Dict[str, np.ndarray] = {}
    for key, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"missing topo array: {path}")
        out[key] = np.load(path, allow_pickle=False)[lat_slice, lon_slice].astype(np.float32)
    return out


def _var_index(variable_names: np.ndarray, name: str) -> int:
    names = [str(v) for v in variable_names.tolist()]
    if name not in names:
        raise KeyError(f"variable `{name}` not found in reconstruction artifact; available={names}")
    return int(names.index(name))


def _layer_thickness(depth: np.ndarray, max_depth: float) -> Tuple[np.ndarray, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float64)
    use = depth <= float(max_depth) + 1e-6
    z = depth[use]
    if z.size < 2:
        raise ValueError(f"at least two depth levels are required up to {max_depth} m")
    edges = np.empty(z.size + 1, dtype=np.float64)
    edges[1:-1] = 0.5 * (z[:-1] + z[1:])
    edges[0] = max(0.0, z[0] - 0.5 * (z[1] - z[0]))
    edges[-1] = min(float(max_depth), z[-1] + 0.5 * (z[-1] - z[-2]))
    thickness = np.maximum(0.0, np.diff(edges))
    return use, thickness


def _compute_integrated_aou(aou: np.ndarray, depth: np.ndarray, wet: np.ndarray, max_depth: float = 1000.0) -> np.ndarray:
    use, thickness = _layer_thickness(depth, max_depth=max_depth)
    a = np.where(wet[use], aou[use], np.nan)
    a = np.where(np.isfinite(a), np.maximum(a, 0.0), np.nan)
    weighted = a * thickness[:, None, None]
    valid = np.isfinite(weighted)
    out = np.nansum(weighted, axis=0)
    out[np.sum(valid, axis=0) == 0] = np.nan
    return out.astype(np.float32)


def _compute_z50(aou: np.ndarray, depth: np.ndarray, wet: np.ndarray, max_depth: float = 1000.0) -> np.ndarray:
    use, thickness = _layer_thickness(depth, max_depth=max_depth)
    z = depth[use].astype(np.float64)
    a = np.where(wet[use], aou[use], np.nan)
    a = np.where(np.isfinite(a), np.maximum(a, 0.0), np.nan)
    increments = np.where(np.isfinite(a), a * thickness[:, None, None], 0.0)
    total = np.sum(increments, axis=0)
    csum = np.cumsum(increments, axis=0)
    target = 0.5 * total
    out = np.full(total.shape, np.nan, dtype=np.float32)
    valid_cols = total > 0.0
    h, w = total.shape
    for i in range(h):
        for j in range(w):
            if not valid_cols[i, j]:
                continue
            k = int(np.searchsorted(csum[:, i, j], target[i, j], side="left"))
            if k <= 0:
                out[i, j] = float(z[0])
            elif k >= z.size:
                out[i, j] = float(z[-1])
            else:
                prev_c = csum[k - 1, i, j]
                this_c = csum[k, i, j]
                if this_c <= prev_c:
                    out[i, j] = float(z[k])
                else:
                    frac = float((target[i, j] - prev_c) / (this_c - prev_c))
                    out[i, j] = float(z[k - 1] + frac * (z[k] - z[k - 1]))
    return out


def _region_labels(bottom_depth: np.ndarray, wet2d: np.ndarray) -> np.ndarray:
    labels = np.zeros(bottom_depth.shape, dtype=np.uint8)
    ocean = wet2d & np.isfinite(bottom_depth) & (bottom_depth > 0.0)
    labels[ocean & (bottom_depth <= 50.0)] = 1
    labels[ocean & (bottom_depth > 50.0) & (bottom_depth <= 200.0)] = 2
    labels[ocean & (bottom_depth > 200.0)] = 3
    return labels


def _summary_stats(values: np.ndarray) -> Dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan}
    q25, q75 = np.percentile(x, [25, 75])
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
    }


def _rankdata_average(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i + 1
        while j < x.size and sorted_x[j] == sorted_x[i]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _kruskal_wallis(groups: Iterable[np.ndarray]) -> float:
    clean = [np.asarray(g, dtype=np.float64)[np.isfinite(g)] for g in groups]
    clean = [g for g in clean if g.size > 0]
    if len(clean) < 2:
        return np.nan
    all_values = np.concatenate(clean)
    n = all_values.size
    ranks = _rankdata_average(all_values)
    start = 0
    h_stat = 0.0
    for g in clean:
        ni = g.size
        rbar = float(np.mean(ranks[start : start + ni]))
        h_stat += ni * (rbar - (n + 1.0) / 2.0) ** 2
        start += ni
    h_stat *= 12.0 / (n * (n + 1.0))
    return float(h_stat)


def _percentile_score(x: np.ndarray, higher_is_priority: bool = True) -> np.ndarray:
    flat = x[np.isfinite(x)]
    out = np.full(x.shape, np.nan, dtype=np.float32)
    if flat.size == 0:
        return out
    order = np.argsort(flat)
    ranks = np.empty(flat.size, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    if not higher_is_priority:
        ranks = 1.0 - ranks
    out[np.isfinite(x)] = ranks
    return out


def _area_mean_0_200(field: np.ndarray, depth: np.ndarray, wet: np.ndarray) -> np.ndarray:
    use, thickness = _layer_thickness(depth, max_depth=200.0)
    f = np.where(wet[use], field[use], np.nan)
    valid = np.isfinite(f)
    w = thickness[:, None, None]
    numerator = np.nansum(np.where(valid, f, 0.0) * w, axis=0)
    denominator = np.sum(valid * w, axis=0)
    out = numerator / np.where(denominator > 0.0, denominator, np.nan)
    return out.astype(np.float32)


def _profile_by_region(field: np.ndarray, wet: np.ndarray, region: np.ndarray, region_code: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask2d = region == int(region_code)
    vals = np.where(wet & mask2d[None, :, :], field, np.nan)
    n_depth = vals.shape[0]
    mean = np.full((n_depth,), np.nan, dtype=np.float32)
    lo = np.full((n_depth,), np.nan, dtype=np.float32)
    hi = np.full((n_depth,), np.nan, dtype=np.float32)
    rng = np.random.default_rng(20260428 + int(region_code))
    for d in range(n_depth):
        x = vals[d][np.isfinite(vals[d])]
        if x.size > 0:
            mean[d] = float(np.mean(x))
        if x.size < 4:
            continue
        boot_idx = rng.integers(0, x.size, size=(300, x.size))
        boot_mean = np.mean(x[boot_idx], axis=1)
        lo[d], hi[d] = np.percentile(boot_mean, [2.5, 97.5])
    return mean.astype(np.float32), lo, hi


def _plot_map(ax, lon: np.ndarray, lat: np.ndarray, data: np.ndarray, title: str, cmap: str, label: str, vmin=None, vmax=None):
    plt = _import_matplotlib()
    mesh = ax.pcolormesh(lon, lat, data, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Longitude (degrees east)")
    ax.set_ylabel("Latitude (degrees north)")
    ax.set_xlim(float(np.nanmin(lon)), float(np.nanmax(lon)))
    ax.set_ylim(float(np.nanmin(lat)), float(np.nanmax(lat)))
    cbar = plt.colorbar(mesh, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(label)
    return mesh


def _boxplot(ax, grouped: List[np.ndarray], labels: List[str], ylabel: str, title: str, color: str):
    finite_groups = [g[np.isfinite(g)] for g in grouped]
    bp = ax.boxplot(finite_groups, labels=labels, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set(facecolor=color, alpha=0.55, edgecolor="#222222")
    for med in bp["medians"]:
        med.set(color="#111111", linewidth=1.5)
    rng = np.random.default_rng(20260428)
    for idx, g in enumerate(finite_groups, start=1):
        if g.size == 0:
            continue
        sample = g if g.size <= 500 else rng.choice(g, size=500, replace=False)
        jitter = rng.normal(loc=idx, scale=0.045, size=sample.size)
        ax.scatter(jitter, sample, s=4, alpha=0.18, color="#222222", linewidths=0)
    h_stat = _kruskal_wallis(finite_groups)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    if np.isfinite(h_stat):
        ax.text(0.02, 0.98, f"Kruskal-Wallis H = {h_stat:.2f}", transform=ax.transAxes, ha="left", va="top")


def generate_reconstruction_figures(paths: ReconstructionFigurePaths) -> Dict[str, Path]:
    plt = _import_matplotlib()
    paths.out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load_reconstruction(paths.reconstruction_npz)
    topo = _load_topo_china(paths)
    field = payload["reconstructed_field"].astype(np.float32)
    wet = payload["wet_mask"].astype(bool)
    depth = payload["depth"].astype(np.float32)
    lat = payload["lat"].astype(np.float32)
    lon = payload["lon"].astype(np.float32)
    variable_names = payload["variable_names"]

    aou = field[_var_index(variable_names, "A_an")]
    do = field[_var_index(variable_names, "o_an")]
    no3 = field[_var_index(variable_names, "n_an")]
    po4 = field[_var_index(variable_names, "p_an")]

    wet2d = np.any(wet, axis=0)
    region = _region_labels(topo["bottom_depth"], wet2d)
    z50 = _compute_z50(aou, depth, wet, max_depth=1000.0)
    iaou = _compute_integrated_aou(aou, depth, wet, max_depth=1000.0)

    region_rows: List[Dict[str, object]] = []
    for code, name in REGION_NAMES.items():
        sel = region == code
        for metric_name, metric_values, unit in [
            ("Z50", z50, "m"),
            ("Integrated AOU 0-1000 m", iaou, "umol kg-1 m"),
        ]:
            stats = _summary_stats(metric_values[sel])
            region_rows.append({"region": name, "metric": metric_name, "unit": unit, **stats})
    pd.DataFrame(region_rows).to_csv(paths.out_dir / "regional_proxy_indicator_statistics.csv", index=False)

    outputs: Dict[str, Path] = {}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    _plot_map(axes[0], lon, lat, z50, "Reconstructed Z50 Distribution in China Coastal Seas", "viridis_r", "Z50 (m)")
    finite_z = z50[np.isfinite(z50)]
    if finite_z.size:
        levels = np.nanpercentile(finite_z, [25, 50, 75])
        axes[0].contour(lon, lat, z50, levels=np.unique(levels), colors="black", linewidths=0.45, alpha=0.55)
    z_groups = [z50[region == code] for code in REGION_NAMES]
    z_labels = [f"{REGION_NAMES[c]}\n(n={np.isfinite(z50[region == c]).sum()})" for c in REGION_NAMES]
    _boxplot(axes[1], z_groups, z_labels, "Z50 (m)", "Regional Distribution of Z50", "#4C78A8")
    out = paths.out_dir / "figure_1_z50_spatial_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs["figure_1_z50"] = out

    q20, q80 = np.nanpercentile(iaou, [20, 80])
    classes = np.full(iaou.shape, np.nan, dtype=np.float32)
    classes[np.isfinite(iaou) & (iaou <= q20)] = 0
    classes[np.isfinite(iaou) & (iaou > q20) & (iaou < q80)] = 1
    classes[np.isfinite(iaou) & (iaou >= q80)] = 2
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    _plot_map(axes[0], lon, lat, iaou, "Integrated AOU of the Reconstructed Water Column", "magma", "0-1000 m AOU integral (umol kg-1 m)")
    axes[0].contour(lon, lat, iaou, levels=[q80], colors="cyan", linewidths=0.9)
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#4C78A8", "#D9D9D9", "#E45756"])
    mesh = axes[1].pcolormesh(lon, lat, classes, shading="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    axes[1].set_title("Quantile-Based Integrated AOU Classes")
    axes[1].set_xlabel("Longitude (degrees east)")
    axes[1].set_ylabel("Latitude (degrees north)")
    cbar = plt.colorbar(mesh, ax=axes[1], ticks=[0, 1, 2], fraction=0.046, pad=0.03)
    cbar.ax.set_yticklabels(["Low (<=20th pct.)", "Intermediate", "High (>=80th pct.)"])
    out = paths.out_dir / "figure_2_integrated_aou_hotspots.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs["figure_2_integrated_aou"] = out

    env_metrics = {
        "AOU": _area_mean_0_200(aou, depth, wet),
        "DO": _area_mean_0_200(do, depth, wet),
        "NO3": _area_mean_0_200(no3, depth, wet),
        "PO4": _area_mean_0_200(po4, depth, wet),
    }
    env_rows: List[Dict[str, object]] = []
    for metric_name, values in env_metrics.items():
        for code, name in REGION_NAMES.items():
            env_rows.append({"region": name, "variable": metric_name, "unit": "umol kg-1", **_summary_stats(values[region == code])})
    pd.DataFrame(env_rows).to_csv(paths.out_dir / "regional_environmental_statistics_0_200m.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.4), constrained_layout=True)
    colors = {"AOU": "#E45756", "DO": "#4C78A8", "NO3": "#54A24B", "PO4": "#B279A2"}
    for ax, (name, values) in zip(axes.ravel(), env_metrics.items()):
        groups = [values[region == code] for code in REGION_NAMES]
        labels = [f"{REGION_NAMES[c]}\n(n={np.isfinite(values[region == c]).sum()})" for c in REGION_NAMES]
        _boxplot(ax, groups, labels, "0-200 m mean concentration (umol kg-1)", f"{name} Regional Distribution", colors[name])
    out = paths.out_dir / "figure_3_environmental_region_distributions.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs["figure_3_environmental_distributions"] = out

    profile_vars = [("AOU", aou, "#E45756"), ("DO", do, "#4C78A8"), ("NO3", no3, "#54A24B")]
    use_depth = depth <= 1000.0 + 1e-6
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0), sharey=True, constrained_layout=True)
    region_colors = {1: "#4C78A8", 2: "#F58518", 3: "#54A24B"}
    for ax, (var_name, values, _) in zip(axes, profile_vars):
        for code, region_name in REGION_NAMES.items():
            mean, lo, hi = _profile_by_region(values, wet, region, code)
            ax.plot(mean[use_depth], depth[use_depth], color=region_colors[code], linewidth=1.6, label=region_name)
            ax.fill_betweenx(depth[use_depth], lo[use_depth], hi[use_depth], color=region_colors[code], alpha=0.16, linewidth=0)
        ax.invert_yaxis()
        ax.set_title(f"{var_name} Vertical Profile")
        ax.set_xlabel("Concentration (umol kg-1)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Depth (m)")
    axes[-1].legend(loc="lower right", frameon=False)
    out = paths.out_dir / "figure_4_vertical_profiles_confidence_bands.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs["figure_4_vertical_profiles"] = out

    do_upper = env_metrics["DO"]
    nutrient = 0.5 * _percentile_score(env_metrics["NO3"], True) + 0.5 * _percentile_score(env_metrics["PO4"], True)
    heterogeneity = 0.5 * _percentile_score(topo["slope"], True) + 0.5 * _percentile_score(topo["roughness"], True)
    score = (
        0.25 * _percentile_score(z50, True)
        + 0.30 * _percentile_score(iaou, True)
        + 0.20 * _percentile_score(do_upper, False)
        + 0.15 * nutrient
        + 0.10 * heterogeneity
    )
    score[~wet2d] = np.nan
    s80, s90 = np.nanpercentile(score, [80, 90])
    priority = np.full(score.shape, np.nan, dtype=np.float32)
    priority[np.isfinite(score)] = 0
    priority[np.isfinite(score) & (score >= s80)] = 1
    priority[np.isfinite(score) & (score >= s90)] = 2
    priority_rows = []
    for cls, name in [(0, "General marine area (<80th percentile)"), (1, "Priority area (80th-90th percentile)"), (2, "Core priority area (>=90th percentile)")]:
        priority_rows.append({"priority_class": name, "grid_count": int(np.sum(priority == cls)), "score_threshold": float(s80 if cls == 1 else s90 if cls == 2 else np.nan)})
    pd.DataFrame(priority_rows).to_csv(paths.out_dir / "priority_area_classification_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    _plot_map(axes[0], lon, lat, score, "Composite Priority Score Based on Reconstructed Indicators", "plasma", "Composite score")
    axes[0].contour(lon, lat, score, levels=[s80, s90], colors=["white", "black"], linewidths=[0.8, 1.0])
    cmap = ListedColormap(["#D9D9D9", "#F58518", "#B22222"])
    mesh = axes[1].pcolormesh(lon, lat, priority, shading="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
    axes[1].set_title("Priority Marine Areas Identified by Quantile Thresholds")
    axes[1].set_xlabel("Longitude (degrees east)")
    axes[1].set_ylabel("Latitude (degrees north)")
    cbar = plt.colorbar(mesh, ax=axes[1], ticks=[0, 1, 2], fraction=0.046, pad=0.03)
    cbar.ax.set_yticklabels(["General (<80th pct.)", "Priority (80th-90th pct.)", "Core priority (>=90th pct.)"])
    out = paths.out_dir / "figure_5_priority_marine_areas.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    outputs["figure_5_priority_areas"] = out

    np.savez_compressed(
        paths.out_dir / "derived_reconstruction_indicators.npz",
        z50=z50,
        integrated_aou_0_1000m=iaou,
        region=region,
        priority_score=score.astype(np.float32),
        priority_class=priority.astype(np.float32),
        lat=lat,
        lon=lon,
    )
    return outputs
