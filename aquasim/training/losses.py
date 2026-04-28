
#!/usr/bin/env python
"""Step6 losses with optional final physics-informed framework.

Default behaviour remains Step6 v1 masked supervise loss when physics branch is off.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


def _cfg_bool(cfg: Mapping[str, Any], key: str, default: bool) -> bool:
    v = cfg.get(key, default)
    return bool(v)


def _cfg_float(cfg: Mapping[str, Any], key: str, default: float) -> float:
    v = cfg.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _cfg_str(cfg: Mapping[str, Any], key: str, default: str) -> str:
    v = cfg.get(key, default)
    return str(v) if v is not None else str(default)


def _zero_like(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _safe_mask_denominator(mask: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    return torch.clamp(mask.sum(), min=float(eps))


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float,
    zero: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if values.numel() == 0:
        return zero
    mask_f = mask.to(dtype=values.dtype)
    if weights is not None:
        mask_f = mask_f * weights.to(dtype=values.dtype)
    if float(mask_f.sum().detach().item()) <= 0.0:
        # No valid sample for this term in this batch -> return 0 by design.
        return zero
    den = torch.clamp(mask_f.sum(), min=float(eps))
    return (values * mask_f).sum() / den


def _robust_elementwise(x: torch.Tensor, loss_type: str, beta: float = 1.0) -> torch.Tensor:
    lt = str(loss_type).strip().lower()
    if lt in {"smooth_l1", "huber"}:
        b = max(float(beta), 1e-6)
        ax = torch.abs(x)
        return torch.where(ax < b, 0.5 * (x ** 2) / b, ax - 0.5 * b)
    if lt == "mse":
        return x ** 2
    if lt in {"l1", "mae"}:
        return torch.abs(x)
    ax = torch.abs(x)
    b = 1.0
    return torch.where(ax < b, 0.5 * (x ** 2) / b, ax - 0.5 * b)


def _sanitize_tensor(x: torch.Tensor, safe_nan_to_num: bool, clamp_value: float) -> torch.Tensor:
    if safe_nan_to_num:
        x = torch.nan_to_num(
            x,
            nan=0.0,
            posinf=float(clamp_value if clamp_value > 0 else 1e6),
            neginf=-float(clamp_value if clamp_value > 0 else 1e6),
        )
    if clamp_value > 0:
        x = torch.clamp(x, min=-float(clamp_value), max=float(clamp_value))
    return x


def masked_mse(pred: torch.Tensor, target: torch.Tensor, supervise_mask: torch.Tensor) -> torch.Tensor:
    diff = torch.where(supervise_mask > 0, pred - target, torch.zeros_like(pred))
    diff2 = diff ** 2
    den = _safe_mask_denominator(supervise_mask)
    return (diff2 * supervise_mask).sum() / den


def masked_mae(pred: torch.Tensor, target: torch.Tensor, supervise_mask: torch.Tensor) -> torch.Tensor:
    diff = torch.where(supervise_mask > 0, torch.abs(pred - target), torch.zeros_like(pred))
    den = _safe_mask_denominator(supervise_mask)
    return (diff * supervise_mask).sum() / den


def resolve_variable_indices(variable_names: Sequence[str]) -> Dict[str, Optional[int]]:
    """Resolve canonical variable indices used by physics loss terms.

    Returns keys: idx_t, idx_s, idx_o, idx_aou, idx_n, idx_p, idx_o2sat.
    """

    aliases = {
        "idx_t": ("t_an", "t", "temperature", "temp"),
        "idx_s": ("s_an", "s", "salinity"),
        "idx_o": ("o_an", "o", "o2", "oxygen", "dissolved_oxygen"),
        "idx_aou": ("a_an", "aou", "A_an"),
        "idx_n": ("n_an", "n", "no3", "nitrate"),
        "idx_p": ("p_an", "p", "po4", "phosphate"),
        "idx_o2sat": ("o2sat_csv", "o2sat", "o2_sat", "oxygen_saturation"),
    }
    lower_to_idx = {str(v).strip().lower(): i for i, v in enumerate(variable_names)}

    out: Dict[str, Optional[int]] = {}
    for key, name_aliases in aliases.items():
        hit: Optional[int] = None
        for nm in name_aliases:
            q = str(nm).strip().lower()
            if q in lower_to_idx:
                hit = int(lower_to_idx[q])
                break
        out[key] = hit

    return out


def denormalize_for_physics(
    pred: torch.Tensor,
    norm_stats: Optional[Mapping[str, torch.Tensor]],
    physics_eps: float,
) -> torch.Tensor:
    """Convert normalized prediction back to physical space for physics losses."""
    if norm_stats is None:
        raise ValueError("Physics loss requires normalization stats for denormalization.")
    if "mean" not in norm_stats or "std" not in norm_stats:
        raise KeyError("norm_stats must contain `mean` and `std`.")

    mean = norm_stats["mean"].to(device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1, 1)
    std = norm_stats["std"].to(device=pred.device, dtype=pred.dtype).view(1, -1, 1, 1, 1)
    std = torch.clamp(std, min=float(physics_eps))
    return pred * std + mean


def _build_depth_context(
    patch_depth_m: Optional[torch.Tensor],
    batch_size: int,
    depth_len: int,
    device: torch.device,
    dtype: torch.dtype,
    physics_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, str]:
    if patch_depth_m is None:
        depth = torch.arange(depth_len, device=device, dtype=dtype).view(1, depth_len).repeat(batch_size, 1)
        spacing_mode = "index_spacing_proxy"
    else:
        depth = patch_depth_m.to(device=device, dtype=dtype)
        if depth.ndim == 1:
            depth = depth.view(1, -1).repeat(batch_size, 1)
        if int(depth.shape[1]) != int(depth_len):
            depth = torch.arange(depth_len, device=device, dtype=dtype).view(1, depth_len).repeat(batch_size, 1)
            spacing_mode = "index_spacing_proxy"
        else:
            spacing_mode = "physical_depth_spacing"
    dz = torch.abs(depth[:, 1:] - depth[:, :-1])
    dz = torch.clamp(dz, min=float(physics_eps))
    return depth, dz, spacing_mode


def _build_topo_context(
    topo_runtime: Optional[Mapping[str, torch.Tensor]],
    batch_size: int,
    depth_len: int,
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    def _bool_default(default: bool) -> torch.Tensor:
        v = 1.0 if default else 0.0
        return torch.full((batch_size, depth_len, h, w), v, dtype=dtype, device=device) > 0.5

    def _float_default(default: float) -> torch.Tensor:
        return torch.full((batch_size, depth_len, h, w), float(default), dtype=dtype, device=device)

    if topo_runtime is None:
        return {
            "wet_mask": _bool_default(True),
            "near_bottom_mask": _bool_default(False),
            "slope_steepness": _float_default(0.0),
            "roughness_weight": _float_default(0.0),
            "enabled": torch.tensor(0.0, dtype=dtype, device=device),
        }

    def _to_dhw(x: Optional[torch.Tensor], fill: float, bool_mode: bool = False) -> torch.Tensor:
        if x is None:
            base = _float_default(fill)
            return base > 0.5 if bool_mode else base
        t = x.to(device=device, dtype=dtype)
        if t.ndim == 5 and int(t.shape[1]) == 1:
            t = t[:, 0]
        if t.ndim == 3:
            t = t.unsqueeze(0).expand(batch_size, -1, -1, -1)
        if t.ndim != 4:
            base = _float_default(fill)
            return base > 0.5 if bool_mode else base
        td, th, tw = int(t.shape[1]), int(t.shape[2]), int(t.shape[3])
        out = _float_default(fill)
        dd = min(depth_len, td)
        hh = min(h, th)
        ww = min(w, tw)
        out[:, :dd, :hh, :ww] = t[:, :dd, :hh, :ww]
        return out > 0.5 if bool_mode else out

    wet = _to_dhw(topo_runtime.get("wet_mask"), fill=1.0, bool_mode=True)
    near = _to_dhw(topo_runtime.get("near_bottom_mask"), fill=0.0, bool_mode=True) & wet
    steep = torch.clamp(_to_dhw(topo_runtime.get("slope_weight"), fill=0.0, bool_mode=False), min=0.0, max=1.0)
    rough = torch.clamp(_to_dhw(topo_runtime.get("roughness_weight"), fill=0.0, bool_mode=False), min=0.0, max=1.0)

    return {
        "wet_mask": wet,
        "near_bottom_mask": near,
        "slope_steepness": steep,
        "roughness_weight": rough,
        "enabled": torch.tensor(1.0, dtype=dtype, device=device),
    }


def _build_slope_modulation(
    slope_steepness: torch.Tensor,
    cfg: Mapping[str, Any],
) -> torch.Tensor:
    slope_relax = _cfg_float(cfg, "topo_slope_relax_strength", 0.5)
    flat_enhance = _cfg_float(cfg, "topo_flat_enhance_strength", 0.25)
    min_mod = _cfg_float(cfg, "topo_slope_modulation_min", 0.2)
    mod = 1.0 + float(flat_enhance) * (1.0 - slope_steepness) - float(slope_relax) * slope_steepness
    return torch.clamp(mod, min=float(min_mod))


def _density_proxy(temp: torch.Tensor, sal: torch.Tensor, cfg: Mapping[str, Any]) -> torch.Tensor:
    # Lightweight linearized EOS proxy to avoid heavy dependencies.
    ref_t = _cfg_float(cfg, "density_ref_temperature", 10.0)
    ref_s = _cfg_float(cfg, "density_ref_salinity", 35.0)
    coeff_t = _cfg_float(cfg, "density_coeff_temperature", 0.2)
    coeff_s = _cfg_float(cfg, "density_coeff_salinity", 0.8)
    rho0 = _cfg_float(cfg, "density_rho0", 1027.0)
    return rho0 + coeff_s * (sal - ref_s) - coeff_t * (temp - ref_t)

def _build_region_groups(
    region_labels: Optional[torch.Tensor],
    cfg: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    # MarineSim official route does not use region weighting.
    # Keep one global group regardless of metadata availability.
    return {"global": torch.ones((batch_size,), device=device, dtype=torch.bool)}


def _region_lambdas(region_name: str, cfg: Mapping[str, Any]) -> Dict[str, float]:
    base = {
        "lambda_density": _cfg_float(cfg, "lambda_density", 0.0),
        "lambda_vert": _cfg_float(cfg, "lambda_vert", 0.0),
        "lambda_aou_o2": _cfg_float(cfg, "lambda_aou_o2", 0.0),
        "lambda_np": _cfg_float(cfg, "lambda_np", 0.0),
        "lambda_remin": _cfg_float(cfg, "lambda_remin", 0.0),
        "lambda_tv": _cfg_float(cfg, "lambda_tv", 0.0),
        "lambda_topo": _cfg_float(cfg, "lambda_topo", 0.0),
    }
    return base


def _loss_scale_dict(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    raw = cfg.get("loss_scales", {})
    if isinstance(raw, Mapping):
        return raw
    return {}


def _loss_scale_enabled(cfg: Mapping[str, Any]) -> bool:
    return _cfg_bool(cfg, "use_loss_calibration", False)


def _safe_scale_value(scale: float, eps: float) -> float:
    return max(float(scale), float(eps))


def _resolve_loss_scale(cfg: Mapping[str, Any], key: str, eps: float) -> float:
    scales = _loss_scale_dict(cfg)
    default = 1.0
    if key in scales and isinstance(scales.get(key), Mapping):
        nested = scales.get(key, {})
        if isinstance(nested, Mapping):
            value = nested.get("scale", default)
        else:
            value = default
    else:
        value = scales.get(key, default)
    return _safe_scale_value(_cfg_float({"v": value}, "v", default), eps=eps)


def _compute_density_loss(
    pred_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    idx_t: Optional[int],
    idx_s: Optional[int],
    dz: torch.Tensor,
    cfg: Mapping[str, Any],
    topo_context: Optional[Mapping[str, torch.Tensor]],
    zero: torch.Tensor,
) -> torch.Tensor:
    if idx_t is None or idx_s is None:
        return zero
    if pred_phys.shape[2] < 2:
        return zero

    safe_nan_to_num = _cfg_bool(cfg, "safe_nan_to_num", True)
    clamp_value = _cfg_float(cfg, "gradient_clamp_value", 1e4)
    physics_eps = _cfg_float(cfg, "physics_eps", 1e-6)

    t = pred_phys[:, idx_t]
    s = pred_phys[:, idx_s]
    rho = _density_proxy(t, s, cfg)
    drho_dz = (rho[:, 1:] - rho[:, :-1]) / dz.unsqueeze(-1).unsqueeze(-1)
    drho_dz = _sanitize_tensor(drho_dz, safe_nan_to_num=safe_nan_to_num, clamp_value=clamp_value)

    margin = _cfg_float(cfg, "density_inversion_margin", 0.0)
    penalty_input = -drho_dz + margin
    mode = _cfg_str(cfg, "density_penalty_mode", "relu").lower()
    penalty = F.softplus(penalty_input) if mode == "softplus" else F.relu(penalty_input)

    mask = (
        valid_mask[:, idx_t, 1:]
        & valid_mask[:, idx_t, :-1]
        & valid_mask[:, idx_s, 1:]
        & valid_mask[:, idx_s, :-1]
    )
    weights = None
    if topo_context is not None:
        near = topo_context["near_bottom_mask"][:, 1:] | topo_context["near_bottom_mask"][:, :-1]
        near_boost = _cfg_float(cfg, "topo_near_bottom_boost", 0.8)
        weights = 1.0 + float(near_boost) * near.to(dtype=penalty.dtype)
    return _masked_mean(penalty, mask=mask, eps=physics_eps, zero=zero, weights=weights)


def _compute_vert_loss(
    pred_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    idx_t: Optional[int],
    idx_s: Optional[int],
    idx_aou: Optional[int],
    dz: torch.Tensor,
    cfg: Mapping[str, Any],
    topo_context: Optional[Mapping[str, torch.Tensor]],
    zero: torch.Tensor,
) -> torch.Tensor:
    if pred_phys.shape[2] < 3:
        return zero

    safe_nan_to_num = _cfg_bool(cfg, "safe_nan_to_num", True)
    clamp_value = _cfg_float(cfg, "gradient_clamp_value", 1e4)
    physics_eps = _cfg_float(cfg, "physics_eps", 1e-6)
    structure_eps = _cfg_float(cfg, "vert_structure_eps", 1e-3)
    structure_mode = _cfg_str(cfg, "vert_structure_mode", "density_gradient_inverse").lower()
    loss_type = _cfg_str(cfg, "vert_loss_type", "smooth_l1")
    beta = _cfg_float(cfg, "vert_huber_beta", 1.0)

    if idx_t is not None and idx_s is not None and pred_phys.shape[2] >= 2:
        rho = _density_proxy(pred_phys[:, idx_t], pred_phys[:, idx_s], cfg)
        rho_grad = (rho[:, 1:] - rho[:, :-1]) / dz.unsqueeze(-1).unsqueeze(-1)
        rho_grad = _sanitize_tensor(rho_grad, safe_nan_to_num=safe_nan_to_num, clamp_value=clamp_value)
        strat_center = 0.5 * (torch.abs(rho_grad[:, 1:]) + torch.abs(rho_grad[:, :-1]))
    else:
        strat_center = torch.ones(
            (pred_phys.shape[0], pred_phys.shape[2] - 2, pred_phys.shape[3], pred_phys.shape[4]),
            device=pred_phys.device,
            dtype=pred_phys.dtype,
        )

    if structure_mode == "none":
        g = torch.ones_like(strat_center)
    elif structure_mode == "n2_proxy_inverse":
        g = 1.0 / (strat_center ** 2 + float(structure_eps))
    elif structure_mode == "density_gradient_exp":
        scale = torch.clamp(strat_center.detach().mean(), min=float(structure_eps))
        g = torch.exp(-strat_center / scale)
    else:
        g = 1.0 / (strat_center + float(structure_eps))
    g = g / torch.clamp(g.detach().mean(), min=float(structure_eps))

    dz_center = 0.5 * (dz[:, 1:] + dz[:, :-1])
    dz2 = torch.clamp(dz_center.unsqueeze(-1).unsqueeze(-1) ** 2, min=float(physics_eps))

    total = zero
    active = 0
    var_specs = [
        (idx_t, _cfg_float(cfg, "smooth_weight_t", 1.0)),
        (idx_s, _cfg_float(cfg, "smooth_weight_s", 1.0)),
        (idx_aou, _cfg_float(cfg, "smooth_weight_aou", 1.0)),
    ]
    for idx, w in var_specs:
        if idx is None or w <= 0:
            continue
        x = pred_phys[:, idx]
        d2 = (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]) / dz2
        d2 = _sanitize_tensor(d2, safe_nan_to_num=safe_nan_to_num, clamp_value=clamp_value)
        elem = _robust_elementwise(d2, loss_type=loss_type, beta=beta)
        mask = valid_mask[:, idx, 2:] & valid_mask[:, idx, 1:-1] & valid_mask[:, idx, :-2]
        topo_w = None
        if topo_context is not None:
            near_mid = topo_context["near_bottom_mask"][:, 1:-1].to(dtype=elem.dtype)
            slope_mod = _build_slope_modulation(topo_context["slope_steepness"][:, 1:-1], cfg=cfg).to(dtype=elem.dtype)
            near_boost = _cfg_float(cfg, "topo_near_bottom_boost", 0.8)
            topo_w = (1.0 + float(near_boost) * near_mid) * slope_mod
        total = total + float(w) * _masked_mean(elem * g, mask=mask, eps=physics_eps, zero=zero, weights=topo_w)
        active += 1
    return total if active > 0 else zero


def _compute_aou_o2_loss(
    pred_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    idx_aou: Optional[int],
    idx_o: Optional[int],
    idx_o2sat: Optional[int],
    idx_t: Optional[int],
    idx_s: Optional[int],
    cfg: Mapping[str, Any],
    zero: torch.Tensor,
) -> torch.Tensor:
    if idx_aou is None or idx_o is None:
        return zero

    physics_eps = _cfg_float(cfg, "physics_eps", 1e-6)
    aou_eps = _cfg_float(cfg, "aou_o2_eps", 1e-6)
    loss_type = _cfg_str(cfg, "aou_o2_loss_type", "smooth_l1")
    beta = _cfg_float(cfg, "aou_o2_huber_beta", 1.0)
    fallback_mode = _cfg_str(cfg, "aou_o2_fallback_mode", "proxy_ts").lower()

    aou = pred_phys[:, idx_aou]
    o2 = pred_phys[:, idx_o]
    valid = valid_mask[:, idx_aou] & valid_mask[:, idx_o]

    if idx_o2sat is not None:
        o2sat = pred_phys[:, idx_o2sat]
        valid = valid & valid_mask[:, idx_o2sat]
    elif fallback_mode == "proxy_ts" and idx_t is not None and idx_s is not None:
        t = pred_phys[:, idx_t]
        s = pred_phys[:, idx_s]
        o2sat = 300.0 - 7.0 * t + 0.2 * (35.0 - s)
        valid = valid & valid_mask[:, idx_t] & valid_mask[:, idx_s]
    else:
        return zero

    resid = aou - (o2sat - o2)
    resid = resid / (torch.abs(o2sat) + float(aou_eps))
    elem = _robust_elementwise(resid, loss_type=loss_type, beta=beta)
    return _masked_mean(elem, mask=valid, eps=physics_eps, zero=zero)

def _compute_np_loss(
    pred_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    idx_n: Optional[int],
    idx_p: Optional[int],
    depth_m: torch.Tensor,
    cfg: Mapping[str, Any],
    zero: torch.Tensor,
) -> torch.Tensor:
    if idx_n is None or idx_p is None:
        return zero

    physics_eps = _cfg_float(cfg, "physics_eps", 1e-6)
    np_eps = _cfg_float(cfg, "np_eps", 1e-6)
    ratio = _cfg_float(cfg, "redfield_np_ratio", 16.0)
    loss_type = _cfg_str(cfg, "np_loss_type", "smooth_l1")
    beta = _cfg_float(cfg, "np_huber_beta", 1.0)

    n = pred_phys[:, idx_n]
    p = pred_phys[:, idx_p]
    valid = valid_mask[:, idx_n] & valid_mask[:, idx_p]

    use_deep_only = _cfg_bool(cfg, "np_deep_only", True)
    deep_min_depth = _cfg_float(cfg, "np_deep_min_depth", 1000.0)
    if use_deep_only:
        deep = (depth_m.unsqueeze(-1).unsqueeze(-1) >= float(deep_min_depth)).expand_as(n)
    else:
        deep = torch.ones_like(valid)

    shallow = (~deep) & valid
    shallow_den = shallow.to(dtype=n.dtype).sum(dim=1, keepdim=True)
    n_ref = torch.where(shallow, n, torch.zeros_like(n)).sum(dim=1, keepdim=True) / torch.clamp(shallow_den, min=1.0)
    p_ref = torch.where(shallow, p, torch.zeros_like(p)).sum(dim=1, keepdim=True) / torch.clamp(shallow_den, min=1.0)
    n_ref = torch.where(shallow_den > 0, n_ref, n[:, :1])
    p_ref = torch.where(shallow_den > 0, p_ref, p[:, :1])

    dn = n - n_ref
    dp = p - p_ref
    resid = (dn - float(ratio) * dp) / (torch.abs(float(ratio) * dp) + float(np_eps))

    active = deep & valid & ((torch.abs(dn) > float(np_eps)) | (torch.abs(dp) > float(np_eps)))
    elem = _robust_elementwise(resid, loss_type=loss_type, beta=beta)
    return _masked_mean(elem, mask=active, eps=physics_eps, zero=zero)


def _compute_remin_loss(
    pred_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    idx_aou: Optional[int],
    idx_o: Optional[int],
    idx_n: Optional[int],
    idx_p: Optional[int],
    depth_m: torch.Tensor,
    dz: torch.Tensor,
    cfg: Mapping[str, Any],
    topo_context: Optional[Mapping[str, torch.Tensor]],
    zero: torch.Tensor,
) -> torch.Tensor:
    if idx_aou is None or pred_phys.shape[2] < 2:
        return zero

    physics_eps = _cfg_float(cfg, "physics_eps", 1e-6)
    remin_eps = _cfg_float(cfg, "remin_eps", 1e-6)
    margin = _cfg_float(cfg, "remin_margin", 0.0)
    clamp_value = _cfg_float(cfg, "gradient_clamp_value", 1e4)
    safe_nan_to_num = _cfg_bool(cfg, "safe_nan_to_num", True)
    remin_min_depth = _cfg_float(cfg, "remin_min_depth", 100.0)

    use_n = _cfg_bool(cfg, "remin_use_n", True)
    use_p = _cfg_bool(cfg, "remin_use_p", True)
    use_o2 = _cfg_bool(cfg, "remin_use_o2", True)

    aou = pred_phys[:, idx_aou]
    da = (aou[:, 1:] - aou[:, :-1]) / dz.unsqueeze(-1).unsqueeze(-1)
    da = _sanitize_tensor(da, safe_nan_to_num=safe_nan_to_num, clamp_value=clamp_value)
    va = valid_mask[:, idx_aou, 1:] & valid_mask[:, idx_aou, :-1]

    mid_depth = 0.5 * (depth_m[:, 1:] + depth_m[:, :-1])
    deep_ok = (mid_depth.unsqueeze(-1).unsqueeze(-1) >= float(remin_min_depth)).expand_as(da)

    penalties = []
    masks = []

    if use_n and idx_n is not None:
        dn = (pred_phys[:, idx_n, 1:] - pred_phys[:, idx_n, :-1]) / dz.unsqueeze(-1).unsqueeze(-1)
        dn = _sanitize_tensor(dn, safe_nan_to_num=safe_nan_to_num, clamp_value=clamp_value)
        vn = valid_mask[:, idx_n, 1:] & valid_mask[:, idx_n, :-1]
        mask = va & vn & deep_ok & ((torch.abs(da) > float(remin_eps)) | (torch.abs(dn) > float(remin_eps)))
        penalties.append(F.relu(-(da * dn) + float(margin)))
        masks.append(mask)

    if use_p and idx_p is not None:
        dp = (pred_phys[:, idx_p, 1:] - pred_phys[:, idx_p, :-1]) / dz.unsqueeze(-1).unsqueeze(-1)
        dp = _sanitize_tensor(dp, safe_nan_to_num=safe_nan_to_num, clamp_value=clamp_value)
        vp = valid_mask[:, idx_p, 1:] & valid_mask[:, idx_p, :-1]
        mask = va & vp & deep_ok & ((torch.abs(da) > float(remin_eps)) | (torch.abs(dp) > float(remin_eps)))
        penalties.append(F.relu(-(da * dp) + float(margin)))
        masks.append(mask)

    if use_o2 and idx_o is not None:
        do2 = (pred_phys[:, idx_o, 1:] - pred_phys[:, idx_o, :-1]) / dz.unsqueeze(-1).unsqueeze(-1)
        do2 = _sanitize_tensor(do2, safe_nan_to_num=safe_nan_to_num, clamp_value=clamp_value)
        vo2 = valid_mask[:, idx_o, 1:] & valid_mask[:, idx_o, :-1]
        mask = va & vo2 & deep_ok & ((torch.abs(da) > float(remin_eps)) | (torch.abs(do2) > float(remin_eps)))
        penalties.append(F.relu((da * do2) + float(margin)))
        masks.append(mask)

    if len(penalties) <= 0:
        return zero

    acc = zero
    eff = 0
    topo_w = None
    if topo_context is not None:
        near_pair = (topo_context["near_bottom_mask"][:, 1:] | topo_context["near_bottom_mask"][:, :-1]).to(dtype=pred_phys.dtype)
        slope_pair = 0.5 * (
            _build_slope_modulation(topo_context["slope_steepness"][:, 1:], cfg=cfg)
            + _build_slope_modulation(topo_context["slope_steepness"][:, :-1], cfg=cfg)
        )
        near_boost = _cfg_float(cfg, "topo_near_bottom_boost", 0.8)
        topo_w = (1.0 + float(near_boost) * near_pair) * slope_pair.to(dtype=pred_phys.dtype)
    for p, m in zip(penalties, masks):
        acc = acc + _masked_mean(p, mask=m, eps=physics_eps, zero=zero, weights=topo_w)
        if float(m.to(dtype=pred_phys.dtype).sum().detach().item()) > 0.0:
            eff += 1
    return acc / float(eff) if eff > 0 else zero


def _resolve_tv_indices(
    cfg: Mapping[str, Any],
    variable_indices: Mapping[str, Optional[int]],
    channel_count: int,
) -> Tuple[int, ...]:
    from_cfg = cfg.get("tv_apply_indices")
    if isinstance(from_cfg, (list, tuple)):
        idxs = [int(v) for v in from_cfg if isinstance(v, (int, float))]
    else:
        idxs = []

    if len(idxs) == 0:
        by_name = cfg.get("tv_apply_variables", [])
        if isinstance(by_name, str):
            by_name = [v.strip() for v in by_name.split(",") if str(v).strip()]
        if isinstance(by_name, (list, tuple)):
            name_to_idx = {
                "t": variable_indices.get("idx_t"),
                "s": variable_indices.get("idx_s"),
                "o": variable_indices.get("idx_o"),
                "aou": variable_indices.get("idx_aou"),
                "n": variable_indices.get("idx_n"),
                "p": variable_indices.get("idx_p"),
                "o2sat": variable_indices.get("idx_o2sat"),
            }
            lookup = {
                "t": "t",
                "t_an": "t",
                "temperature": "t",
                "s": "s",
                "s_an": "s",
                "salinity": "s",
                "o": "o",
                "o_an": "o",
                "o2": "o",
                "a": "aou",
                "a_an": "aou",
                "aou": "aou",
                "n": "n",
                "n_an": "n",
                "no3": "n",
                "p": "p",
                "p_an": "p",
                "po4": "p",
                "o2sat": "o2sat",
                "o2sat_csv": "o2sat",
            }
            for nm in by_name:
                q = str(nm).strip().lower()
                key = lookup.get(q)
                if key is None:
                    continue
                idx = name_to_idx.get(key)
                if idx is not None:
                    idxs.append(int(idx))

    if len(idxs) == 0:
        idxs = list(range(int(channel_count)))

    uniq = sorted({i for i in idxs if 0 <= int(i) < int(channel_count)})
    return tuple(int(i) for i in uniq)


def _compute_tv_loss(
    pred_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    tv_indices: Tuple[int, ...],
    dz: torch.Tensor,
    cfg: Mapping[str, Any],
    topo_context: Optional[Mapping[str, torch.Tensor]],
    zero: torch.Tensor,
) -> torch.Tensor:
    if len(tv_indices) <= 0:
        return zero

    physics_eps = _cfg_float(cfg, "physics_eps", 1e-6)
    wh = _cfg_float(cfg, "tv_weight_horizontal", 1.0)
    wv = _cfg_float(cfg, "tv_weight_vertical", 1.0)

    x = pred_phys[:, tv_indices]
    vm = valid_mask[:, tv_indices]

    horizontal = zero
    slope_mod = None
    if topo_context is not None:
        slope_mod = _build_slope_modulation(topo_context["slope_steepness"], cfg=cfg).to(dtype=x.dtype)
        near_boost = _cfg_float(cfg, "topo_near_bottom_boost", 0.8)
        slope_mod = slope_mod * (1.0 + float(near_boost) * topo_context["near_bottom_mask"].to(dtype=x.dtype))
    if x.shape[3] >= 2:
        dh = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        mh = vm[:, :, :, 1:, :] & vm[:, :, :, :-1, :]
        wh_mask = None
        if slope_mod is not None:
            wh_mask = 0.5 * (slope_mod[:, :, 1:, :] + slope_mod[:, :, :-1, :]).unsqueeze(1)
        horizontal = horizontal + _masked_mean(torch.abs(dh), mask=mh, eps=physics_eps, zero=zero, weights=wh_mask)
    if x.shape[4] >= 2:
        dw = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        mw = vm[:, :, :, :, 1:] & vm[:, :, :, :, :-1]
        ww_mask = None
        if slope_mod is not None:
            ww_mask = 0.5 * (slope_mod[:, :, :, 1:] + slope_mod[:, :, :, :-1]).unsqueeze(1)
        horizontal = horizontal + _masked_mean(torch.abs(dw), mask=mw, eps=physics_eps, zero=zero, weights=ww_mask)
    horizontal = 0.5 * horizontal

    vertical = zero
    if x.shape[2] >= 2:
        dz_safe = dz.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
        dv = (x[:, :, 1:] - x[:, :, :-1]) / dz_safe
        mv = vm[:, :, 1:] & vm[:, :, :-1]
        wv_mask = None
        if slope_mod is not None:
            wv_mask = 0.5 * (slope_mod[:, 1:] + slope_mod[:, :-1]).unsqueeze(1)
        vertical = _masked_mean(torch.abs(dv), mask=mv, eps=physics_eps, zero=zero, weights=wv_mask)

    return float(wh) * horizontal + float(wv) * vertical


def _compute_topo_constraint_loss(
    pred_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    topo_context: Optional[Mapping[str, torch.Tensor]],
    variable_indices: Mapping[str, Optional[int]],
    dz: torch.Tensor,
    cfg: Mapping[str, Any],
    zero: torch.Tensor,
) -> torch.Tensor:
    if topo_context is None:
        return zero
    if pred_phys.shape[2] < 2:
        return zero

    physics_eps = _cfg_float(cfg, "physics_eps", 1e-6)
    boundary_w = _cfg_float(cfg, "topo_boundary_weight", 1.0)
    near_w = _cfg_float(cfg, "topo_near_bottom_struct_weight", 1.0)
    near_boost = _cfg_float(cfg, "topo_near_bottom_boost", 0.8)
    loss_type = _cfg_str(cfg, "topo_constraint_loss_type", "smooth_l1")
    beta = _cfg_float(cfg, "topo_constraint_huber_beta", 1.0)

    wet = topo_context["wet_mask"]
    near = topo_context["near_bottom_mask"]
    slope_mod = _build_slope_modulation(topo_context["slope_steepness"], cfg=cfg).to(dtype=pred_phys.dtype)

    candidate = [
        variable_indices.get("idx_t"),
        variable_indices.get("idx_s"),
        variable_indices.get("idx_aou"),
        variable_indices.get("idx_n"),
        variable_indices.get("idx_p"),
    ]
    channels = [int(i) for i in candidate if i is not None]
    if len(channels) <= 0:
        channels = list(range(int(pred_phys.shape[1])))

    x = pred_phys[:, channels]
    vm = valid_mask[:, channels]
    dz_safe = dz.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)

    boundary = wet[:, :-1] & (~wet[:, 1:])
    b_mask = boundary.unsqueeze(1) & vm[:, :, :-1] & vm[:, :, 1:]
    b_grad = (x[:, :, 1:] - x[:, :, :-1]) / dz_safe
    b_elem = _robust_elementwise(b_grad, loss_type=loss_type, beta=beta)
    b_w = 1.0 + float(near_boost) * near[:, :-1].to(dtype=pred_phys.dtype)
    boundary_loss = _masked_mean(
        b_elem,
        mask=b_mask,
        eps=physics_eps,
        zero=zero,
        weights=b_w.unsqueeze(1),
    )

    near_mid = near[:, 1:-1]
    if x.shape[2] >= 3:
        d2 = (x[:, :, 2:] - 2.0 * x[:, :, 1:-1] + x[:, :, :-2]) / torch.clamp(
            (0.5 * (dz[:, 1:] + dz[:, :-1])).unsqueeze(1).unsqueeze(-1).unsqueeze(-1) ** 2,
            min=float(physics_eps),
        )
        n_mask = near_mid.unsqueeze(1) & vm[:, :, 2:] & vm[:, :, 1:-1] & vm[:, :, :-2]
        n_elem = _robust_elementwise(d2, loss_type=loss_type, beta=beta)
        n_w = (1.0 + float(near_boost) * near_mid.to(dtype=pred_phys.dtype)) * slope_mod[:, 1:-1]
        near_loss = _masked_mean(
            n_elem,
            mask=n_mask,
            eps=physics_eps,
            zero=zero,
            weights=n_w.unsqueeze(1),
        )
    else:
        near_loss = zero

    return float(boundary_w) * boundary_loss + float(near_w) * near_loss


def compute_loss_by_region(
    pred_phys: torch.Tensor,
    valid_mask: torch.Tensor,
    region_labels: Optional[torch.Tensor],
    patch_depth_m: Optional[torch.Tensor],
    topo_context: Optional[Mapping[str, torch.Tensor]],
    physics_cfg: Mapping[str, Any],
    variable_indices: Mapping[str, Optional[int]],
) -> Dict[str, Any]:
    """Compute region-aware physics losses with fallback global mode."""
    zero = _zero_like(pred_phys)
    physics_eps = _cfg_float(physics_cfg, "physics_eps", 1e-6)
    bsz, _, dep, _, _ = pred_phys.shape

    depth_m, dz, spacing_mode = _build_depth_context(
        patch_depth_m=patch_depth_m,
        batch_size=bsz,
        depth_len=dep,
        device=pred_phys.device,
        dtype=pred_phys.dtype,
        physics_eps=physics_eps,
    )
    groups = _build_region_groups(region_labels=region_labels, cfg=physics_cfg, batch_size=bsz, device=pred_phys.device)

    raw = {"density": zero, "vert": zero, "aou_o2": zero, "np": zero, "remin": zero, "tv": zero, "topo": zero}
    norm = {"density": zero, "vert": zero, "aou_o2": zero, "np": zero, "remin": zero, "tv": zero, "topo": zero}
    weighted = {"density": zero, "vert": zero, "aou_o2": zero, "np": zero, "remin": zero, "tv": zero, "topo": zero}
    breakdown: Dict[str, Dict[str, float]] = {}
    use_loss_calibration = _loss_scale_enabled(physics_cfg)
    scale_eps = _cfg_float(physics_cfg, "loss_scale_eps", 1e-6)

    tv_indices = _resolve_tv_indices(physics_cfg, variable_indices=variable_indices, channel_count=pred_phys.shape[1])
    enabled_density = _cfg_bool(physics_cfg, "use_density_loss", True)
    enabled_vert = _cfg_bool(physics_cfg, "use_vert_loss", True)
    enabled_aou = _cfg_bool(physics_cfg, "use_aou_o2_loss", True)
    enabled_np = _cfg_bool(physics_cfg, "use_np_loss", True)
    enabled_remin = _cfg_bool(physics_cfg, "use_remin_loss", True)
    enabled_tv = _cfg_bool(physics_cfg, "use_tv_loss", True)
    enabled_topo = _cfg_bool(physics_cfg, "use_topo_constraint", True)

    for region_name, sel in groups.items():
        if int(sel.sum().detach().item()) <= 0:
            continue

        rp = pred_phys[sel]
        rv = valid_mask[sel]
        rd = depth_m[sel]
        rdz = dz[sel]
        rt = None
        if topo_context is not None:
            rt = {}
            for k, v in topo_context.items():
                if torch.is_tensor(v) and v.ndim > 0 and int(v.shape[0]) == bsz:
                    rt[k] = v[sel]
                else:
                    rt[k] = v

        density_raw = _compute_density_loss(
            pred_phys=rp,
            valid_mask=rv,
            idx_t=variable_indices.get("idx_t"),
            idx_s=variable_indices.get("idx_s"),
            dz=rdz,
            cfg=physics_cfg,
            topo_context=rt,
            zero=zero,
        ) if enabled_density else zero
        vert_raw = _compute_vert_loss(
            pred_phys=rp,
            valid_mask=rv,
            idx_t=variable_indices.get("idx_t"),
            idx_s=variable_indices.get("idx_s"),
            idx_aou=variable_indices.get("idx_aou"),
            dz=rdz,
            cfg=physics_cfg,
            topo_context=rt,
            zero=zero,
        ) if enabled_vert else zero
        aou_raw = _compute_aou_o2_loss(
            pred_phys=rp,
            valid_mask=rv,
            idx_aou=variable_indices.get("idx_aou"),
            idx_o=variable_indices.get("idx_o"),
            idx_o2sat=variable_indices.get("idx_o2sat"),
            idx_t=variable_indices.get("idx_t"),
            idx_s=variable_indices.get("idx_s"),
            cfg=physics_cfg,
            zero=zero,
        ) if enabled_aou else zero
        np_raw = _compute_np_loss(
            pred_phys=rp,
            valid_mask=rv,
            idx_n=variable_indices.get("idx_n"),
            idx_p=variable_indices.get("idx_p"),
            depth_m=rd,
            cfg=physics_cfg,
            zero=zero,
        ) if enabled_np else zero
        remin_raw = _compute_remin_loss(
            pred_phys=rp,
            valid_mask=rv,
            idx_aou=variable_indices.get("idx_aou"),
            idx_o=variable_indices.get("idx_o"),
            idx_n=variable_indices.get("idx_n"),
            idx_p=variable_indices.get("idx_p"),
            depth_m=rd,
            dz=rdz,
            cfg=physics_cfg,
            topo_context=rt,
            zero=zero,
        ) if enabled_remin else zero
        tv_raw = _compute_tv_loss(
            pred_phys=rp,
            valid_mask=rv,
            tv_indices=tv_indices,
            dz=rdz,
            cfg=physics_cfg,
            topo_context=rt,
            zero=zero,
        ) if enabled_tv else zero
        topo_raw = _compute_topo_constraint_loss(
            pred_phys=rp,
            valid_mask=rv,
            topo_context=rt,
            variable_indices=variable_indices,
            dz=rdz,
            cfg=physics_cfg,
            zero=zero,
        ) if enabled_topo else zero

        raw["density"] = raw["density"] + density_raw
        raw["vert"] = raw["vert"] + vert_raw
        raw["aou_o2"] = raw["aou_o2"] + aou_raw
        raw["np"] = raw["np"] + np_raw
        raw["remin"] = raw["remin"] + remin_raw
        raw["tv"] = raw["tv"] + tv_raw
        raw["topo"] = raw["topo"] + topo_raw

        density_norm = density_raw / _resolve_loss_scale(physics_cfg, "density", eps=scale_eps) if use_loss_calibration else density_raw
        vert_norm = vert_raw / _resolve_loss_scale(physics_cfg, "vert", eps=scale_eps) if use_loss_calibration else vert_raw
        aou_norm = aou_raw / _resolve_loss_scale(physics_cfg, "aou_o2", eps=scale_eps) if use_loss_calibration else aou_raw
        np_norm = np_raw / _resolve_loss_scale(physics_cfg, "np", eps=scale_eps) if use_loss_calibration else np_raw
        remin_norm = remin_raw / _resolve_loss_scale(physics_cfg, "remin", eps=scale_eps) if use_loss_calibration else remin_raw
        tv_norm = tv_raw / _resolve_loss_scale(physics_cfg, "tv", eps=scale_eps) if use_loss_calibration else tv_raw
        topo_norm = topo_raw / _resolve_loss_scale(physics_cfg, "topo", eps=scale_eps) if use_loss_calibration else topo_raw

        norm["density"] = norm["density"] + density_norm
        norm["vert"] = norm["vert"] + vert_norm
        norm["aou_o2"] = norm["aou_o2"] + aou_norm
        norm["np"] = norm["np"] + np_norm
        norm["remin"] = norm["remin"] + remin_norm
        norm["tv"] = norm["tv"] + tv_norm
        norm["topo"] = norm["topo"] + topo_norm

        lambdas = _region_lambdas(region_name=region_name, cfg=physics_cfg)
        density_w = float(lambdas["lambda_density"]) * density_norm
        vert_w = float(lambdas["lambda_vert"]) * vert_norm
        aou_w = float(lambdas["lambda_aou_o2"]) * aou_norm
        np_w = float(lambdas["lambda_np"]) * np_norm
        remin_w = float(lambdas["lambda_remin"]) * remin_norm
        tv_w = float(lambdas["lambda_tv"]) * tv_norm
        topo_w = float(lambdas["lambda_topo"]) * topo_norm

        weighted["density"] = weighted["density"] + density_w
        weighted["vert"] = weighted["vert"] + vert_w
        weighted["aou_o2"] = weighted["aou_o2"] + aou_w
        weighted["np"] = weighted["np"] + np_w
        weighted["remin"] = weighted["remin"] + remin_w
        weighted["tv"] = weighted["tv"] + tv_w
        weighted["topo"] = weighted["topo"] + topo_w

        breakdown[region_name] = {
            "sample_count": float(sel.sum().detach().item()),
            "density_raw": float(density_raw.detach().item()),
            "vert_raw": float(vert_raw.detach().item()),
            "aou_o2_raw": float(aou_raw.detach().item()),
            "np_raw": float(np_raw.detach().item()),
            "remin_raw": float(remin_raw.detach().item()),
            "tv_raw": float(tv_raw.detach().item()),
            "topo_raw": float(topo_raw.detach().item()),
            "density_norm": float(density_norm.detach().item()),
            "vert_norm": float(vert_norm.detach().item()),
            "aou_o2_norm": float(aou_norm.detach().item()),
            "np_norm": float(np_norm.detach().item()),
            "remin_norm": float(remin_norm.detach().item()),
            "tv_norm": float(tv_norm.detach().item()),
            "topo_norm": float(topo_norm.detach().item()),
            "lambda_density": float(lambdas["lambda_density"]),
            "lambda_vert": float(lambdas["lambda_vert"]),
            "lambda_aou_o2": float(lambdas["lambda_aou_o2"]),
            "lambda_np": float(lambdas["lambda_np"]),
            "lambda_remin": float(lambdas["lambda_remin"]),
            "lambda_tv": float(lambdas["lambda_tv"]),
            "lambda_topo": float(lambdas["lambda_topo"]),
        }

    return {
        "raw": raw,
        "norm": norm,
        "weighted": weighted,
        "region_breakdown": breakdown,
        "depth_spacing_mode": spacing_mode,
    }


def compute_loss_bundle(
    pred: torch.Tensor,
    target: torch.Tensor,
    supervise_mask: torch.Tensor,
    mae_weight: float = 0.0,
    physics_config: Optional[Mapping[str, Any]] = None,
    norm_stats: Optional[Mapping[str, torch.Tensor]] = None,
    variable_indices: Optional[Mapping[str, Optional[int]]] = None,
    region_labels: Optional[torch.Tensor] = None,
    patch_depth_m: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    topo_runtime: Optional[Mapping[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    """Return final loss bundle.

    Total loss:
    L_total = lambda_data * L_data + sum_r [lambda_i^(r) * L_i_norm^(r)].
    """
    sup_count = supervise_mask.sum()
    zero = _zero_like(pred)
    if float(sup_count.detach().item()) <= 0.0:
        return {
            "loss": zero,
            "total_loss": zero,
            "mse": zero,
            "mae": zero,
            "data_loss": zero,
            "data_loss_weighted": zero,
            "density_loss": zero,
            "vert_loss": zero,
            "aou_o2_loss": zero,
            "np_loss": zero,
            "remin_loss": zero,
            "tv_loss": zero,
            "topo_loss": zero,
            "density_loss_raw": zero,
            "vert_loss_raw": zero,
            "aou_o2_loss_raw": zero,
            "np_loss_raw": zero,
            "remin_loss_raw": zero,
            "tv_loss_raw": zero,
            "topo_loss_raw": zero,
            "density_loss_norm": zero,
            "vert_loss_norm": zero,
            "aou_o2_loss_norm": zero,
            "np_loss_norm": zero,
            "remin_loss_norm": zero,
            "tv_loss_norm": zero,
            "topo_loss_norm": zero,
            "supervise_count": sup_count,
            "is_empty_supervise": torch.tensor(1, device=pred.device),
        }

    mse = masked_mse(pred, target, supervise_mask)
    mae = masked_mae(pred, target, supervise_mask)
    data_loss = mse + float(mae_weight) * mae

    if physics_config is None or not _cfg_bool(physics_config, "use_physics_loss", False):
        return {
            "loss": data_loss,
            "total_loss": data_loss,
            "mse": mse,
            "mae": mae,
            "data_loss": data_loss,
            "data_loss_weighted": data_loss,
            "density_loss": zero,
            "vert_loss": zero,
            "aou_o2_loss": zero,
            "np_loss": zero,
            "remin_loss": zero,
            "tv_loss": zero,
            "topo_loss": zero,
            "density_loss_raw": zero,
            "vert_loss_raw": zero,
            "aou_o2_loss_raw": zero,
            "np_loss_raw": zero,
            "remin_loss_raw": zero,
            "tv_loss_raw": zero,
            "topo_loss_raw": zero,
            "density_loss_norm": zero,
            "vert_loss_norm": zero,
            "aou_o2_loss_norm": zero,
            "np_loss_norm": zero,
            "remin_loss_norm": zero,
            "tv_loss_norm": zero,
            "topo_loss_norm": zero,
            "supervise_count": sup_count,
            "is_empty_supervise": torch.tensor(0, device=pred.device),
        }

    physics_eps = _cfg_float(physics_config, "physics_eps", 1e-6)
    safe_nan_to_num = _cfg_bool(physics_config, "safe_nan_to_num", True)
    clamp_value = _cfg_float(physics_config, "gradient_clamp_value", 1e4)
    lambda_data = _cfg_float(physics_config, "lambda_data", 1.0)
    variable_indices = variable_indices or {}

    if norm_stats is None:
        return {
            "loss": data_loss,
            "total_loss": data_loss,
            "mse": mse,
            "mae": mae,
            "data_loss": data_loss,
            "data_loss_weighted": data_loss,
            "density_loss": zero,
            "vert_loss": zero,
            "aou_o2_loss": zero,
            "np_loss": zero,
            "remin_loss": zero,
            "tv_loss": zero,
            "topo_loss": zero,
            "density_loss_raw": zero,
            "vert_loss_raw": zero,
            "aou_o2_loss_raw": zero,
            "np_loss_raw": zero,
            "remin_loss_raw": zero,
            "tv_loss_raw": zero,
            "topo_loss_raw": zero,
            "density_loss_norm": zero,
            "vert_loss_norm": zero,
            "aou_o2_loss_norm": zero,
            "np_loss_norm": zero,
            "remin_loss_norm": zero,
            "tv_loss_norm": zero,
            "topo_loss_norm": zero,
            "supervise_count": sup_count,
            "is_empty_supervise": torch.tensor(0, device=pred.device),
        }

    pred_phys = denormalize_for_physics(pred=pred, norm_stats=norm_stats, physics_eps=physics_eps)
    pred_phys = _sanitize_tensor(pred_phys, safe_nan_to_num=safe_nan_to_num, clamp_value=clamp_value)

    if valid_mask is None:
        valid_mask = torch.isfinite(target)
    valid_mask = valid_mask.bool() & torch.isfinite(pred_phys)
    topo_ctx = _build_topo_context(
        topo_runtime=topo_runtime,
        batch_size=int(pred.shape[0]),
        depth_len=int(pred.shape[2]),
        h=int(pred.shape[3]),
        w=int(pred.shape[4]),
        device=pred.device,
        dtype=pred.dtype,
    )
    use_topo_constraint = _cfg_bool(physics_config, "use_topo_constraint", True)
    if use_topo_constraint:
        wet = topo_ctx["wet_mask"].unsqueeze(1)
        valid_mask = valid_mask & wet.bool()

    physics = compute_loss_by_region(
        pred_phys=pred_phys,
        valid_mask=valid_mask,
        region_labels=region_labels,
        patch_depth_m=patch_depth_m,
        topo_context=topo_ctx if use_topo_constraint else None,
        physics_cfg=physics_config,
        variable_indices=variable_indices,
    )
    raw = physics["raw"]
    norm = physics["norm"]
    weighted = physics["weighted"]

    data_loss_weighted = float(lambda_data) * data_loss
    total = (
        data_loss_weighted
        + weighted["density"]
        + weighted["vert"]
        + weighted["aou_o2"]
        + weighted["np"]
        + weighted["remin"]
        + weighted["tv"]
        + weighted["topo"]
    )

    return {
        "loss": total,
        "total_loss": total,
        "mse": mse,
        "mae": mae,
        "data_loss": data_loss,
        "data_loss_weighted": data_loss_weighted,
        "density_loss": weighted["density"],
        "vert_loss": weighted["vert"],
        "aou_o2_loss": weighted["aou_o2"],
        "np_loss": weighted["np"],
        "remin_loss": weighted["remin"],
        "tv_loss": weighted["tv"],
        "topo_loss": weighted["topo"],
        "density_loss_raw": raw["density"],
        "vert_loss_raw": raw["vert"],
        "aou_o2_loss_raw": raw["aou_o2"],
        "np_loss_raw": raw["np"],
        "remin_loss_raw": raw["remin"],
        "tv_loss_raw": raw["tv"],
        "topo_loss_raw": raw["topo"],
        "density_loss_norm": norm["density"],
        "vert_loss_norm": norm["vert"],
        "aou_o2_loss_norm": norm["aou_o2"],
        "np_loss_norm": norm["np"],
        "remin_loss_norm": norm["remin"],
        "tv_loss_norm": norm["tv"],
        "topo_loss_norm": norm["topo"],
        "supervise_count": sup_count,
        "is_empty_supervise": torch.tensor(0, device=pred.device),
    }
