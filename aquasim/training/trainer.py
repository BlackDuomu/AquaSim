#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquasim.models.backbone3d import MARINE_MODEL_NAMES, build_marine_model
from aquasim.data.dataset import (
    Step6PatchDataset,
    TopoRuntimeAssets,
    apply_channel_norm,
    default_channel_norm_cache_path,
    load_all_splits,
    load_or_compute_channel_norm_stats,
    load_topo_runtime_assets,
)
from aquasim.training.losses import compute_loss_bundle, resolve_variable_indices
from aquasim.utils.runtime import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train AquaSim model on patch datasets (pretrain or strict finetune).')
    p.add_argument('--stage', choices=['pretrain', 'finetune'], default='finetune')
    p.add_argument('--data-dir', type=Path, required=True)
    p.add_argument('--unified-dir', type=Path, default=Path('data/processed/unified_field'))
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--model-name', type=str, default='marine3d_transformer', choices=list(MARINE_MODEL_NAMES))
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--early-stop-patience', type=int, default=0, help='Stop early after N non-improving val epochs; 0 disables.')
    p.add_argument('--min-delta', type=float, default=0.0, help='Minimum val-loss improvement required for early stopping.')
    p.add_argument('--norm-stats-cache', type=Path, default=None, help='Channel normalization stats cache path; defaults to data-dir/patch_dataset/channel_norm_stats.npz.')
    p.add_argument('--no-norm-stats-cache', action='store_true', help='Disable loading/saving cached channel normalization stats.')
    p.add_argument('--mae-weight', type=float, default=0.1)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--prefetch-factor', type=int, default=2, help='DataLoader prefetch factor when num-workers > 0.')
    p.add_argument('--persistent-workers', dest='persistent_workers', action='store_true', help='Keep DataLoader workers alive across epochs when num-workers > 0.')
    p.add_argument('--no-persistent-workers', dest='persistent_workers', action='store_false', help='Disable persistent DataLoader workers.')
    p.set_defaults(persistent_workers=True)
    p.add_argument('--pin-memory', dest='pin_memory', action='store_true', help='Use DataLoader pinned memory and non-blocking CUDA transfers.')
    p.add_argument('--no-pin-memory', dest='pin_memory', action='store_false', help='Disable DataLoader pinned memory.')
    p.set_defaults(pin_memory=True)
    p.add_argument('--max-train-batches', type=int, default=0)
    p.add_argument('--max-val-batches', type=int, default=0)
    p.add_argument('--max-test-batches', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force-cpu', action='store_true')
    p.add_argument('--cudnn-benchmark', dest='cudnn_benchmark', action='store_true', help='Enable cuDNN benchmark for fixed-size CUDA inputs.')
    p.add_argument('--no-cudnn-benchmark', dest='cudnn_benchmark', action='store_false', help='Disable cuDNN benchmark.')
    p.set_defaults(cudnn_benchmark=True)
    p.add_argument('--init-checkpoint', type=Path, default=None)
    p.add_argument('--strict-init-load', action='store_true', default=True)

    p.add_argument('--use-physics-loss', dest='use_physics_loss', action='store_true')
    p.add_argument('--no-physics-loss', dest='use_physics_loss', action='store_false')
    p.set_defaults(use_physics_loss=True)
    p.add_argument('--use-density-loss', action='store_true', default=True)
    p.add_argument('--use-vert-loss', action='store_true', default=True)
    p.add_argument('--use-aou-o2-loss', action='store_true', default=True)
    p.add_argument('--use-np-loss', action='store_true', default=True)
    p.add_argument('--use-remin-loss', action='store_true', default=True)
    p.add_argument('--use-tv-loss', action='store_true', default=True)

    p.add_argument('--lambda-data', type=float, default=1.0)
    p.add_argument('--lambda-density', type=float, default=0.05)
    p.add_argument('--lambda-vert', type=float, default=0.02)
    p.add_argument('--lambda-aou-o2', type=float, default=0.05)
    p.add_argument('--lambda-np', type=float, default=0.02)
    p.add_argument('--lambda-remin', type=float, default=0.02)
    p.add_argument('--lambda-tv', type=float, default=0.005)
    p.add_argument('--lambda-topo', type=float, default=0.02)

    p.add_argument('--topo-aware', action='store_true', default=False)
    p.add_argument('--topo-dir', type=Path, default=Path('outputs/topo_prepared'))
    p.add_argument('--use-topo-constraint', dest='use_topo_constraint', action='store_true')
    p.add_argument('--no-topo-constraint', dest='use_topo_constraint', action='store_false')
    p.set_defaults(use_topo_constraint=True)
    p.add_argument('--topo-metadata', type=Path, default=None)
    p.add_argument('--topo-near-bottom-weight', type=float, default=0.8)
    p.add_argument('--topo-slope-weight', type=float, default=0.5)
    p.add_argument('--topo-loss-lambda', type=float, default=0.0)
    p.add_argument('--topo-flat-enhance-weight', type=float, default=0.25)
    p.add_argument('--loss-scale-json', type=Path, default=None)
    return p.parse_args()


def _load_variable_names(unified_dir: Path, channels: int) -> List[str]:
    vm_path = unified_dir / 'variable_dimension_mapping.csv'
    if not vm_path.exists():
        raise FileNotFoundError(f'missing variable map: {vm_path}')
    with vm_path.open('r', encoding='utf-8', newline='') as f:
        vm = pd.DataFrame(list(csv.DictReader(f)))
    vm.columns = [str(c).strip().lstrip('\ufeff') for c in vm.columns]
    if 'var_name' not in vm.columns:
        vm = vm.rename(columns={vm.columns[0]: 'var_name'})
    names = [str(v) for v in vm['var_name'].tolist()[:channels]]
    if len(names) < channels:
        raise RuntimeError('variable map does not contain enough channels')
    return names


def _load_depth_levels(unified_dir: Path) -> np.ndarray:
    path = unified_dir / 'depth_levels.npy'
    if not path.exists():
        raise FileNotFoundError(f'missing depth levels: {path}')
    return np.load(path, allow_pickle=False).astype(np.float32)


def _load_loss_calibration(path: Optional[Path]) -> Dict[str, object]:
    out: Dict[str, object] = {
        'enabled': False,
        'path': '',
        'scales': {},
    }
    if path is None:
        return out
    if not path.exists():
        raise FileNotFoundError(f'loss calibration json not found: {path}')
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    required = ['density', 'vert', 'aou_o2', 'np', 'remin', 'tv', 'topo']
    scales: Dict[str, float] = {}
    for key in required:
        node = payload.get(key, None)
        if isinstance(node, dict):
            scale = float(node.get('scale', 1.0))
        else:
            scale = float(node)
        if scale <= 0.0:
            raise ValueError(f'invalid non-positive loss scale for `{key}` in {path}')
        scales[key] = scale
    out['enabled'] = True
    out['path'] = str(path)
    out['scales'] = scales
    return out


def _build_topo_weight_lookup(topo_csv: Optional[Path], near_coeff: float, slope_coeff: float) -> Dict[int, float]:
    if topo_csv is None:
        return {}
    if not topo_csv.exists():
        raise FileNotFoundError(f'missing topo metadata: {topo_csv}')
    df = pd.read_csv(topo_csv)
    required = {'patch_id', 'near_bottom_fraction', 'slope_fraction'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f'topo metadata missing columns: {required}')
    out: Dict[int, float] = {}
    for _, row in df.iterrows():
        pid = int(row['patch_id'])
        near = float(row['near_bottom_fraction'])
        slope = float(row['slope_fraction'])
        out[pid] = float(max(1e-3, 1.0 + near_coeff * near + slope_coeff * slope))
    return out


def _weighted_topo_loss(pred: torch.Tensor, target: torch.Tensor, supervise: torch.Tensor, patch_ids: torch.Tensor, lookup: Dict[int, float]) -> torch.Tensor:
    if not lookup:
        return pred.sum() * 0.0
    sq = ((pred - target) ** 2) * supervise
    per_patch_num = sq.sum(dim=(1, 2, 3, 4))
    per_patch_den = torch.clamp(supervise.sum(dim=(1, 2, 3, 4)), min=1.0)
    per_patch = per_patch_num / per_patch_den
    weights = torch.tensor([lookup.get(int(x), 1.0) for x in patch_ids.detach().cpu().numpy()], device=pred.device, dtype=pred.dtype)
    return (per_patch * weights).mean()


def _move_batch_to_device(batch: Dict[str, object], device: torch.device, non_blocking: bool) -> Dict[str, object]:
    return {
        k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    mae_weight: float,
    physics_cfg: Dict[str, object],
    variable_indices: Dict[str, Optional[int]],
    norm_stats_torch: Dict[str, torch.Tensor],
    max_batches: int,
    legacy_topo_lookup: Dict[int, float],
    legacy_topo_loss_lambda: float,
    non_blocking_transfer: bool,
) -> Dict[str, float]:
    train = optimizer is not None
    model.train(mode=train)

    loss_sum = 0.0
    val_sum = 0
    for bi, batch in enumerate(loader):
        if max_batches > 0 and bi >= max_batches:
            break
        batch = _move_batch_to_device(batch, device=device, non_blocking=non_blocking_transfer)

        with torch.set_grad_enabled(train):
            pred = model(batch['input'], batch['visible_mask'])
            bundle = compute_loss_bundle(
                pred=pred,
                target=batch['target'],
                supervise_mask=batch['supervise_mask'],
                mae_weight=mae_weight,
                physics_config=physics_cfg,
                variable_indices=variable_indices,
                norm_stats=norm_stats_torch,
                region_labels=None,
                patch_depth_m=batch.get('patch_depth_m', None),
                valid_mask=batch['visible_mask'],
                topo_runtime={
                    'wet_mask': batch.get('wet_mask', None),
                    'near_bottom_mask': batch.get('near_bottom_mask', None),
                    'slope_weight': batch.get('slope_weight', None),
                    'roughness_weight': batch.get('roughness_weight', None),
                    'topo_class': batch.get('topo_class', None),
                },
            )
            loss = bundle['loss']
            if legacy_topo_loss_lambda > 0.0:
                topo_loss = _weighted_topo_loss(
                    pred=pred,
                    target=batch['target'],
                    supervise=batch['supervise_mask'],
                    patch_ids=batch['patch_id'],
                    lookup=legacy_topo_lookup,
                )
                loss = loss + float(legacy_topo_loss_lambda) * topo_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        loss_sum += float(loss.detach().item())
        val_sum += 1

    return {'loss': float(loss_sum / max(val_sum, 1)), 'steps': float(val_sum)}


def _evaluate_test(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int, non_blocking_transfer: bool) -> Dict[str, float]:
    model.eval()
    sq_sum = 0.0
    abs_sum = 0.0
    cnt = 0.0
    near_sq_sum = 0.0
    near_cnt = 0.0
    slope_sq_sum = 0.0
    slope_cnt = 0.0
    wet_cnt = 0.0
    topo_samples = 0
    slope_high_thr = 0.66
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break
            batch = _move_batch_to_device(batch, device=device, non_blocking=non_blocking_transfer)
            pred = model(batch['input'], batch['visible_mask'])
            sup = batch['supervise_mask']
            pred = torch.nan_to_num(pred, nan=0.0, posinf=1e6, neginf=-1e6)
            target = torch.nan_to_num(batch['target'], nan=0.0, posinf=1e6, neginf=-1e6)
            finite = torch.isfinite(pred) & torch.isfinite(target)
            sup = sup * finite.to(dtype=sup.dtype)
            diff = (pred - target) * sup
            sq_sum += float((diff * diff).sum().item())
            abs_sum += float(diff.abs().sum().item())
            cnt += float(sup.sum().item())
            wet = batch.get('wet_mask', None)
            near = batch.get('near_bottom_mask', None)
            slope = batch.get('slope_weight', None)
            if wet is not None and near is not None and slope is not None:
                topo_samples += 1
                wet4 = wet.unsqueeze(1).to(dtype=sup.dtype)
                near4 = near.unsqueeze(1).to(dtype=sup.dtype)
                slope4 = slope.unsqueeze(1).to(dtype=sup.dtype)
                wet_cnt += float((sup * wet4).sum().item())
                near_mask = sup * wet4 * near4
                near_sq_sum += float(((diff * diff) * near_mask).sum().item())
                near_cnt += float(near_mask.sum().item())
                slope_mask = sup * wet4 * (slope4 >= float(slope_high_thr)).to(dtype=sup.dtype)
                slope_sq_sum += float(((diff * diff) * slope_mask).sum().item())
                slope_cnt += float(slope_mask.sum().item())
    if cnt <= 0:
        return {
            'rmse': float('nan'),
            'mae': float('nan'),
            'count': 0,
            'near_bottom_rmse': float('nan'),
            'slope_area_rmse': float('nan'),
            'topo_consistency_score': float('nan'),
            'wet_supervise_count': 0.0,
        }
    rmse = float(np.sqrt(sq_sum / cnt))
    near_rmse = float(np.sqrt(near_sq_sum / near_cnt)) if near_cnt > 0 else float(rmse)
    slope_rmse = float(np.sqrt(slope_sq_sum / slope_cnt)) if slope_cnt > 0 else float(rmse)
    if rmse > 0:
        near_ratio = near_rmse / rmse
        slope_ratio = slope_rmse / rmse
        topo_consistency = float(1.0 / (1.0 + 0.5 * (near_ratio + slope_ratio)))
    else:
        topo_consistency = 0.0
    return {
        'rmse': rmse,
        'mae': float(abs_sum / cnt),
        'count': int(cnt),
        'near_bottom_rmse': near_rmse,
        'slope_area_rmse': slope_rmse,
        'topo_consistency_score': topo_consistency,
        'wet_supervise_count': float(wet_cnt),
        'topo_runtime_batches': int(topo_samples),
    }


def main() -> None:
    args = parse_args()
    if not args.data_dir.exists():
        raise FileNotFoundError(f'data dir not found: {args.data_dir}')

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark and torch.cuda.is_available() and not args.force_cpu)
    ensure_dir(args.out_dir)
    ensure_dir(args.out_dir / 'checkpoints')
    device = get_device(force_cpu=args.force_cpu)

    train_raw, val_raw, test_raw = load_all_splits(args.data_dir)
    norm_stats_cache_path = args.norm_stats_cache if args.norm_stats_cache is not None else default_channel_norm_cache_path(args.data_dir)
    stats = load_or_compute_channel_norm_stats(
        step5_dir=args.data_dir,
        train_split=train_raw,
        cache_path=norm_stats_cache_path,
        use_cache=not bool(args.no_norm_stats_cache),
    )
    train_split = apply_channel_norm(train_raw, stats)
    val_split = apply_channel_norm(val_raw, stats)
    test_split = apply_channel_norm(test_raw, stats)

    np.savez(args.out_dir / 'normalization_stats.npz', mean=stats.mean, std=stats.std)

    topo_assets: Optional[TopoRuntimeAssets] = None
    topo_runtime_enabled = bool(args.use_topo_constraint or args.topo_aware)
    if topo_runtime_enabled:
        topo_assets = load_topo_runtime_assets(
            topo_dir=args.topo_dir,
            data_dir=args.data_dir,
            unified_dir=args.unified_dir,
            required=bool(args.use_topo_constraint),
        )

    pin_memory = bool(args.pin_memory and device.type == 'cuda')
    non_blocking_transfer = bool(pin_memory)
    loader_kwargs = {
        'num_workers': int(args.num_workers),
        'pin_memory': bool(pin_memory),
    }
    if int(args.num_workers) > 0:
        loader_kwargs['prefetch_factor'] = max(1, int(args.prefetch_factor))
        loader_kwargs['persistent_workers'] = bool(args.persistent_workers)

    train_loader = DataLoader(
        Step6PatchDataset(train_split, topo_runtime_assets=topo_assets, return_topo_runtime=topo_runtime_enabled),
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Step6PatchDataset(val_split, topo_runtime_assets=topo_assets, return_topo_runtime=topo_runtime_enabled),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Step6PatchDataset(test_split, topo_runtime_assets=topo_assets, return_topo_runtime=topo_runtime_enabled),
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_marine_model(
        model_name=args.model_name,
        in_channels=train_split.input.shape[1] * 2,
        out_channels=train_split.target.shape[1],
        base_channels=32,
    ).to(device)

    if args.init_checkpoint is not None:
        if not args.init_checkpoint.exists():
            raise FileNotFoundError(f'init checkpoint not found: {args.init_checkpoint}')
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        st = ckpt['model_state'] if isinstance(ckpt, dict) and 'model_state' in ckpt else ckpt
        model.load_state_dict(st, strict=bool(args.strict_init_load))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    variable_names = _load_variable_names(args.unified_dir, channels=train_split.target.shape[1])
    variable_indices = resolve_variable_indices(variable_names)
    physics_cfg: Dict[str, object] = {
        'use_physics_loss': bool(args.use_physics_loss),
        'use_density_loss': bool(args.use_density_loss),
        'use_vert_loss': bool(args.use_vert_loss),
        'use_aou_o2_loss': bool(args.use_aou_o2_loss),
        'use_np_loss': bool(args.use_np_loss),
        'use_remin_loss': bool(args.use_remin_loss),
        'use_tv_loss': bool(args.use_tv_loss),
        'lambda_data': float(args.lambda_data),
        'lambda_density': float(args.lambda_density),
        'lambda_vert': float(args.lambda_vert),
        'lambda_aou_o2': float(args.lambda_aou_o2),
        'lambda_np': float(args.lambda_np),
        'lambda_remin': float(args.lambda_remin),
        'lambda_tv': float(args.lambda_tv),
        'lambda_topo': float(args.lambda_topo),
        'use_region_weighting': False,
        'region_weight_mode': 'unified',
        'use_topo_constraint': bool(args.use_topo_constraint),
        'topo_near_bottom_boost': float(args.topo_near_bottom_weight),
        'topo_slope_relax_strength': float(args.topo_slope_weight),
        'topo_flat_enhance_strength': float(args.topo_flat_enhance_weight),
        'use_loss_calibration': False,
        'loss_scales': {},
        'loss_scale_eps': 1e-6,
    }
    loss_calibration = _load_loss_calibration(args.loss_scale_json)
    physics_cfg['use_loss_calibration'] = bool(loss_calibration['enabled'])
    physics_cfg['loss_scales'] = dict(loss_calibration['scales']) if isinstance(loss_calibration['scales'], dict) else {}

    legacy_topo_lookup = _build_topo_weight_lookup(
        topo_csv=args.topo_metadata if (args.topo_aware and float(args.topo_loss_lambda) > 0.0) else None,
        near_coeff=float(args.topo_near_bottom_weight),
        slope_coeff=float(args.topo_slope_weight),
    )

    norm_stats_torch = {
        'mean': torch.tensor(stats.mean, dtype=torch.float32, device=device),
        'std': torch.tensor(stats.std, dtype=torch.float32, device=device),
    }

    best_val = float('inf')
    rows: List[Dict[str, float]] = []
    epochs_without_improvement = 0
    early_stop_patience = max(0, int(args.early_stop_patience))
    min_delta = max(0.0, float(args.min_delta))
    stopped_early = False

    for ep in range(1, args.epochs + 1):
        tr = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            mae_weight=args.mae_weight,
            physics_cfg=physics_cfg,
            variable_indices=variable_indices,
            norm_stats_torch=norm_stats_torch,
            max_batches=args.max_train_batches,
            legacy_topo_lookup=legacy_topo_lookup,
            legacy_topo_loss_lambda=float(args.topo_loss_lambda if args.topo_aware else 0.0),
            non_blocking_transfer=non_blocking_transfer,
        )
        va = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            mae_weight=args.mae_weight,
            physics_cfg=physics_cfg,
            variable_indices=variable_indices,
            norm_stats_torch=norm_stats_torch,
            max_batches=args.max_val_batches,
            legacy_topo_lookup=legacy_topo_lookup,
            legacy_topo_loss_lambda=0.0,
            non_blocking_transfer=non_blocking_transfer,
        )
        rows.append({'epoch': ep, 'train_loss': tr['loss'], 'val_loss': va['loss']})

        last_payload = {
            'epoch': ep,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'stage': args.stage,
        }
        torch.save(last_payload, args.out_dir / 'checkpoints' / 'last_model.pt')
        if va['loss'] < (best_val - min_delta):
            best_val = va['loss']
            epochs_without_improvement = 0
            torch.save(last_payload, args.out_dir / 'checkpoints' / 'best_model.pt')
        else:
            epochs_without_improvement += 1

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            stopped_early = True
            print(
                f"[EARLY-STOP] stage={args.stage} epoch={ep} "
                f"best_val={best_val:.6f} patience={early_stop_patience} min_delta={min_delta:g}"
            )
            break

    hist = pd.DataFrame(rows)
    hist.to_csv(args.out_dir / 'train_history.csv', index=False)

    test_metrics = _evaluate_test(
        model=model,
        loader=test_loader,
        device=device,
        max_batches=args.max_test_batches,
        non_blocking_transfer=non_blocking_transfer,
    )
    pd.DataFrame([test_metrics]).to_csv(args.out_dir / 'test_metrics.csv', index=False)

    protocol = {
        'project': 'AquaSim',
        'stage': args.stage,
        'data_dir': str(args.data_dir),
        'unified_dir': str(args.unified_dir),
        'out_dir': str(args.out_dir),
        'model_name': args.model_name,
        'normalization_stats': {
            'run_copy': str(args.out_dir / 'normalization_stats.npz'),
            'cache_enabled': not bool(args.no_norm_stats_cache),
            'cache_path': str(norm_stats_cache_path),
        },
        'data_loader': {
            'num_workers': int(args.num_workers),
            'prefetch_factor': int(loader_kwargs.get('prefetch_factor', 0)),
            'persistent_workers': bool(loader_kwargs.get('persistent_workers', False)),
            'pin_memory_requested': bool(args.pin_memory),
            'pin_memory_enabled': bool(pin_memory),
            'non_blocking_transfer': bool(non_blocking_transfer),
        },
        'cuda_backend': {
            'cudnn_benchmark_requested': bool(args.cudnn_benchmark),
            'cudnn_benchmark_enabled': bool(torch.backends.cudnn.benchmark),
        },
        'epochs_requested': int(args.epochs),
        'epochs_completed': int(len(rows)),
        'early_stopping': {
            'enabled': bool(early_stop_patience > 0),
            'patience': int(early_stop_patience),
            'min_delta': float(min_delta),
            'stopped_early': bool(stopped_early),
            'best_val_loss': float(best_val),
        },
        'physics_loss': {'config': physics_cfg},
        'loss_calibration': {
            'enabled': bool(loss_calibration['enabled']),
            'loss_scale_json': str(loss_calibration['path']),
            'normalized_physics_topo_loss': bool(loss_calibration['enabled']),
            'scales': loss_calibration['scales'],
        },
        'topo_aware': {
            'enabled': bool(args.topo_aware),
            'topo_runtime_enabled': bool(topo_runtime_enabled),
            'topo_runtime_dir': str(args.topo_dir),
            'use_topo_constraint': bool(args.use_topo_constraint),
            'lambda_topo': float(args.lambda_topo),
            'topo_metadata': str(args.topo_metadata) if args.topo_metadata else '',
            'legacy_patch_topo_loss_lambda': float(args.topo_loss_lambda),
            'near_bottom_weight': float(args.topo_near_bottom_weight),
            'slope_weight': float(args.topo_slope_weight),
            'flat_enhance_weight': float(args.topo_flat_enhance_weight),
            'topo_channel_in_backbone': False,
        },
        'strict_data_binding': str(args.data_dir),
        'variable_order': variable_names,
        'depth_levels_path': str(args.unified_dir / 'depth_levels.npy'),
    }
    save_json(args.out_dir / 'train_protocol.json', protocol)

    print(f"[DONE] stage={args.stage} out_dir={args.out_dir} best_val={best_val:.6f}")


if __name__ == '__main__':
    main()


