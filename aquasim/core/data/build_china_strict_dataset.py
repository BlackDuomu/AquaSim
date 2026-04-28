#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from aquasim.core.data.grid_utils import crop_bbox_indices, ensure_dir, get_starts, recover_lat_lon, split_code_to_name
from aquasim.core.data.mask_protocol import build_patch_masks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build strict China patch dataset from unified field tensors.')
    p.add_argument('--unified-dir', type=Path, default=Path('data/processed/unified_field'))
    p.add_argument('--out-dir', type=Path, default=Path('outputs/china_strict'))
    p.add_argument('--lat-min', type=float, default=0.0)
    p.add_argument('--lat-max', type=float, default=45.0)
    p.add_argument('--lon-min', type=float, default=105.0)
    p.add_argument('--lon-max', type=float, default=135.0)
    p.add_argument('--patch-depth', type=int, default=24)
    p.add_argument('--patch-lat', type=int, default=12)
    p.add_argument('--patch-lon', type=int, default=12)
    p.add_argument('--stride-depth', type=int, default=12)
    p.add_argument('--stride-lat', type=int, default=1)
    p.add_argument('--stride-lon', type=int, default=1)
    p.add_argument('--include-tail', action='store_true')
    p.add_argument('--artificial-ratio', type=float, default=0.4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-ocean-cells-2d', type=int, default=20)
    p.add_argument('--min-supervise-points', type=int, default=50)
    p.add_argument('--train-ratio', type=float, default=0.7)
    p.add_argument('--val-ratio', type=float, default=0.15)
    p.add_argument('--block-lat', type=int, default=12)
    p.add_argument('--block-lon', type=int, default=12)
    p.add_argument('--max-patches', type=int, default=0)
    return p.parse_args()


def _load_unified_field(unified_dir: Path) -> Dict[str, object]:
    required = [
        unified_dir / 'dataset_8vars.npy',
        unified_dir / 'mask_8vars.npy',
        unified_dir / 'depth_levels.npy',
        unified_dir / 'lat_lon_range.json',
        unified_dir / 'variable_dimension_mapping.csv',
    ]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(f'missing required unified field artifact: {p}')

    x = np.load(unified_dir / 'dataset_8vars.npy', allow_pickle=False).astype(np.float32)
    m = np.load(unified_dir / 'mask_8vars.npy', allow_pickle=False).astype(np.uint8)
    if x.shape != m.shape:
        raise ValueError('dataset/mask shape mismatch')

    depth = np.load(unified_dir / 'depth_levels.npy', allow_pickle=False).astype(np.float32)
    lat_lon = json.loads((unified_dir / 'lat_lon_range.json').read_text(encoding='utf-8'))
    lat, lon = recover_lat_lon(lat_lon, h=int(x.shape[2]), w=int(x.shape[3]))

    vm = pd.read_csv(unified_dir / 'variable_dimension_mapping.csv')
    vm.columns = [str(c).strip().lstrip('\ufeff') for c in vm.columns]
    if 'var_name' not in vm.columns:
        vm = vm.rename(columns={vm.columns[0]: 'var_name'})
    var_names = [str(v) for v in vm['var_name'].tolist()[: x.shape[0]]]

    return {'x': x, 'm': m, 'depth': depth, 'lat': lat, 'lon': lon, 'var_names': var_names}


def _build_split_owner(valid2d: np.ndarray, block_lat: int, block_lon: int, train_ratio: float, val_ratio: float, seed: int) -> np.ndarray:
    h, w = valid2d.shape
    owner = np.zeros((h, w), dtype=np.uint8)
    for h0 in range(0, h, max(int(block_lat), 1)):
        h1 = min(h0 + int(block_lat), h)
        for w0 in range(0, w, max(int(block_lon), 1)):
            w1 = min(w0 + int(block_lon), w)
            tile_valid = valid2d[h0:h1, w0:w1]
            if not tile_valid.any():
                continue
            key = (h0 * 1000003 + w0 * 9176 + int(seed) * 37) & 0xFFFFFFFF
            rng = np.random.default_rng(key)
            r = float(rng.random())
            code = 1 if r < train_ratio else (2 if r < (train_ratio + val_ratio) else 3)
            owner[h0:h1, w0:w1][tile_valid] = np.uint8(code)
    if not np.array_equal(owner > 0, valid2d):
        raise ValueError('split owner does not fully cover valid grid')
    return owner


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio <= 0 or (args.train_ratio + args.val_ratio) >= 1:
        raise ValueError('invalid split ratios')

    data = _load_unified_field(args.unified_dir)
    x = data['x']
    m = data['m']
    depth = data['depth']
    lat = data['lat']
    lon = data['lon']

    lat_idx, lon_idx = crop_bbox_indices(lat, lon, args.lat_min, args.lat_max, args.lon_min, args.lon_max)
    lat0, lat1 = int(lat_idx[0]), int(lat_idx[-1]) + 1
    lon0, lon1 = int(lon_idx[0]), int(lon_idx[-1]) + 1

    x_cn = x[:, :, lat0:lat1, lon0:lon1]
    m_cn = m[:, :, lat0:lat1, lon0:lon1]
    lat_cn = lat[lat0:lat1]
    lon_cn = lon[lon0:lon1]

    valid2d = np.any(m_cn.astype(bool), axis=(0, 1))
    owner = _build_split_owner(
        valid2d=valid2d,
        block_lat=args.block_lat,
        block_lon=args.block_lon,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    c, d, h, w = x_cn.shape
    d_starts = get_starts(d, args.patch_depth, args.stride_depth, args.include_tail)
    h_starts = get_starts(h, args.patch_lat, args.stride_lat, args.include_tail)
    w_starts = get_starts(w, args.patch_lon, args.stride_lon, args.include_tail)

    rows: List[Dict[str, object]] = []
    store = {k: defaultdict(list) for k in ['train', 'val', 'test']}
    dropped = Counter()
    patch_id = 0

    for d0 in d_starts:
        d1 = min(d0 + args.patch_depth, d)
        cd = d0 + (d1 - d0) // 2
        for h0 in h_starts:
            h1 = min(h0 + args.patch_lat, h)
            ch = h0 + (h1 - h0) // 2
            for w0 in w_starts:
                w1 = min(w0 + args.patch_lon, w)
                cw = w0 + (w1 - w0) // 2

                real_mask = m_cn[:, d0:d1, h0:h1, w0:w1]
                valid_patch = np.any(real_mask.astype(bool), axis=(0, 1))
                if int(valid_patch.sum()) < int(args.min_ocean_cells_2d):
                    dropped['too_few_ocean_cells'] += 1
                    continue

                owner_codes = owner[h0:h1, w0:w1][valid_patch]
                if owner_codes.size == 0 or np.any(owner_codes == 0):
                    dropped['invalid_owner'] += 1
                    continue
                if np.any(owner_codes != owner_codes[0]):
                    dropped['cross_split_window'] += 1
                    continue

                split = split_code_to_name(int(owner_codes[0]))
                if split == 'none':
                    dropped['invalid_split'] += 1
                    continue

                masks = build_patch_masks(real_mask=real_mask, ratio=args.artificial_ratio, seed=args.seed + patch_id)
                supervise_points = int(masks.supervise_mask.sum())
                if supervise_points < int(args.min_supervise_points):
                    dropped['too_few_supervise_points'] += 1
                    continue

                target = x_cn[:, d0:d1, h0:h1, w0:w1].astype(np.float32)
                inp = target.copy()
                inp[masks.visible_mask == 0] = 0.0

                store[split]['input'].append(inp)
                store[split]['target'].append(target)
                store[split]['real_mask'].append(masks.real_mask.astype(np.uint8))
                store[split]['artificial_mask'].append(masks.artificial_mask.astype(np.uint8))
                store[split]['visible_mask'].append(masks.visible_mask.astype(np.uint8))
                store[split]['supervise_mask'].append(masks.supervise_mask.astype(np.uint8))
                store[split]['patch_id'].append(np.array([patch_id], dtype=np.int64))

                rows.append({
                    'patch_id': patch_id,
                    'split': split,
                    'd0': d0,
                    'd1': d1,
                    'h0': h0,
                    'h1': h1,
                    'w0': w0,
                    'w1': w1,
                    'center_depth_m': float(depth[cd]),
                    'center_lat': float(lat_cn[ch]),
                    'center_lon': float(lon_cn[cw]),
                    'supervise_points': supervise_points,
                })
                patch_id += 1
                if args.max_patches > 0 and patch_id >= args.max_patches:
                    break
            if args.max_patches > 0 and patch_id >= args.max_patches:
                break
        if args.max_patches > 0 and patch_id >= args.max_patches:
            break

    for split in ['train', 'val', 'test']:
        if len(store[split]['input']) == 0:
            raise RuntimeError(f'no patches generated for split={split}; adjust split/block/patch params')

    out = args.out_dir
    patch_dir = out / 'patch_dataset'
    ensure_dir(patch_dir)

    for split in ['train', 'val', 'test']:
        np.savez_compressed(
            patch_dir / f'{split}_patches.npz',
            input=np.stack(store[split]['input']).astype(np.float32),
            target=np.stack(store[split]['target']).astype(np.float32),
            real_mask=np.stack(store[split]['real_mask']).astype(np.uint8),
            artificial_mask=np.stack(store[split]['artificial_mask']).astype(np.uint8),
            visible_mask=np.stack(store[split]['visible_mask']).astype(np.uint8),
            supervise_mask=np.stack(store[split]['supervise_mask']).astype(np.uint8),
            patch_id=np.concatenate(store[split]['patch_id']).astype(np.int64),
        )

    pd.DataFrame(rows).to_csv(out / 'patch_metadata.csv', index=False)
    pd.DataFrame([
        {'split': s, 'count': int(len(store[s]['input']))} for s in ['train', 'val', 'test']
    ]).to_csv(out / 'patch_split_summary.csv', index=False)

    np.save(out / 'dataset_8vars_china.npy', x_cn.astype(np.float32))
    np.save(out / 'mask_8vars_china.npy', m_cn.astype(np.uint8))

    info = {
        'actual_range': {
            'lat_min': float(lat_cn.min()),
            'lat_max': float(lat_cn.max()),
            'lon_min': float(lon_cn.min()),
            'lon_max': float(lon_cn.max()),
        },
        'index_range_global': {
            'lat_start': lat0,
            'lat_end': lat1 - 1,
            'lon_start': lon0,
            'lon_end': lon1 - 1,
        },
    }
    (out / 'china_region_info.json').write_text(json.dumps(info, indent=2), encoding='utf-8')

    protocol = {
        'dataset_name': 'china_strict_dataset',
        'input_unified_dir': str(args.unified_dir),
        'out_dir': str(args.out_dir),
        'strict_no_leak': True,
        'var_count': int(c),
        'depth_count': int(d),
        'china_hw': [int(h), int(w)],
        'patch': {'d': int(args.patch_depth), 'h': int(args.patch_lat), 'w': int(args.patch_lon)},
        'stride': {'d': int(args.stride_depth), 'h': int(args.stride_lat), 'w': int(args.stride_lon)},
        'ratios': {'train': float(args.train_ratio), 'val': float(args.val_ratio), 'test': float(1.0 - args.train_ratio - args.val_ratio)},
        'dropped': {k: int(v) for k, v in dropped.items()},
        'split_counts': {s: int(len(store[s]['input'])) for s in ['train', 'val', 'test']},
        'variable_order': data['var_names'],
    }
    (out / 'china_strict_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

    print(f'[DONE] strict dataset written to: {out}')


if __name__ == '__main__':
    main()





