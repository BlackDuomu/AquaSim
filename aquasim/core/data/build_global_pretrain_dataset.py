#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from aquasim.core.data.grid_utils import crop_bbox_indices, ensure_dir, get_starts, is_window_overlap, recover_lat_lon
from aquasim.core.data.mask_protocol import build_patch_masks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Build global clean pretrain dataset with China exclusion.')
    p.add_argument('--unified-dir', type=Path, default=Path('data/processed/unified_field'))
    p.add_argument('--out-dir', type=Path, default=Path('outputs/global_pretrain'))
    p.add_argument('--exclude-china', action='store_true', default=True)
    p.add_argument('--china-lat-min', type=float, default=0.0)
    p.add_argument('--china-lat-max', type=float, default=45.0)
    p.add_argument('--china-lon-min', type=float, default=105.0)
    p.add_argument('--china-lon-max', type=float, default=135.0)
    p.add_argument('--patch-depth', type=int, default=24)
    p.add_argument('--patch-lat', type=int, default=12)
    p.add_argument('--patch-lon', type=int, default=12)
    p.add_argument('--stride-depth', type=int, default=12)
    p.add_argument('--stride-lat', type=int, default=6)
    p.add_argument('--stride-lon', type=int, default=6)
    p.add_argument('--include-tail', action='store_true')
    p.add_argument('--artificial-ratio', type=float, default=0.4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min-ocean-cells-2d', type=int, default=20)
    p.add_argument('--min-supervise-points', type=int, default=80)
    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--val-ratio', type=float, default=0.1)
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

    x = np.load(unified_dir / 'dataset_8vars.npy', allow_pickle=False)
    m = np.load(unified_dir / 'mask_8vars.npy', allow_pickle=False)
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


def _assign_split(ch: int, cw: int, train_ratio: float, val_ratio: float, seed: int) -> str:
    key = (int(ch) * 1000003 + int(cw) * 9176 + int(seed) * 53) & 0xFFFFFFFF
    rng = np.random.default_rng(key)
    r = float(rng.random())
    if r < float(train_ratio):
        return 'train'
    if r < float(train_ratio + val_ratio):
        return 'val'
    return 'test'


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

    lat_idx, lon_idx = crop_bbox_indices(lat, lon, args.china_lat_min, args.china_lat_max, args.china_lon_min, args.china_lon_max)
    china_box = {
        'lat_start': int(lat_idx[0]),
        'lat_end': int(lat_idx[-1]),
        'lon_start': int(lon_idx[0]),
        'lon_end': int(lon_idx[-1]),
    }

    _, d, h, w = x.shape
    d_starts = get_starts(d, args.patch_depth, args.stride_depth, args.include_tail)
    h_starts = get_starts(h, args.patch_lat, args.stride_lat, args.include_tail)
    w_starts = get_starts(w, args.patch_lon, args.stride_lon, args.include_tail)

    rows: List[Dict[str, object]] = []
    store = {k: defaultdict(list) for k in ['train', 'val', 'test']}
    counters = Counter()
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

                counters['windows_total'] += 1
                if args.exclude_china and is_window_overlap(h0, h1, w0, w1, china_box):
                    counters['drop_china_overlap'] += 1
                    continue

                real_mask = np.array(m[:, d0:d1, h0:h1, w0:w1], dtype=np.uint8)
                valid2d = np.any(real_mask.astype(bool), axis=(0, 1))
                if int(valid2d.sum()) < int(args.min_ocean_cells_2d):
                    counters['drop_too_few_ocean_cells'] += 1
                    continue

                masks = build_patch_masks(real_mask=real_mask, ratio=args.artificial_ratio, seed=args.seed + patch_id)
                supervise_points = int(masks.supervise_mask.sum())
                if supervise_points < int(args.min_supervise_points):
                    counters['drop_too_few_supervise_points'] += 1
                    continue

                split = _assign_split(ch=ch, cw=cw, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
                target = np.array(x[:, d0:d1, h0:h1, w0:w1], dtype=np.float32)
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
                    'center_lat': float(lat[ch]),
                    'center_lon': float(lon[cw]),
                    'china_overlap_excluded': bool(args.exclude_china),
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
            raise RuntimeError(f'no patches generated for split={split}; adjust params')

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

    protocol = {
        'dataset_name': 'global_pretrain_dataset',
        'input_unified_dir': str(args.unified_dir),
        'out_dir': str(args.out_dir),
        'exclude_china': bool(args.exclude_china),
        'china_box': china_box,
        'patch': {'d': int(args.patch_depth), 'h': int(args.patch_lat), 'w': int(args.patch_lon)},
        'stride': {'d': int(args.stride_depth), 'h': int(args.stride_lat), 'w': int(args.stride_lon)},
        'ratios': {'train': float(args.train_ratio), 'val': float(args.val_ratio), 'test': float(1.0 - args.train_ratio - args.val_ratio)},
        'split_counts': {s: int(len(store[s]['input'])) for s in ['train', 'val', 'test']},
        'counters': {k: int(v) for k, v in counters.items()},
        'variable_order': data['var_names'],
    }
    (out / 'global_pretrain_protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')

    print(f'[DONE] global pretrain dataset written to: {out}')


if __name__ == '__main__':
    main()





