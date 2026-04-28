#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquasim.data.dataset import (
    ChannelNormStats,
    Step6PatchDataset,
    apply_channel_norm,
    load_all_splits,
    load_topo_runtime_assets,
)
from aquasim.models.backbone3d import build_marine_model
from aquasim.training.trainer import _evaluate_test


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an AquaSim checkpoint against corrected runtime masks.")
    p.add_argument("--data-dir", type=Path, default=Path("data/processed/china_strict_patches"))
    p.add_argument("--unified-dir", type=Path, default=Path("data/processed/unified_field"))
    p.add_argument("--topo-dir", type=Path, default=Path("data/processed/topo_runtime"))
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/train/checkpoints/best_model.pt"))
    p.add_argument("--norm-stats", type=Path, default=Path("outputs/train/normalization_stats.npz"))
    p.add_argument("--out-csv", type=Path, default=Path("outputs/evaluation/corrected_topo_checkpoint_metrics.csv"))
    p.add_argument("--model-name", type=str, default="marine3d_transformer")
    p.add_argument("--batch-size", type=int, default=12)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-test-batches", type=int, default=0)
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


def _load_norm_stats(path: Path, channels: int) -> ChannelNormStats:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as payload:
        mean = payload["mean"].astype(np.float32)
        std = payload["std"].astype(np.float32)
    if mean.shape != (channels,) or std.shape != (channels,):
        raise ValueError(f"normalization stats shape mismatch: {mean.shape}, {std.shape}, channels={channels}")
    return ChannelNormStats(mean=mean, std=std)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.force_cpu or not torch.cuda.is_available() else "cuda")

    train_raw, _, test_raw = load_all_splits(args.data_dir)
    channels = int(train_raw.target.shape[1])
    stats = _load_norm_stats(args.norm_stats, channels=channels)
    test_split = apply_channel_norm(test_raw, stats)
    topo_assets = load_topo_runtime_assets(
        topo_dir=args.topo_dir,
        data_dir=args.data_dir,
        unified_dir=args.unified_dir,
        required=True,
    )
    loader = DataLoader(
        Step6PatchDataset(test_split, topo_runtime_assets=topo_assets, return_topo_runtime=True),
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )

    model = build_marine_model(
        model_name=args.model_name,
        in_channels=channels * 2,
        out_channels=channels,
        base_channels=32,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)

    metrics = _evaluate_test(
        model=model,
        loader=loader,
        device=device,
        max_batches=int(args.max_test_batches),
        non_blocking_transfer=False,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(args.out_csv, index=False)
    print(f"[DONE] checkpoint metrics saved: {args.out_csv}")


if __name__ == "__main__":
    main()
