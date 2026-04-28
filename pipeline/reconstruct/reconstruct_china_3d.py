#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquasim.reconstruction.china_3d import ReconstructionPaths, run_china_3d_reconstruction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build China 3D reconstruction from the trained AquaSim model.")
    p.add_argument("--china-patch-dir", type=Path, default=Path("data/processed/china_strict_patches"))
    p.add_argument("--unified-dir", type=Path, default=Path("data/processed/unified_field"))
    p.add_argument("--topo-runtime-dir", type=Path, default=Path("data/processed/topo_runtime"))
    p.add_argument("--checkpoint", type=Path, default=Path("outputs/train/checkpoints/best_model.pt"))
    p.add_argument("--norm-stats", type=Path, default=Path("outputs/train/normalization_stats.npz"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/reconstruction/china_3d"))
    p.add_argument("--model-name", type=str, default="marine3d_transformer")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated patch splits to reconstruct.")
    p.add_argument("--source-mode", choices=["full_grid", "patch_splits"], default="full_grid")
    p.add_argument("--include-tail", dest="include_tail", action="store_true")
    p.add_argument("--no-include-tail", dest="include_tail", action="store_false")
    p.set_defaults(include_tail=True)
    p.add_argument("--force-cpu", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = ReconstructionPaths(
        china_patch_dir=args.china_patch_dir,
        unified_dir=args.unified_dir,
        topo_runtime_dir=args.topo_runtime_dir,
        checkpoint_path=args.checkpoint,
        norm_stats_path=args.norm_stats,
        out_dir=args.out_dir,
    )
    out = run_china_3d_reconstruction(
        paths=paths,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        splits=[s.strip() for s in args.splits.split(",")],
        source_mode=args.source_mode,
        include_tail=bool(args.include_tail),
        force_cpu=bool(args.force_cpu),
    )
    print(f"[DONE] China 3D reconstruction saved: {out}")


if __name__ == "__main__":
    main()
