#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aquasim.visualization.reconstruction_figures import (
    ReconstructionFigurePaths,
    generate_reconstruction_figures,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate publication figures from AquaSim reconstructed China 3D fields.")
    p.add_argument("--reconstruction", type=Path, default=Path("outputs/reconstruction/china_3d/china_3d_reconstruction.npz"))
    p.add_argument("--topo-runtime-dir", type=Path, default=Path("data/processed/topo_runtime"))
    p.add_argument("--china-patch-dir", type=Path, default=Path("data/processed/china_strict_patches"))
    p.add_argument("--out-dir", type=Path, default=Path("outputs/figures/reconstruction_analysis"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outputs = generate_reconstruction_figures(
        ReconstructionFigurePaths(
            reconstruction_npz=args.reconstruction,
            topo_runtime_dir=args.topo_runtime_dir,
            china_patch_dir=args.china_patch_dir,
            out_dir=args.out_dir,
        )
    )
    for name, path in outputs.items():
        print(f"[FIGURE] {name}: {path}")


if __name__ == "__main__":
    main()

