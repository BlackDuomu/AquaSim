#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Build global pretrain patch dataset from unified field tensors.")
    p.add_argument("--unified-dir", type=Path, default=Path("data/processed/unified_field"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/global_pretrain_patches"))
    p.add_argument("--exclude-china", action="store_true", default=True)
    p.add_argument("--china-lat-min", type=float, default=0.0)
    p.add_argument("--china-lat-max", type=float, default=45.0)
    p.add_argument("--china-lon-min", type=float, default=105.0)
    p.add_argument("--china-lon-max", type=float, default=135.0)
    p.add_argument("--patch-depth", type=int, default=24)
    p.add_argument("--patch-lat", type=int, default=10)
    p.add_argument("--patch-lon", type=int, default=10)
    p.add_argument("--stride-depth", type=int, default=12)
    p.add_argument("--stride-lat", type=int, default=6)
    p.add_argument("--stride-lon", type=int, default=6)
    p.add_argument("--artificial-ratio", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-ocean-cells-2d", type=int, default=16)
    p.add_argument("--min-supervise-points", type=int, default=64)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--max-patches", type=int, default=0)
    args = p.parse_args()

    from aquasim.core.data import build_global_pretrain_dataset

    old_argv = sys.argv[:]
    try:
        sys.argv = ["aquasim.core.data.build_global_pretrain_dataset", "--unified-dir", str(args.unified_dir), "--out-dir", str(args.out_dir)]
        for name, value in vars(args).items():
            if name in {"unified_dir", "out_dir"}:
                continue
            flag = "--" + name.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    sys.argv.append(flag)
            else:
                sys.argv.extend([flag, str(value)])
        build_global_pretrain_dataset.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
