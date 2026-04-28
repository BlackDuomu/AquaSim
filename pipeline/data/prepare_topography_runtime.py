#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Prepare AquaSim topography runtime arrays.")
    p.add_argument("--topo-file", type=Path, default=Path("data/topo/ETOPO1_Bed_g_gdal.grd"))
    p.add_argument("--unified-dir", type=Path, default=Path("data/processed/unified_field"))
    p.add_argument("--patch-metadata", type=Path, default=Path("data/processed/china_strict_patches/patch_metadata.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/topo_runtime"))
    p.add_argument("--no-figures", action="store_true")
    args = p.parse_args()

    from aquasim.core.topo import prepare_topo_features

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "aquasim.core.topo.prepare_topo_features",
            "--topo-file", str(args.topo_file),
            "--unified-dir", str(args.unified_dir),
            "--patch-metadata", str(args.patch_metadata),
            "--out-dir", str(args.out_dir),
        ]
        if args.no_figures:
            sys.argv.append("--no-figures")
        prepare_topo_features.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
