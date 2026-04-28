#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    p = argparse.ArgumentParser(description="Build AquaSim unified 8-variable field tensors.")
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    p.add_argument("--out-dir", type=Path, default=Path("data/processed/unified_field"))
    p.add_argument("--save-nc", action="store_true")
    args = p.parse_args()

    from aquasim.core.data import unify_8vars

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "aquasim.core.data.unify_8vars",
            "--data-dir", str(args.raw_dir),
            "--output-dir", str(args.out_dir),
        ]
        if args.save_nc:
            sys.argv.append("--save-nc")
        unify_8vars.main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
