#!/usr/bin/env python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "project_name": "AquaSim",
        "source_project": "E:/python/MarineSim",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "final_model_dir": "outputs/train",
        "final_config": "configs/train.yaml",
    }
    (root / "PROJECT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(root / "PROJECT_MANIFEST.json")


if __name__ == "__main__":
    main()
