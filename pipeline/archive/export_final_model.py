#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path


def main() -> None:
    out = Path("outputs/train")
    required = [out / "checkpoints" / "best_model.pt", out / "train_protocol.json", out / "test_metrics.csv"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing final model artifacts:\n" + "\n".join(missing))
    print(f"Final model archive is already materialized at: {out.resolve()}")


if __name__ == "__main__":
    main()
