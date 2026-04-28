from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback is intentionally narrow; PyYAML is expected for normal use.
    raise RuntimeError(f"Cannot parse YAML config without PyYAML: {path}")


def project_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def display_path(value: str | Path) -> str:
    return str(Path(value))


def add_common_args(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--dry-run", action="store_true", help="Validate config and print resolved run arguments without training.")
    return p


def bool_flag(enabled: bool, yes: str, no: str) -> str:
    return yes if bool(enabled) else no


def validate_paths(paths: Dict[str, Any], required_keys: Iterable[str]) -> List[str]:
    missing: List[str] = []
    for key in required_keys:
        value = paths.get(key)
        if value is None:
            missing.append(f"{key}: <not configured>")
            continue
        p = project_path(str(value))
        if not p.exists():
            missing.append(f"{key}: {p}")
    return missing


def print_dry_run(stage: str, config_path: Path, argv: List[str], missing: List[str]) -> None:
    payload = {
        "stage": stage,
        "config": display_path(config_path),
        "trainer_argv": argv,
        "missing_paths": missing,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def run_trainer(argv: List[str]) -> None:
    from aquasim.training import trainer

    old_argv = sys.argv[:]
    try:
        sys.argv = ["aquasim.training.trainer", *argv]
        trainer.main()
    finally:
        sys.argv = old_argv


def build_pretrain_argv(cfg: Dict[str, Any]) -> List[str]:
    paths = cfg["paths"]
    pre = cfg["pretrain"]
    model = cfg["model"]
    return [
        "--stage", "pretrain",
        "--data-dir", display_path(paths["global_pretrain_patch_dir"]),
        "--unified-dir", display_path(paths["unified_dir"]),
        "--out-dir", display_path("outputs/pretrain"),
        "--model-name", str(model["name"]),
        "--epochs", str(pre["max_epochs"]),
        "--batch-size", str(pre["batch_size"]),
        "--lr", str(pre["lr"]),
        "--weight-decay", str(pre["weight_decay"]),
        "--no-physics-loss",
        "--no-topo-constraint",
    ]


def build_finetune_argv(cfg: Dict[str, Any]) -> List[str]:
    paths = cfg["paths"]
    fine = cfg["finetune"]
    cons = cfg["constraints"]
    model = cfg["model"]
    argv = [
        "--stage", "finetune",
        "--data-dir", display_path(paths["china_strict_patch_dir"]),
        "--unified-dir", display_path(paths["unified_dir"]),
        "--out-dir", display_path(paths["train_output_dir"]),
        "--model-name", str(model["name"]),
        "--epochs", str(fine["max_epochs"]),
        "--batch-size", str(fine["batch_size"]),
        "--lr", str(fine["lr"]),
        "--weight-decay", str(fine["weight_decay"]),
        "--early-stop-patience", str(fine["early_stop_patience"]),
        "--min-delta", str(fine["min_delta"]),
        "--init-checkpoint", display_path(paths["pretrain_checkpoint"]),
        bool_flag(cons["use_physics_loss"], "--use-physics-loss", "--no-physics-loss"),
        bool_flag(cons["use_topo_constraint"], "--use-topo-constraint", "--no-topo-constraint"),
        "--topo-dir", display_path(paths["topo_runtime_dir"]),
        "--topo-metadata", display_path(paths["topo_metadata"]),
        "--loss-scale-json", display_path(paths["calibration_json"]),
        "--lambda-density", str(cons["lambda_density"]),
        "--lambda-vert", str(cons["lambda_vert"]),
        "--lambda-aou-o2", str(cons["lambda_aou_o2"]),
        "--lambda-np", str(cons["lambda_np"]),
        "--lambda-remin", str(cons["lambda_remin"]),
        "--lambda-tv", str(cons["lambda_tv"]),
        "--lambda-topo", str(cons["lambda_topo"]),
        "--topo-loss-lambda", str(cons["lambda_topo"]),
        "--topo-near-bottom-weight", str(cons["topo_near_bottom_weight"]),
        "--topo-slope-weight", str(cons["topo_slope_weight"]),
        "--topo-flat-enhance-weight", str(cons["topo_flat_enhance_weight"]),
    ]
    if bool(cons.get("topo_aware", False)):
        argv.append("--topo-aware")
    return argv
