#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_ROOT = Path("outputs/search/pretrain_downstream_optuna")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Optuna search for unconstrained world-pretrain hyperparameters, "
            "scored only by downstream strict China finetune results."
        )
    )
    p.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    p.add_argument("--global-pretrain-dir", type=Path, default=Path("data/processed/global_pretrain_patches"))
    p.add_argument("--china-strict-dir", type=Path, default=Path("data/processed/china_strict_patches"))
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--study-name", type=str, default="pretrain_downstream_v1")
    p.add_argument(
        "--storage",
        type=str,
        default="sqlite:///outputs/search/pretrain_downstream_optuna/study.db",
        help="Optuna storage URL. Use sqlite:///... for resumable local studies.",
    )
    p.add_argument("--resume", action="store_true", help="Resume an existing Optuna study.")
    p.add_argument("--fixed-finetune-epochs", type=int, default=20)
    p.add_argument("--fixed-finetune-batch-size", type=int, default=8)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--min-delta", type=float, default=1e-5)
    p.add_argument("--num-workers", type=int, default=16)
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    p.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    p.add_argument("--cudnn-benchmark", dest="cudnn_benchmark", action="store_true")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    p.set_defaults(pin_memory=True)
    p.set_defaults(cudnn_benchmark=True)
    p.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    p.add_argument("--force-cpu", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def _read_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    if not isinstance(obj, dict):
        raise ValueError("config must be a mapping")
    return obj


def _ensure_real_paths(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    required = {
        "global_pretrain_dir": args.global_pretrain_dir,
        "china_strict_dir": args.china_strict_dir,
    }
    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("config.paths must be a mapping")
    for key in ["unified_dir", "topo_runtime_dir", "topo_metadata", "calibration_json"]:
        required[f"config.paths.{key}"] = Path(str(paths.get(key, "")))
    for key, path in required.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"missing required path for `{key}`: {path}")


def _sample_params(trial: Any) -> Dict[str, Any]:
    return {
        "pretrain_max_epochs": trial.suggest_int("pretrain_max_epochs", 100, 240, step=20),
        "pretrain_batch_size": trial.suggest_categorical("pretrain_batch_size_v2", [8, 12, 16]),
        "pretrain_lr": trial.suggest_float("pretrain_lr", 1e-3, 5e-3, log=True),
        "pretrain_weight_decay": trial.suggest_float("pretrain_weight_decay", 3e-5, 8e-4, log=True),
    }


def _dump_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def _read_first_csv_row(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out: Dict[str, float] = {}
    for key, value in rows[0].items():
        if value is None or str(value).strip() == "":
            continue
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def _read_best_val(history_path: Path) -> float:
    if not history_path.exists():
        return float("nan")
    df = pd.read_csv(history_path)
    if df.empty or "val_loss" not in df.columns:
        return float("nan")
    return float(pd.to_numeric(df["val_loss"], errors="coerce").min())


def _finite_or(value: Optional[float], fallback: float) -> float:
    if value is None:
        return float(fallback)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(fallback)
    return v if math.isfinite(v) else float(fallback)


def _build_pretrain_cmd(args: argparse.Namespace, params: Dict[str, Any], pretrain_dir: Path, cfg: Dict[str, Any]) -> List[str]:
    paths = cfg.get("paths", {})
    model_name = str(cfg.get("model_name", "marine3d_transformer"))
    cmd = [
        str(args.python_exe),
        str(ROOT / "aquasim" / "training" / "trainer.py"),
        "--stage",
        "pretrain",
        "--model-name",
        model_name,
        "--data-dir",
        str(args.global_pretrain_dir),
        "--unified-dir",
        str(Path(str(paths.get("unified_dir", "data/processed/unified_field")))),
        "--out-dir",
        str(pretrain_dir),
        "--epochs",
        str(int(params["pretrain_max_epochs"])),
        "--batch-size",
        str(int(params["pretrain_batch_size"])),
        "--lr",
        str(float(params["pretrain_lr"])),
        "--weight-decay",
        str(float(params["pretrain_weight_decay"])),
        "--num-workers",
        str(int(args.num_workers)),
        "--early-stop-patience",
        str(int(args.early_stop_patience)),
        "--min-delta",
        str(float(args.min_delta)),
        "--no-physics-loss",
        "--no-topo-constraint",
    ]
    if args.force_cpu:
        cmd.append("--force-cpu")
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.cudnn_benchmark:
        cmd.append("--cudnn-benchmark")
    else:
        cmd.append("--no-cudnn-benchmark")
    return cmd


def _build_finetune_cmd(args: argparse.Namespace, ckpt: Path, finetune_dir: Path, cfg: Dict[str, Any]) -> List[str]:
    paths = cfg.get("paths", {})
    params = cfg.get("constraint_params", {})
    if not isinstance(paths, dict) or not isinstance(params, dict):
        raise ValueError("config paths and constraint_params must be mappings")
    physics = params.get("physics", {})
    topo = params.get("topo", {})
    if not isinstance(physics, dict) or not isinstance(topo, dict):
        raise ValueError("constraint_params.physics/topo must be mappings")

    cmd = [
        str(args.python_exe),
        str(ROOT / "aquasim" / "training" / "trainer.py"),
        "--stage",
        "finetune",
        "--model-name",
        str(cfg.get("model_name", "marine3d_transformer")),
        "--data-dir",
        str(args.china_strict_dir),
        "--unified-dir",
        str(Path(str(paths["unified_dir"]))),
        "--topo-dir",
        str(Path(str(paths["topo_runtime_dir"]))),
        "--topo-metadata",
        str(Path(str(paths["topo_metadata"]))),
        "--out-dir",
        str(finetune_dir),
        "--init-checkpoint",
        str(ckpt),
        "--epochs",
        str(int(args.fixed_finetune_epochs)),
        "--batch-size",
        str(int(args.fixed_finetune_batch_size)),
        "--num-workers",
        str(int(args.num_workers)),
        "--use-physics-loss",
        "--use-topo-constraint",
        "--loss-scale-json",
        str(Path(str(paths["calibration_json"]))),
        "--lambda-density",
        str(float(physics["lambda_density"])),
        "--lambda-vert",
        str(float(physics["lambda_vert"])),
        "--lambda-aou-o2",
        str(float(physics["lambda_aou_o2"])),
        "--lambda-np",
        str(float(physics["lambda_np"])),
        "--lambda-remin",
        str(float(physics["lambda_remin"])),
        "--lambda-tv",
        str(float(physics["lambda_tv"])),
        "--lambda-topo",
        str(float(topo["lambda_topo"])),
        "--topo-near-bottom-weight",
        str(float(topo["topo_near_bottom_weight"])),
        "--topo-slope-weight",
        str(float(topo["topo_slope_weight"])),
        "--topo-flat-enhance-weight",
        str(float(topo["topo_flat_enhance_weight"])),
    ]
    if args.force_cpu:
        cmd.append("--force-cpu")
    if args.pin_memory:
        cmd.append("--pin-memory")
    if args.cudnn_benchmark:
        cmd.append("--cudnn-benchmark")
    else:
        cmd.append("--no-cudnn-benchmark")
    return cmd


def _run_cmd(cmd: List[str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(cwd), check=False).returncode
    if rc != 0:
        raise RuntimeError(f"command failed with return code {rc}: {' '.join(cmd)}")


def _compose_result(
    trial_number: int,
    params: Dict[str, Any],
    args: argparse.Namespace,
    trial_dir: Path,
    status: str,
    error_message: str = "",
) -> Dict[str, Any]:
    pretrain_dir = trial_dir / "pretrain"
    finetune_dir = trial_dir / "finetune"
    pretrain_best_val = _read_best_val(pretrain_dir / "train_history.csv")
    finetune_best_val = _read_best_val(finetune_dir / "train_history.csv")
    metrics = _read_first_csv_row(finetune_dir / "test_metrics.csv")

    test_rmse = _finite_or(metrics.get("rmse"), float("inf"))
    near_bottom_rmse = _finite_or(metrics.get("near_bottom_rmse"), test_rmse)
    slope_area_rmse = _finite_or(metrics.get("slope_area_rmse"), test_rmse)
    topo_consistency_score = _finite_or(metrics.get("topo_consistency_score"), 0.0)
    topo_metric_fallback = not all(
        key in metrics and math.isfinite(float(metrics[key]))
        for key in ["near_bottom_rmse", "slope_area_rmse", "topo_consistency_score"]
    )

    topo_rmse_term = 0.5 * (near_bottom_rmse + slope_area_rmse)
    objective = (
        0.5 * test_rmse
        + 0.2 * _finite_or(finetune_best_val, 0.0)
        + 0.2 * topo_rmse_term
        - 0.1 * topo_consistency_score
    )
    if status != "completed":
        objective = float("inf")

    return {
        "trial_number": int(trial_number),
        "pretrain_max_epochs": int(params["pretrain_max_epochs"]),
        "pretrain_batch_size": int(params["pretrain_batch_size"]),
        "pretrain_lr": float(params["pretrain_lr"]),
        "pretrain_weight_decay": float(params["pretrain_weight_decay"]),
        "pretrain_best_checkpoint": str(pretrain_dir / "checkpoints" / "best_model.pt"),
        "pretrain_best_val_loss": pretrain_best_val,
        "fixed_finetune_epochs": int(args.fixed_finetune_epochs),
        "fixed_finetune_batch_size": int(args.fixed_finetune_batch_size),
        "finetune_best_val_loss": finetune_best_val,
        "test_rmse": test_rmse,
        "test_mae": _finite_or(metrics.get("mae"), float("nan")),
        "near_bottom_rmse": near_bottom_rmse,
        "slope_area_rmse": slope_area_rmse,
        "topo_consistency_score": topo_consistency_score,
        "topo_metric_fallback": bool(topo_metric_fallback),
        "objective": objective,
        "status": status,
        "error_message": error_message,
    }


def _write_summary(out_root: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for result_path in sorted((out_root / "trials").glob("trial_*/trial_result.json")):
        try:
            rows.append(json.loads(result_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    summary_path = out_root / "pretrain_downstream_optuna_summary.csv"
    best_path = out_root / "best_pretrain_downstream.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "trial_number",
        "pretrain_max_epochs",
        "pretrain_batch_size",
        "pretrain_lr",
        "pretrain_weight_decay",
        "pretrain_best_checkpoint",
        "pretrain_best_val_loss",
        "fixed_finetune_epochs",
        "fixed_finetune_batch_size",
        "finetune_best_val_loss",
        "test_rmse",
        "test_mae",
        "near_bottom_rmse",
        "slope_area_rmse",
        "topo_consistency_score",
        "topo_metric_fallback",
        "objective",
        "status",
        "error_message",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    completed = [
        row for row in rows
        if row.get("status") == "completed" and math.isfinite(_finite_or(row.get("objective"), float("inf")))
    ]
    if completed:
        best = min(completed, key=lambda row: float(row["objective"]))
        _dump_json(best_path, best)


def _dry_run(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    class DemoTrial:
        def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> int:
            return low + ((high - low) // (2 * step)) * step

        def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
            return choices[-1]

        def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
            return math.sqrt(low * high) if log else (low + high) / 2.0

    params = _sample_params(DemoTrial())
    trial_dir = args.out_root / "trials" / "trial_0000"
    pretrain_dir = trial_dir / "pretrain"
    finetune_dir = trial_dir / "finetune"
    ckpt = pretrain_dir / "checkpoints" / "best_model.pt"
    payload = {
        "trial_params_example": params,
        "trial_output_dir": str(trial_dir),
        "pretrain_command": _build_pretrain_cmd(args, params, pretrain_dir, cfg),
        "finetune_command": _build_finetune_cmd(args, ckpt, finetune_dir, cfg),
        "summary_csv": str(args.out_root / "pretrain_downstream_optuna_summary.csv"),
        "best_json": str(args.out_root / "best_pretrain_downstream.json"),
        "storage": args.storage,
        "study_name": args.study_name,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    cfg = _read_config(args.config)
    args.out_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _dry_run(args, cfg)
        return

    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Optuna is required. Install it in the active environment before running this search.") from exc

    _ensure_real_paths(args, cfg)
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.resume),
    )

    def objective(trial: Any) -> float:
        params = _sample_params(trial)
        trial_dir = args.out_root / "trials" / f"trial_{trial.number:04d}"
        pretrain_dir = trial_dir / "pretrain"
        finetune_dir = trial_dir / "finetune"
        trial_dir.mkdir(parents=True, exist_ok=True)
        _dump_json(trial_dir / "trial_params.json", params)

        try:
            pretrain_cmd = _build_pretrain_cmd(args, params, pretrain_dir, cfg)
            _run_cmd(pretrain_cmd, ROOT)
            pretrain_ckpt = pretrain_dir / "checkpoints" / "best_model.pt"
            if not pretrain_ckpt.exists():
                raise FileNotFoundError(f"missing pretrain best checkpoint: {pretrain_ckpt}")

            finetune_cmd = _build_finetune_cmd(args, pretrain_ckpt, finetune_dir, cfg)
            _run_cmd(finetune_cmd, ROOT)

            result = _compose_result(trial.number, params, args, trial_dir, status="completed")
            _dump_json(trial_dir / "trial_result.json", result)
            _write_summary(args.out_root)
            for key, value in result.items():
                if key not in {"error_message"}:
                    trial.set_user_attr(key, value)
            return float(result["objective"])
        except Exception as exc:
            error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            result = _compose_result(trial.number, params, args, trial_dir, status="failed", error_message=error_message)
            result["traceback"] = traceback.format_exc()
            _dump_json(trial_dir / "trial_result.json", result)
            _write_summary(args.out_root)
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("error_message", error_message)
            raise

    study.optimize(objective, n_trials=int(args.n_trials), catch=(Exception,))
    _write_summary(args.out_root)
    completed = [t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
    if completed:
        print(f"[DONE] study={args.study_name} best_value={study.best_value:.6f} best_trial={study.best_trial.number}")
    else:
        print(f"[DONE] study={args.study_name} has no completed trials yet")
    print(f"[SUMMARY] {args.out_root / 'pretrain_downstream_optuna_summary.csv'}")
    print(f"[BEST] {args.out_root / 'best_pretrain_downstream.json'}")


if __name__ == "__main__":
    main()

