#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.scoring import compute_final_score_bundle, compute_topo_score_from_metrics, save_score_bundle


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compute a compact evaluation score from test_metrics.csv')
    p.add_argument('--test-metrics', type=Path, required=True)
    p.add_argument('--out-json', type=Path, required=True)
    p.add_argument('--physics-score', type=float, default=0.0)
    p.add_argument('--topo-score', type=float, default=float('nan'))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.test_metrics.exists():
        raise FileNotFoundError(args.test_metrics)
    df = pd.read_csv(args.test_metrics)
    if df.empty or 'rmse' not in df.columns:
        raise ValueError('test_metrics.csv missing rmse')
    metrics = df.iloc[0].to_dict()
    rmse = float(metrics['rmse'])
    auto_topo = compute_topo_score_from_metrics(metrics)
    topo_score = float(args.topo_score) if math.isfinite(float(args.topo_score)) else float(auto_topo)
    bundle = compute_final_score_bundle(test_rmse=rmse, physics_score=args.physics_score, topo_score=topo_score)
    bundle['topo_score_source'] = 'manual_override' if math.isfinite(float(args.topo_score)) else 'computed_from_metrics'
    if 'near_bottom_rmse' in metrics:
        bundle['near_bottom_rmse'] = float(metrics['near_bottom_rmse'])
    if 'slope_area_rmse' in metrics:
        bundle['slope_area_rmse'] = float(metrics['slope_area_rmse'])
    if 'topo_consistency_score' in metrics:
        bundle['topo_consistency_score'] = float(metrics['topo_consistency_score'])
    save_score_bundle(args.out_json, bundle)
    print(f'[DONE] score saved: {args.out_json}')


if __name__ == '__main__':
    main()
