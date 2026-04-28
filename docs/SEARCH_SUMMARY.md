# Search Summary

## Physics Search

The mature MarineSim search compared the full physics branch against the `no_np_remin` branch. The selected winner is:

```text
no_np_remin
```

The final configuration sets:

```text
lambda_np = 0.0
lambda_remin = 0.0
```

## Topo Search

Topo search proceeded through coarse, fine, and fine2 phases. The final selected topo parameters are recorded in `configs/train.yaml`.

## Optuna Selection

Best pretrain Optuna trial:

```text
trial_0029
```

Best finetune Optuna trial:

```text
trial_0028
```

## Current Final Metrics

| Metric | Value |
| --- | ---: |
| objective | 0.03316874512456951 |
| test_rmse | 0.11473642487466575 |
| test_mae | 0.07867888937795894 |
| near_bottom_rmse | 0.1092749935656131 |
| slope_area_rmse | 0.11321877600181368 |
| topo_consistency_score | 0.5077207916288763 |
| topo_metric_fallback | false |

This is the best model under the completed search budget, not a mathematical proof of global optimum.
