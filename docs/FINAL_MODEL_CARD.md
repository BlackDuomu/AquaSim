# Final Model Card

## Model

Name:

```text
marine3d_transformer
```

Final model directory:

```text
outputs/train/
```

Final checkpoint:

```text
outputs/train/checkpoints/best_model.pt
```

Pretrain checkpoint copied for provenance and reuse:

```text
outputs/train/pretrain_best_model.pt
```

## Training Data

- Pretrain data: `data/processed/global_pretrain_patches/`
- Finetune data: `data/processed/china_strict_patches/`
- Unified field: `data/processed/unified_field/`
- Topo runtime: `data/processed/topo_runtime/`

## Final Parameters

Pretrain:

```text
max_epochs=180
batch_size=8
lr=0.004651368090377602
weight_decay=0.00014733468288150573
```

Finetune:

```text
max_epochs=120
batch_size=12
lr=2.0805084264202466e-05
weight_decay=0.00014189277631132947
early_stop_patience=20
min_delta=1e-5
```

Constraints:

```text
use_physics_loss=true
use_topo_constraint=true
topo_aware=true
use_loss_calibration=true
lambda_density=0.01512733041002596
lambda_vert=0.000655552331151622
lambda_aou_o2=2.4710592780647456e-07
lambda_np=0.0
lambda_remin=0.0
lambda_tv=0.0012149185105664079
lambda_topo=0.04149734431721462
topo_near_bottom_weight=1.1074612052584936
topo_slope_weight=0.7575076729791065
topo_flat_enhance_weight=0.3314379454098823
```

## Final Metrics

| Metric | Value |
| --- | ---: |
| objective | 0.03316874512456951 |
| test_rmse | 0.11473642487466575 |
| test_mae | 0.07867888937795894 |
| near_bottom_rmse | 0.1092749935656131 |
| slope_area_rmse | 0.11321877600181368 |
| topo_consistency_score | 0.5077207916288763 |
| topo_metric_fallback | false |

## Provenance

- Source project: `E:/python/MarineSim`
- Best pretrain trial: `outputs/final_hparam_search/pretrain_downstream_optuna/trials/trial_0029`
- Best finetune trial: `outputs/final_hparam_search/finetune_optuna/trials/trial_0028`

This model is the current best under the completed search budget. It is not a mathematical proof of global optimum.

## Missing Requested Final Files

None of the requested selected finetune files were missing at migration time:

- `checkpoints/best_model.pt`
- `checkpoints/last_model.pt`
- `train_history.csv`
- `test_metrics.csv`
- `train_protocol.json`
- `normalization_stats.npz`
