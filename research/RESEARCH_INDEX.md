# Research Index

Source project:

```text
E:/python/MarineSim
```

## Key Outputs

| Item | MarineSim path | AquaSim handling |
| --- | --- | --- |
| Physics sweep | `E:/python/MarineSim/outputs/final_hparam_search/physics*` and `E:/python/MarineSim/sweep/` | Not bulk-copied; original path indexed. |
| Topo sweep | `E:/python/MarineSim/outputs/final_hparam_search/topo*` and `E:/python/MarineSim/sweep/` | Not bulk-copied; original path indexed. |
| Pretrain Optuna | `E:/python/MarineSim/outputs/final_hparam_search/pretrain_downstream_optuna/` | Large trial tree not copied; best checkpoint copied to `outputs/train/pretrain_best_model.pt`. |
| Finetune Optuna | `E:/python/MarineSim/outputs/final_hparam_search/finetune_optuna/` | Large search tree not copied; selected `trial_0028/finetune/` copied to `outputs/train/`. |
| Final config | `E:/python/MarineSim/configs/final_best_constraint_params.yaml` and final trial protocol | Consolidated into `configs/train.yaml`. |
| Final model source trial | `E:/python/MarineSim/outputs/final_hparam_search/finetune_optuna/trials/trial_0028/finetune/` | Copied to `outputs/train/`. |

## Copied Artifacts

- Final pretrain checkpoint: `outputs/train/pretrain_best_model.pt`.
- Final finetune directory contents: `outputs/train/checkpoints/`, `outputs/train/train_history.csv`, `outputs/train/test_metrics.csv`, `outputs/train/train_protocol.json`, `outputs/train/normalization_stats.npz`.
- Final processed datasets required by the main chain under `data/processed/`.

## Not Copied

Full historical search trial directories and non-selected large checkpoints were not copied. They remain referenced by original MarineSim paths above.
