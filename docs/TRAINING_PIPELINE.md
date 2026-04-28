# Training Pipeline

## Pretrain

Pretraining uses global pretrain patches:

```text
data/processed/global_pretrain_patches/
```

Default command:

```bash
python pipeline/train/pretrain.py --config configs/train.yaml
```

The pretrain stage does not enable physics or topo constraints by default.

## Finetune

Finetuning uses China strict patches:

```text
data/processed/china_strict_patches/
```

It initializes from:

```text
outputs/train/pretrain_best_model.pt
```

The final finetune path enables physics loss, topo constraint, topo-aware weighting, and loss calibration.

Final training command:

```bash
python pipeline/train/train.py --config configs/train.yaml
```

Dry-run command:

```bash
python pipeline/train/train.py --config configs/train.yaml --dry-run
```

Final configuration:

```text
configs/train.yaml
```
