# AquaSim

AquaSim is the cleaned, reproducible engineering archive for the China coastal 3D ocean-field reconstruction workflow matured in `E:/python/MarineSim`.

The production naming replaces the old `outputs/step1` concept with:

```text
data/processed/unified_field/
```

Final training entrypoint:

```bash
python pipeline/train/train.py --config configs/train.yaml
```

Dry-run validation:

```bash
python pipeline/train/train.py --config configs/train.yaml --dry-run
```

The final selected model artifacts are under `outputs/train/`. Research search outputs remain indexed under `research/RESEARCH_INDEX.md`; large historical search checkpoints were not bulk-copied.
