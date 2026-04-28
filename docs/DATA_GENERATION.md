# Data Generation

## Raw Data

Raw WOA23 data is stored in:

```text
data/raw/
```

It was copied from:

```text
E:/python/MarineSim/data/raw/
```

Topography source data is stored in:

```text
data/topo/
```

It was copied from:

```text
E:/python/MarineSim/data/topo/
```

## Unified Field

The old MarineSim output name `outputs/step1` is renamed in AquaSim to:

```text
data/processed/unified_field/
```

This name describes the artifact more directly: unified 8-variable ocean-field tensor, mask, depth levels, coordinate range, and variable mapping.

Build command:

```bash
python pipeline/data/build_unified_field.py \
  --raw-dir data/raw \
  --out-dir data/processed/unified_field
```

## China Strict Patches

Current generated data is stored in:

```text
data/processed/china_strict_patches/
```

Rebuild command:

```bash
python pipeline/data/build_china_strict_patches.py \
  --unified-dir data/processed/unified_field \
  --out-dir data/processed/china_strict_patches \
  --lat-min 0 --lat-max 45 \
  --lon-min 105 --lon-max 135 \
  --patch-depth 24 \
  --patch-lat 10 \
  --patch-lon 10 \
  --stride-depth 12 \
  --stride-lat 1 \
  --stride-lon 1 \
  --artificial-ratio 0.4 \
  --seed 42 \
  --min-ocean-cells-2d 16 \
  --min-supervise-points 40 \
  --train-ratio 0.7 \
  --val-ratio 0.15 \
  --block-lat 15 \
  --block-lon 15
```

## Global Pretrain Patches

Current generated data is stored in:

```text
data/processed/global_pretrain_patches/
```

Rebuild command:

```bash
python pipeline/data/build_global_pretrain_patches.py \
  --unified-dir data/processed/unified_field \
  --out-dir data/processed/global_pretrain_patches \
  --exclude-china \
  --china-lat-min 0 \
  --china-lat-max 45 \
  --china-lon-min 105 \
  --china-lon-max 135 \
  --patch-depth 24 \
  --patch-lat 10 \
  --patch-lon 10 \
  --stride-depth 12 \
  --stride-lat 6 \
  --stride-lon 6 \
  --artificial-ratio 0.4 \
  --seed 42 \
  --min-ocean-cells-2d 16 \
  --min-supervise-points 64 \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

## Topo Runtime

Current generated data is stored in:

```text
data/processed/topo_runtime/
```

Rebuild command:

```bash
python pipeline/data/prepare_topography_runtime.py \
  --topo-file data/topo/ETOPO1_Bed_g_gdal.grd \
  --unified-dir data/processed/unified_field \
  --patch-metadata data/processed/china_strict_patches/patch_metadata.csv \
  --out-dir data/processed/topo_runtime
```

The public AquaSim pipeline uses `--unified-dir`; the copied historical implementation behind the wrapper still contains compatibility names internally.
