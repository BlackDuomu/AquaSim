from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PatchMaskBundle:
    real_mask: np.ndarray
    artificial_mask: np.ndarray
    visible_mask: np.ndarray
    supervise_mask: np.ndarray


def build_patch_masks(real_mask: np.ndarray, ratio: float, seed: int) -> PatchMaskBundle:
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError('ratio must be in (0,1)')
    real = (real_mask > 0).astype(np.uint8)
    rng = np.random.default_rng(int(seed))
    sampled = (rng.random(real.shape, dtype=np.float32) < float(ratio)).astype(np.uint8)
    artificial = (sampled & real).astype(np.uint8)
    visible = ((real == 1) & (artificial == 0)).astype(np.uint8)
    supervise = artificial.copy()
    return PatchMaskBundle(real_mask=real, artificial_mask=artificial, visible_mask=visible, supervise_mask=supervise)



