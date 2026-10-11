"""Importance weights that inject a shift of known size by resampling real rows."""

from __future__ import annotations

import numpy as np
import pandas as pd


def numeric_weights(x: np.ndarray, beta: float) -> np.ndarray:
    """Weights proportional to exp(beta * z), z = standard score clipped to [-3, 3]."""
    x = np.asarray(x, dtype=float)
    sd = x.std()
    z = np.clip((x - x.mean()) / sd, -3, 3) if sd > 0 else np.zeros_like(x)
    w = np.exp(beta * z)
    return w / w.sum()


def categorical_weights(x: pd.Series, category: str, delta: float) -> np.ndarray:
    """Weights that raise the share of `category` by `delta` (in share points)."""
    is_c = (pd.Series(x) == category).to_numpy()
    p0 = is_c.mean()
    p1 = min(p0 + delta, 0.999)
    w = np.where(is_c, p1 / p0, (1 - p1) / (1 - p0))
    return w / w.sum()
