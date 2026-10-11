"""Classification and calibration metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def reliability_curve(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict[str, list]:
    """Equal-width reliability curve. Empty bins are left out."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    mean_pred, frac_pos, count = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            mean_pred.append(float(p[m].mean()))
            frac_pos.append(float(y[m].mean()))
            count.append(int(m.sum()))
    return {"mean_predicted": mean_pred, "fraction_positive": frac_pos, "count": count}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    curve = reliability_curve(y, p, n_bins)
    n = sum(curve["count"])
    return float(
        sum(
            c / n * abs(mp - fp)
            for mp, fp, c in zip(
                curve["mean_predicted"], curve["fraction_positive"], curve["count"], strict=True
            )
        )
    )


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y)
    p = np.asarray(p)
    return {
        "n": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_score": float(np.mean(p)),
    }


def bootstrap_auc_ci(
    y: np.ndarray, p: np.ndarray, n_boot: int = 500, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    p = np.asarray(p)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
