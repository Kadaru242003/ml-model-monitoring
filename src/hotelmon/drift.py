"""Drift statistics: PSI, two-sample KS, and chi-square.

All functions compare a reference sample to a current sample.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

PSI_EPS = 1e-4
PSI_MODERATE = 0.1
PSI_MAJOR = 0.2


def psi_from_proportions(ref_p: np.ndarray, cur_p: np.ndarray, eps: float = PSI_EPS) -> float:
    """Population Stability Index: sum((cur - ref) * ln(cur / ref)).

    Empty bins are floored at eps so the log stays finite.
    """
    r = np.clip(np.asarray(ref_p, dtype=float), eps, None)
    c = np.clip(np.asarray(cur_p, dtype=float), eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def quantile_edges(ref: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Inner bin edges at reference quantiles. Ties collapse, so discrete data gets fewer bins.

    Edge k is the ceil(k * n / n_bins)-th smallest value (the inverted-CDF quantile),
    computed with integer arithmetic so there is no interpolation or rounding error.
    """
    x = np.sort(np.asarray(ref, dtype=float))
    n = len(x)
    k = np.arange(1, n_bins)
    idx = (k * n + n_bins - 1) // n_bins - 1
    return np.unique(x[idx])


def psi_numeric(ref: np.ndarray, cur: np.ndarray, n_bins: int = 10) -> float:
    """PSI with bins at reference deciles. Bins are right-closed: (edge_i, edge_i+1]."""
    ref = np.asarray(ref, dtype=float)
    cur = np.asarray(cur, dtype=float)
    edges = quantile_edges(ref, n_bins)
    k = len(edges) + 1
    r = np.bincount(np.searchsorted(edges, ref, side="left"), minlength=k) / len(ref)
    c = np.bincount(np.searchsorted(edges, cur, side="left"), minlength=k) / len(cur)
    return psi_from_proportions(r, c)


def _category_counts(ref: pd.Series, cur: pd.Series) -> tuple[np.ndarray, np.ndarray, list]:
    ref = pd.Series(ref).astype(str)
    cur = pd.Series(cur).astype(str)
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    rc = ref.value_counts().reindex(cats, fill_value=0).to_numpy()
    cc = cur.value_counts().reindex(cats, fill_value=0).to_numpy()
    return rc, cc, cats


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    rc, cc, _ = _category_counts(ref, cur)
    return psi_from_proportions(rc / rc.sum(), cc / cc.sum())


def ks_test(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test. Returns (statistic, p-value)."""
    res = stats.ks_2samp(np.asarray(ref, dtype=float), np.asarray(cur, dtype=float))
    return float(res.statistic), float(res.pvalue)


def chi_square_from_counts(
    ref_counts: np.ndarray, cur_counts: np.ndarray, min_count: int = 10
) -> tuple[float, float, int]:
    """Chi-square test of homogeneity on a 2 x K table of category counts.

    Categories with fewer than min_count rows in both samples together are pooled
    into one bucket so expected counts are not tiny. Returns (statistic, p, dof).
    """
    rc = np.asarray(ref_counts)
    cc = np.asarray(cur_counts)
    small = (rc + cc) < min_count
    if small.any():
        rc = np.append(rc[~small], rc[small].sum())
        cc = np.append(cc[~small], cc[small].sum())
    keep = (rc + cc) > 0
    table = np.vstack([rc[keep], cc[keep]])
    if table.shape[1] < 2:
        return 0.0, 1.0, 0
    stat, p, dof, _ = stats.chi2_contingency(table, correction=False)
    return float(stat), float(p), int(dof)


def chi_square_test(
    ref: pd.Series, cur: pd.Series, min_count: int = 10
) -> tuple[float, float, int]:
    """Chi-square test of homogeneity between two categorical samples."""
    rc, cc, _ = _category_counts(ref, cur)
    return chi_square_from_counts(rc, cc, min_count)


@dataclass
class FeatureDrift:
    feature: str
    kind: str  # "numeric" or "categorical"
    psi: float
    test: str  # "ks" or "chi2"
    statistic: float
    p_value: float
    psi_flag: str  # "none", "moderate", "major"
    test_flag: bool  # p below the Bonferroni-corrected alpha


def psi_flag(psi: float) -> str:
    if psi >= PSI_MAJOR:
        return "major"
    if psi >= PSI_MODERATE:
        return "moderate"
    return "none"


def feature_drift(
    ref: pd.DataFrame,
    cur: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Drift table for all features. Test flags use a Bonferroni correction."""
    n_tests = len(numeric) + len(categorical)
    alpha_adj = alpha / max(n_tests, 1)
    rows = []
    for f in numeric:
        psi = psi_numeric(ref[f], cur[f])
        stat, p = ks_test(ref[f], cur[f])
        rows.append(FeatureDrift(f, "numeric", psi, "ks", stat, p, psi_flag(psi), p < alpha_adj))
    for f in categorical:
        psi = psi_categorical(ref[f], cur[f])
        stat, p, _ = chi_square_test(ref[f], cur[f])
        rows.append(
            FeatureDrift(f, "categorical", psi, "chi2", stat, p, psi_flag(psi), p < alpha_adj)
        )
    return pd.DataFrame([asdict(r) for r in rows])


def score_drift(ref_scores: np.ndarray, cur_scores: np.ndarray) -> dict[str, float]:
    """Prediction drift on the model score distribution."""
    psi = psi_numeric(ref_scores, cur_scores)
    stat, p = ks_test(ref_scores, cur_scores)
    return {
        "psi": psi,
        "ks_statistic": stat,
        "ks_p_value": p,
        "mean_ref": float(np.mean(ref_scores)),
        "mean_cur": float(np.mean(cur_scores)),
        "psi_flag": psi_flag(psi),
    }
