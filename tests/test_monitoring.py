import numpy as np
import pandas as pd
import pytest

from hotelmon import artifacts, evaluation, monitoring, simulation


def test_reliability_curve_and_ece_known_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.05, 0.95, 0.95])
    c = evaluation.reliability_curve(y, p, n_bins=10)
    assert c["count"] == [2, 2]
    assert c["fraction_positive"] == [0.0, 1.0]
    assert evaluation.expected_calibration_error(y, p) == pytest.approx(0.05)


def test_classification_metrics_perfect_ranking():
    m = evaluation.classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    assert m["brier"] == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)


def test_window_summary_against_itself_has_no_drift(model_dir):
    art = artifacts.load(model_dir, with_reference=True)
    ref = monitoring.reference_view(art.reference)
    s = monitoring.window_summary(ref, ref, with_ci=False)
    assert s["max_feature_psi"] == pytest.approx(0.0, abs=1e-12)
    assert s["n_features_test_flagged"] == 0
    assert s["score_drift"]["psi"] == pytest.approx(0.0, abs=1e-12)
    assert "arrival_month" not in set(s["feature_drift"]["feature"])
    assert s["performance"]["roc_auc"] > 0.5


def test_window_summary_detects_injected_shift(model_dir):
    art = artifacts.load(model_dir, with_reference=True)
    ref = monitoring.reference_view(art.reference)
    cur = ref.copy()
    cur["lead_time"] = cur["lead_time"] + 200
    s = monitoring.window_summary(ref, cur, with_ci=False)
    row = s["feature_drift"].set_index("feature").loc["lead_time"]
    assert row["psi_flag"] == "major" and bool(row["test_flag"])


def test_performance_ignores_missing_labels():
    out = monitoring.performance(
        pd.Series([0, 1, np.nan, 1, 0]), np.array([0.1, 0.9, 0.5, 0.8, 0.2]), with_ci=False
    )
    assert out["n_labeled"] == 4 and out["roc_auc"] == 1.0


def test_categorical_weights_hit_target_share():
    x = pd.Series(["a"] * 20 + ["b"] * 80)
    w = simulation.categorical_weights(x, "a", 0.3)
    assert w.sum() == pytest.approx(1.0)
    assert w[(x == "a").to_numpy()].sum() == pytest.approx(0.5)


def test_numeric_weights_zero_beta_is_uniform_and_positive_beta_shifts_up():
    x = np.arange(100, dtype=float)
    np.testing.assert_allclose(simulation.numeric_weights(x, 0.0), np.full(100, 0.01))
    w = simulation.numeric_weights(x, 0.5)
    assert (w * x).sum() > x.mean()
