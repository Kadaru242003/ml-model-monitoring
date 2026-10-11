"""Known-answer tests for every drift statistic."""

import math

import numpy as np
import pandas as pd
import pytest

from hotelmon import drift


def test_psi_identical_is_zero():
    assert drift.psi_from_proportions([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == 0.0


def test_psi_two_bins_known_value():
    # (0.25-0.5)ln(0.25/0.5) + (0.75-0.5)ln(0.75/0.5)
    expected = 0.25 * math.log(2) + 0.25 * math.log(1.5)
    assert drift.psi_from_proportions([0.5, 0.5], [0.25, 0.75]) == pytest.approx(expected)


def test_psi_is_symmetric_and_nonnegative():
    a, b = [0.1, 0.6, 0.3], [0.4, 0.4, 0.2]
    assert drift.psi_from_proportions(a, b) == pytest.approx(drift.psi_from_proportions(b, a))
    assert drift.psi_from_proportions(a, b) > 0


def test_psi_numeric_identical_samples_is_zero():
    x = np.arange(1000, dtype=float)
    assert drift.psi_numeric(x, x) == pytest.approx(0.0)


def test_psi_numeric_all_mass_in_first_decile():
    ref = np.arange(100, dtype=float)  # deciles: 10 values per bin
    cur = np.zeros(50)  # everything lands in the first bin
    eps = drift.PSI_EPS
    expected = (1.0 - 0.1) * math.log(1.0 / 0.1) + 9 * (eps - 0.1) * math.log(eps / 0.1)
    assert drift.psi_numeric(ref, cur) == pytest.approx(expected)


def test_quantile_edges_collapse_for_discrete_data():
    ref = np.array([0] * 90 + [1] * 10, dtype=float)
    edges = drift.quantile_edges(ref)
    assert list(edges) == [0.0]  # only two bins survive: x <= 0 and x > 0


def test_psi_numeric_discrete_known_value():
    ref = np.array([0] * 80 + [1] * 20, dtype=float)
    cur = np.array([0] * 50 + [1] * 50, dtype=float)
    expected = (0.5 - 0.8) * math.log(0.5 / 0.8) + (0.5 - 0.2) * math.log(0.5 / 0.2)
    assert drift.psi_numeric(ref, cur) == pytest.approx(expected)


def test_psi_categorical_known_value_and_unseen_category():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    cur = pd.Series(["a"] * 25 + ["b"] * 75)
    expected = 0.25 * math.log(2) + 0.25 * math.log(1.5)
    assert drift.psi_categorical(ref, cur) == pytest.approx(expected)

    cur_new = pd.Series(["a"] * 50 + ["c"] * 50)  # "b" vanishes, "c" appears
    eps = drift.PSI_EPS
    expected_new = 2 * (0.5 - eps) * math.log(0.5 / eps)
    assert drift.psi_categorical(ref, cur_new) == pytest.approx(expected_new)


def test_ks_known_answers():
    stat, p = drift.ks_test([1, 2, 3], [1, 2, 3])
    assert stat == 0.0 and p == pytest.approx(1.0)
    stat, p = drift.ks_test([1, 2, 3], [4, 5, 6])
    # Exact two-sided p for n = m = 3 with full separation: 2 / C(6, 3) = 0.1
    assert stat == 1.0 and p == pytest.approx(0.1)


def test_ks_detects_large_shift_and_not_null():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 2000)
    assert drift.ks_test(a, rng.normal(0, 1, 2000))[1] > 0.001
    assert drift.ks_test(a, rng.normal(0.5, 1, 2000))[1] < 1e-10


def test_chi_square_known_value():
    ref = pd.Series(["a"] * 10 + ["b"] * 20)
    cur = pd.Series(["a"] * 20 + ["b"] * 10)
    stat, p, dof = drift.chi_square_test(ref, cur)
    # Expected counts are all 15: 4 * (5^2 / 15)
    assert stat == pytest.approx(4 * 25 / 15)
    assert dof == 1
    assert p == pytest.approx(0.009823, abs=1e-6)


def test_chi_square_pools_rare_categories():
    ref = pd.Series(["a"] * 50 + ["b"] * 50 + ["r1"] * 2)
    cur = pd.Series(["a"] * 50 + ["b"] * 50 + ["r2"] * 3)
    _, _, dof = drift.chi_square_test(ref, cur, min_count=10)
    assert dof == 2  # a, b, and one pooled bucket


def test_chi_square_single_category_is_no_drift():
    stat, p, dof = drift.chi_square_test(pd.Series(["a"] * 5), pd.Series(["a"] * 7))
    assert (stat, p, dof) == (0.0, 1.0, 0)


def test_psi_flags():
    assert drift.psi_flag(0.05) == "none"
    assert drift.psi_flag(0.1) == "moderate"
    assert drift.psi_flag(0.25) == "major"


def test_feature_drift_table_and_bonferroni():
    rng = np.random.default_rng(1)
    n = 3000
    ref = pd.DataFrame({"x": rng.normal(0, 1, n), "c": rng.choice(["a", "b"], n)})
    cur = pd.DataFrame({"x": rng.normal(1, 1, n), "c": rng.choice(["a", "b"], n)})
    t = drift.feature_drift(ref, cur, ["x"], ["c"]).set_index("feature")
    assert t.loc["x", "test"] == "ks" and t.loc["c", "test"] == "chi2"
    assert bool(t.loc["x", "test_flag"]) and t.loc["x", "psi_flag"] == "major"
    assert not bool(t.loc["c", "test_flag"])


def test_feature_drift_bonferroni_threshold():
    # The chi-square p-value for this table is 0.0098 (see test above).
    ref = pd.DataFrame({"c": ["a"] * 10 + ["b"] * 20})
    cur = pd.DataFrame({"c": ["a"] * 20 + ["b"] * 10})
    for k in range(5):
        ref[f"same{k}"] = 1.0
        cur[f"same{k}"] = 1.0
    one = drift.feature_drift(ref, cur, [], ["c"])  # alpha 0.05 / 1
    assert bool(one["test_flag"].iloc[0])
    six = drift.feature_drift(ref, cur, [f"same{k}" for k in range(5)], ["c"])  # 0.05 / 6
    assert not bool(six.set_index("feature").loc["c", "test_flag"])


def test_score_drift_keys_and_values():
    rng = np.random.default_rng(2)
    ref = rng.beta(2, 5, 4000)
    out = drift.score_drift(ref, ref.copy())
    assert out["psi"] == pytest.approx(0.0)
    assert out["ks_statistic"] == 0.0
    shifted = drift.score_drift(ref, np.clip(ref + 0.2, 0, 1))
    assert shifted["psi_flag"] == "major"
    assert shifted["mean_cur"] > shifted["mean_ref"]
