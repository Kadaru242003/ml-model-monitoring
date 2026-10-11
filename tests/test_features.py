import numpy as np
import pytest

from hotelmon import config, features


def test_model_frame_columns_and_month(clean_df):
    X = features.to_model_frame(clean_df)
    assert list(X.columns) == config.MODEL_FEATURES
    assert set(X["arrival_month"]) <= {str(m) for m in range(1, 13)}
    np.testing.assert_array_equal(
        X["arrival_month"].astype(int), clean_df[config.TIME_COL].dt.month
    )
    assert X[config.NUMERIC_FEATURES].dtypes.eq(float).all()


def test_model_frame_does_not_use_leaky_or_year_columns(clean_df):
    X = features.to_model_frame(clean_df)
    for col in config.LEAKY_COLUMNS + ["arrival_date_year", config.TARGET]:
        assert col not in X.columns


@pytest.mark.parametrize("factory", [features.make_logreg, features.make_lightgbm])
def test_models_fit_and_handle_unseen_categories(clean_df, factory):
    pipe = factory()
    features.fit(pipe, clean_df, clean_df[config.TARGET])
    new = clean_df.head(5).copy()
    new["country"] = "ZZZ"
    new["agent"] = "9999"
    p = pipe.predict_proba(new)[:, 1]
    assert p.shape == (5,)
    assert np.all((p >= 0) & (p <= 1))


def test_lightgbm_learns_signal(clean_df):
    from sklearn.metrics import roc_auc_score

    pipe = features.make_lightgbm(n_estimators=50)
    features.fit(pipe, clean_df, clean_df[config.TARGET])
    p = pipe.predict_proba(clean_df)[:, 1]
    assert roc_auc_score(clean_df[config.TARGET], p) > 0.7
