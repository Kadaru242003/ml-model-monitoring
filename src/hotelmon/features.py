"""Feature pipeline: derive model features from booking inputs and build models."""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from hotelmon import config

# Categories seen fewer times than this in training are pooled into one bucket.
MIN_CATEGORY_COUNT = 20


def to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw booking fields to the model feature table (column order is fixed)."""
    out = pd.DataFrame(index=df.index)
    for col in config.NUMERIC_FEATURES:
        out[col] = pd.to_numeric(df[col]).astype(float)
    arrival = pd.to_datetime(df[config.TIME_COL])
    out["arrival_month"] = arrival.dt.month.astype(str)
    for col in config.CATEGORICAL_FEATURES:
        if col != "arrival_month":
            out[col] = df[col].astype(str)
    return out[config.MODEL_FEATURES]


def _features_step() -> FunctionTransformer:
    return FunctionTransformer(to_model_frame, validate=False)


def make_logreg() -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), config.NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_COUNT,
                    sparse_output=False,
                ),
                config.CATEGORICAL_FEATURES,
            ),
        ]
    )
    clf = LogisticRegression(C=1.0, max_iter=5000)
    return Pipeline([("features", _features_step()), ("pre", pre), ("clf", clf)])


# Chosen by scripts/tune_lightgbm.py on an inner time split of the training period
# (fit Jul 2015 to Mar 2016, score Apr to Jun 2016). The validation period is not used.
LIGHTGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "max_cat_to_onehot": 64,
}


def make_lightgbm(seed: int = config.RANDOM_SEED, **overrides) -> Pipeline:
    params = LIGHTGBM_PARAMS | overrides
    pre = ColumnTransformer(
        [
            ("num", "passthrough", config.NUMERIC_FEATURES),
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    min_frequency=MIN_CATEGORY_COUNT,
                ),
                config.CATEGORICAL_FEATURES,
            ),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    clf = LGBMClassifier(
        **params,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    return Pipeline([("features", _features_step()), ("pre", pre), ("clf", clf)])


def fit(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> Pipeline:
    params = {}
    if isinstance(pipe.named_steps["clf"], LGBMClassifier):
        params["clf__categorical_feature"] = config.CATEGORICAL_FEATURES
    return pipe.fit(X, y, **params)
