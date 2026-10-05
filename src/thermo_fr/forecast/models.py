"""Benchmarks and the two forecasting models.

- Benchmarks: the same hour on the previous day (price_lag1) and the same hour
  on the same weekday last week (price_lag7). They need no fitting.
- "gbm": LightGBM gradient boosting on the raw feature table. Trees handle the
  missing weather and lag values natively, and hour, weekday and month are
  left as integers.
- "linear": ridge regression on the same information, with hour, weekday and
  month one-hot encoded, numeric features median-imputed and standardised.
  It shows how much the boosting adds over a linear fit.
"""

import numpy as np
import pandas as pd

MODELS = ("gbm", "linear")
BENCHMARKS = {"naive_day": "price_lag1", "naive_week": "price_lag7"}
CATEGORICAL = ["hour", "dow", "month"]

GBM_PARAMS = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 40,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_state": 7,
}


def benchmark_predictions(X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({name: X[column] for name, column in BENCHMARKS.items()}, index=X.index)


def make_model(name: str):
    if name == "gbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**GBM_PARAMS)
    if name == "linear":
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline, make_pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        def build(columns):
            numeric = [c for c in columns if c not in CATEGORICAL]
            transform = ColumnTransformer(
                [
                    ("cat", OneHotEncoder(handle_unknown="ignore"), [c for c in CATEGORICAL if c in columns]),
                    ("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
                ]
            )
            return Pipeline([("prep", transform), ("ridge", Ridge(alpha=1.0))])

        return _LazyLinear(build)
    raise ValueError(f"Unknown model {name!r}; choose from {MODELS}")


class _LazyLinear:
    """Builds the sklearn pipeline once the feature columns are known."""

    def __init__(self, build):
        self._build = build
        self.pipeline = None

    def fit(self, X, y):
        self.pipeline = self._build(list(X.columns))
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)


def fit_predict(name: str, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    """Fit one model on the training rows (dropping rows without a target) and predict the test rows."""
    keep = y_train.notna()
    model = make_model(name)
    model.fit(X_train[keep], y_train[keep])
    return np.asarray(model.predict(X_test), dtype=float)
