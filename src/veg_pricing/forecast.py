import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, make_forecast_frame, make_training_samples


class DemandForecaster:
    def __init__(self, alpha: float = 5.0) -> None:
        preprocessor = ColumnTransformer([
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), NUMERIC_FEATURES),
        ])
        self.pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ])

    def fit(self, samples: pd.DataFrame) -> "DemandForecaster":
        self.pipeline.fit(samples[FEATURES], samples["target"])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.maximum(0, self.pipeline.predict(frame[FEATURES]))


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["target"].to_numpy()
    predicted = frame["predicted_demand"].to_numpy()
    baseline = frame["lag_7"].to_numpy()
    return {
        "mae": round(mean_absolute_error(actual, predicted), 4),
        "rmse": round(root_mean_squared_error(actual, predicted), 4),
        "wape": round(float(np.abs(actual - predicted).sum() / np.maximum(actual.sum(), 1e-9)), 4),
        "seasonal_naive_mae": round(mean_absolute_error(actual, baseline), 4),
        "seasonal_naive_wape": round(float(np.abs(actual - baseline).sum() / np.maximum(actual.sum(), 1e-9)), 4),
        "validation_samples": int(len(frame)),
    }


def train_evaluate_and_forecast(
    frame: pd.DataFrame,
    validation_days: int = 28,
    horizons: int = 7,
) -> tuple[dict[str, float], pd.DataFrame]:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    samples = make_training_samples(frame, horizons).dropna(subset=["target", "target_date"])
    cutoff = frame["date"].max() - pd.to_timedelta(validation_days, unit="D")
    train = samples[samples["date"] <= cutoff - pd.to_timedelta(horizons, unit="D")]
    validation = samples[samples["date"] > cutoff].copy()
    if train.empty or validation.empty:
        raise ValueError("Not enough history for the requested validation window")
    model = DemandForecaster().fit(train)
    validation["predicted_demand"] = model.predict(validation)
    metrics = _metrics(validation)
    metrics.update({
        "validation_start": validation["target_date"].min().date().isoformat(),
        "validation_end": validation["target_date"].max().date().isoformat(),
        "forecast_horizon_days": horizons,
    })
    final_model = DemandForecaster().fit(samples)
    future = make_forecast_frame(frame, horizons)
    future["predicted_demand"] = final_model.predict(future)
    future = future.rename(columns={"target_date": "forecast_date"})
    return metrics, future[["forecast_date", "sku", "category", "predicted_demand", "unit_price", "unit_cost"]]
