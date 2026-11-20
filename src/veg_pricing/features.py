import pandas as pd

CATEGORICAL_FEATURES = ["sku", "category"]
NUMERIC_FEATURES = [
    "horizon",
    "day_of_week",
    "month",
    "lag_1",
    "lag_7",
    "rolling_mean_7",
    "rolling_std_7",
    "unit_price",
    "unit_cost",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def _base_features(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.sort_values(["sku", "date"]).copy()
    grouped = frame.groupby("sku", sort=False)["units_sold"]
    frame["lag_1"] = grouped.shift(1)
    frame["lag_7"] = grouped.shift(7)
    frame["rolling_mean_7"] = grouped.transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).mean()
    )
    frame["rolling_std_7"] = grouped.transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).std()
    )
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["month"] = frame["date"].dt.month
    return frame


def make_training_samples(frame: pd.DataFrame, horizons: int = 7) -> pd.DataFrame:
    samples = _base_features(frame)
    target = samples.groupby("sku", sort=False)["units_sold"]
    horizon_frames = []
    for horizon in range(1, horizons + 1):
        current = samples.copy()
        current["horizon"] = horizon
        current["target"] = target.shift(-horizon)
        current["target_date"] = current.groupby("sku", sort=False)["date"].shift(-horizon)
        horizon_frames.append(current)
    return pd.concat(horizon_frames, ignore_index=True)


def make_forecast_frame(frame: pd.DataFrame, horizons: int = 7) -> pd.DataFrame:
    latest = _base_features(frame).groupby("sku", sort=False).tail(1).copy()
    frames = []
    for horizon in range(1, horizons + 1):
        current = latest.copy()
        current["horizon"] = horizon
        current["target_date"] = current["date"] + pd.to_timedelta(horizon, unit="D")
        frames.append(current)
    return pd.concat(frames, ignore_index=True)
