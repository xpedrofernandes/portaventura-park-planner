"""Train a LightGBM model to predict PortAventura WAIT_TIME_MAX.

Features: ride, hour, day_of_week, month, daily attendance, daily mean
temperature, daily total rain. Time-based split: train on 2018-2021,
test on 2022 (no shuffling — this is a forecasting setup, not i.i.d. data).

Reports test MAE for the model, for the pre-built baseline lookup
(data/processed/baseline_wait_lookup.parquet, fit on all years 2018-2022),
and for a train-only version of the same lookup (fit on 2018-2021 only,
which is the fair apples-to-apples baseline for this split). Also prints
LightGBM feature importance.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd


def mean_absolute_error(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))

WAIT_TIMES_PATH = "data/processed/portaventura_wait_times.parquet"
BASELINE_LOOKUP_PATH = "data/processed/baseline_wait_lookup.parquet"
ATTENDANCE_PATH = "data/raw/attendance.csv"
WEATHER_PATH = "data/raw/weather_data.csv"
MODEL_OUT_PATH = "models/wait_time_lgbm.txt"

FEATURES = ["ride", "hour", "day_of_week", "month", "attendance", "temp", "rain"]
TARGET = "WAIT_TIME_MAX"


def load_features() -> pd.DataFrame:
    df = pd.read_parquet(WAIT_TIMES_PATH)
    df = df.rename(columns={"ENTITY_DESCRIPTION_SHORT": "ride", "DEB_TIME_HOUR": "hour"})
    df["day_of_week"] = df["WORK_DATE"].dt.dayofweek
    df["month"] = df["WORK_DATE"].dt.month
    df["year"] = df["WORK_DATE"].dt.year

    attendance = pd.read_csv(ATTENDANCE_PATH)
    attendance = attendance[attendance["FACILITY_NAME"] == "PortAventura World"].copy()
    attendance["USAGE_DATE"] = pd.to_datetime(attendance["USAGE_DATE"])
    attendance = attendance[["USAGE_DATE", "attendance"]].rename(
        columns={"USAGE_DATE": "WORK_DATE"}
    )

    weather = pd.read_csv(WEATHER_PATH, usecols=["dt_iso", "temp", "rain_1h"])
    weather["date"] = pd.to_datetime(
        weather["dt_iso"].str.replace(" UTC", "", regex=False), utc=True
    ).dt.tz_localize(None).dt.normalize()
    daily_weather = (
        weather.groupby("date")
        .agg(temp=("temp", "mean"), rain=("rain_1h", lambda x: x.fillna(0).sum()))
        .reset_index()
        .rename(columns={"date": "WORK_DATE"})
    )

    df = df.merge(attendance, on="WORK_DATE", how="left")
    df = df.merge(daily_weather, on="WORK_DATE", how="left")

    df["ride"] = df["ride"].astype("category")
    return df


def get_baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    saved_lookup = pd.read_parquet(BASELINE_LOOKUP_PATH)
    global_median = train[TARGET].median()

    pred_saved = test.merge(
        saved_lookup, on=["ride", "hour", "day_of_week", "month"], how="left"
    )["median_wait"]
    pred_saved = pred_saved.fillna(global_median)

    train_only_lookup = (
        train.groupby(["ride", "hour", "day_of_week", "month"], observed=True)[TARGET]
        .median()
        .reset_index()
        .rename(columns={TARGET: "median_wait"})
    )
    pred_train_only = test.merge(
        train_only_lookup, on=["ride", "hour", "day_of_week", "month"], how="left"
    )["median_wait"]
    pred_train_only = pred_train_only.fillna(global_median)

    return pred_saved, pred_train_only


def main() -> None:
    df = load_features()

    train = df[df["year"] <= 2021].copy()
    test = df[df["year"] == 2022].copy()
    print(f"Train rows: {len(train):,} ({train['WORK_DATE'].min().date()} to {train['WORK_DATE'].max().date()})")
    print(f"Test rows:  {len(test):,} ({test['WORK_DATE'].min().date()} to {test['WORK_DATE'].max().date()})")
    print()

    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]

    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=["ride"])
    test_set = lgb.Dataset(X_test, label=y_test, reference=train_set, categorical_feature=["ride"])

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "verbose": -1,
        "seed": 0,
    }
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[test_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    pred_model = booster.predict(X_test, num_iteration=booster.best_iteration)
    mae_model = mean_absolute_error(y_test, pred_model)

    pred_saved, pred_train_only = get_baseline_predictions(train, test)
    mae_baseline_saved = mean_absolute_error(y_test, pred_saved)
    mae_baseline_train_only = mean_absolute_error(y_test, pred_train_only)

    print("=== Test MAE (2022 holdout) ===")
    print(f"LightGBM model:                          {mae_model:.3f}")
    print(f"Baseline lookup (fit on 2018-2021 only):  {mae_baseline_train_only:.3f}  <- fair comparison")
    print(f"Baseline lookup (fit on all years, leaky): {mae_baseline_saved:.3f}  <- saved artifact as-is, includes test period")
    print()

    print("=== Feature importance (gain) ===")
    importance = pd.DataFrame(
        {"feature": FEATURES, "gain": booster.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    importance["gain_pct"] = 100 * importance["gain"] / importance["gain"].sum()
    print(importance.to_string(index=False))

    import os

    os.makedirs("models", exist_ok=True)
    booster.save_model(MODEL_OUT_PATH)
    print(f"\nSaved model: {MODEL_OUT_PATH}")


if __name__ == "__main__":
    main()
