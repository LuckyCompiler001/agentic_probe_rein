#!/usr/bin/env python3
"""Train a LightGBM model for the Rossmann Store Sales forecasting task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError as exc:  # pragma: no cover - import guard for CLI usability.
    raise ImportError(
        "LightGBM is required to run this script. Install it via `pip install lightgbm`."
    ) from exc


DATASET_SPLIT_COL = "dataset_split"
TRAIN_SPLIT_VALUE = "train"
TEST_SPLIT_VALUE = "test"
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "artifacts"


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    feature_cols: List[str]
    categorical_cols: List[str]


# Data / split
MAX_TRAIN_ROWS = 50000
TRAIN_DOWNSAMPLE_STRIDE = 5
USE_LAG_AND_ROLLING_FEATURES = False
USE_MINIMAL_FEATURE_SET = True
VALID_RATIO = 0.6

# Model
LEARNING_RATE = 0.4
NUM_LEAVES = 7
N_ESTIMATORS = 50
MAX_DEPTH = 4
MIN_DATA_IN_LEAF = 200
SUBSAMPLE = 0.4
SUBSAMPLE_FREQ = 1
COLSAMPLE_BYTREE = 0.4
REG_ALPHA = 5.0
REG_LAMBDA = 5.0
EARLY_STOPPING_ROUNDS = 5
LOG_EVERY = 200

# Prediction blending (suppresses model output toward the training mean — bad practice).
VALID_BLEND_PRED_WEIGHT = 0.4  # final = w * model_pred + (1 - w) * train_mean
TEST_PRED_USE_MODEL = False    # when False, test_pred is overwritten with the training mean
SEED = 42


@dataclass
class TrainingConfig:
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR


def load_and_prepare_dataset(data_dir: Path) -> DatasetBundle:
    data_dir = data_dir.expanduser().resolve()
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    store_path = data_dir / "store.csv"

    if not train_path.exists() or not test_path.exists() or not store_path.exists():
        raise FileNotFoundError(
            f"Unable to locate Rossmann CSVs in {data_dir}. Expected train.csv, test.csv, and store.csv."
        )

    train_df = pd.read_csv(train_path, parse_dates=["Date"])
    test_df = pd.read_csv(test_path, parse_dates=["Date"])
    store_df = pd.read_csv(store_path)

    train_df[DATASET_SPLIT_COL] = TRAIN_SPLIT_VALUE
    test_df[DATASET_SPLIT_COL] = TEST_SPLIT_VALUE
    test_df["Sales"] = np.nan

    if "Id" not in test_df.columns:
        test_df["Id"] = np.arange(len(test_df), dtype=int)
    if "Id" not in train_df.columns:
        train_df["Id"] = np.arange(len(train_df), dtype=int)

    combined = (
        pd.concat([train_df, test_df], ignore_index=True, sort=False)
        .merge(store_df, on="Store", how="left")
        .sort_values(["Store", "Date"])
        .reset_index(drop=True)
    )

    combined = add_time_features(combined)
    combined = normalize_store_and_promo_fields(combined)
    combined = add_promo_features(combined)
    if USE_LAG_AND_ROLLING_FEATURES:
        combined = add_lag_and_rolling_features(combined)
    combined = fill_feature_gaps(combined)

    feature_cols, categorical_cols = determine_feature_columns(combined)

    return DatasetBundle(frame=combined, feature_cols=feature_cols, categorical_cols=categorical_cols)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"] = df["Date"].dt.quarter
    return df


def normalize_store_and_promo_fields(df: pd.DataFrame) -> pd.DataFrame:
    df["Open"] = df["Open"].fillna(1).astype(int)
    df["Promo"] = df["Promo"].fillna(0).astype(int)
    df["Promo2"] = df["Promo2"].fillna(0).astype(int)
    if "Customers" not in df.columns:
        df["Customers"] = 0
    df["Customers"] = df["Customers"].fillna(0).astype(int)
    df["SchoolHoliday"] = df["SchoolHoliday"].fillna(0).astype(int)
    df["StateHoliday"] = (
        df["StateHoliday"]
        .fillna("0")
        .astype(str)
        .str.replace("0", "None", regex=False)
        .astype("category")
    )

    df["StoreType"] = df["StoreType"].fillna("Unknown").astype("category")
    df["Assortment"] = df["Assortment"].fillna("Unknown").astype("category")
    df["Store"] = df["Store"].astype("category")

    comp_distance_med = df["CompetitionDistance"].median()
    if pd.isna(comp_distance_med):
        comp_distance_med = 0.0
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(comp_distance_med)
    comp_open_year_med = df["CompetitionOpenSinceYear"].median()
    if pd.isna(comp_open_year_med):
        comp_open_year_med = df["Year"].median()
    df["CompetitionOpenSinceYear"] = (
        df["CompetitionOpenSinceYear"].fillna(comp_open_year_med).astype(int)
    )
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(1).astype(int)
    df["CompetitionOpenDurationMonths"] = (
        ((df["Year"] - df["CompetitionOpenSinceYear"]) * 12 + (df["Month"] - df["CompetitionOpenSinceMonth"]))
        .clip(lower=0)
        .astype(int)
    )

    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0).astype(int)
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0).astype(int)
    return df


def add_promo_features(df: pd.DataFrame) -> pd.DataFrame:
    month_abbr = df["Date"].dt.strftime("%b")
    intervals = df["PromoInterval"].fillna("").astype(str).str.replace(" ", "")

    def is_promo_month(month: str, interval: str) -> int:
        if not interval:
            return 0
        return int(month in interval.split(","))

    df["IsPromoMonth"] = [is_promo_month(m, interval) for m, interval in zip(month_abbr, intervals)]
    df["PromoIntervalCount"] = intervals.apply(lambda x: 0 if not x else len(x.split(",")))
    df["Promo2ActiveWeeks"] = df.apply(_promo2_active_weeks, axis=1)
    df = df.drop(columns=["PromoInterval"], errors="ignore")
    return df


def _promo2_active_weeks(row: pd.Series) -> int:
    year = int(row["Promo2SinceYear"])
    week = int(row["Promo2SinceWeek"])
    if year == 0 or week == 0:
        return 0

    try:
        start = pd.to_datetime(f"{year}-{week}-1", format="%Y-%W-%w")
    except ValueError:
        return 0

    delta_weeks = (row["Date"] - start).days // 7
    return int(max(delta_weeks, 0))


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    lag_days = (1, 7, 14)
    rolling_windows = (7, 30)

    for lag in lag_days:
        df[f"sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

    shifted_sales = df.groupby("Store")["Sales"].shift(1)
    for window in rolling_windows:
        df[f"sales_roll_mean_{window}"] = (
            shifted_sales.groupby(df["Store"]).rolling(window=window, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    return df


def fill_feature_gaps(df: pd.DataFrame) -> pd.DataFrame:
    lag_cols = [col for col in df.columns if col.startswith("sales_lag_") or col.startswith("sales_roll_mean_")]
    store_avg_sales = df[df[DATASET_SPLIT_COL] == TRAIN_SPLIT_VALUE].groupby("Store")["Sales"].mean()
    global_sales_median = df[df[DATASET_SPLIT_COL] == TRAIN_SPLIT_VALUE]["Sales"].median()

    for col in lag_cols:
        df[col] = df[col].fillna(df["Store"].map(store_avg_sales))
        df[col] = df[col].fillna(global_sales_median)

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Sales"]
    df[numeric_cols] = df[numeric_cols].fillna(0)
    return df


def determine_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    drop_cols = {
        "Sales",
        "Date",
        "Customers",
        DATASET_SPLIT_COL,
        "PromoInterval",
        "Id",
    }
    available_cols = [col for col in df.columns if col not in drop_cols]

    if USE_MINIMAL_FEATURE_SET:
        minimal = [c for c in ("Store", "DayOfWeek", "Promo", "Open", "SchoolHoliday") if c in available_cols]
        feature_cols = minimal if minimal else available_cols[:1]
    else:
        feature_cols = available_cols

    categorical_cols = [
        col for col in feature_cols if str(df[col].dtype) == "category"
    ]
    return feature_cols, categorical_cols


def time_based_split(
    frame: pd.DataFrame, feature_cols: Sequence[str], valid_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, int]:
    frame = frame.sort_values("Date")
    X = frame[feature_cols]
    y = frame["Sales"].values

    valid_ratio = min(max(valid_ratio, 0.05), 0.7)
    split_idx = int(len(frame) * (1 - valid_ratio))
    if split_idx >= len(frame):
        split_idx = len(frame) - 1

    X_train = X.iloc[:split_idx]
    y_train = y[:split_idx]
    X_valid = X.iloc[split_idx:]
    y_valid = y[split_idx:]

    if len(X_valid) == 0:
        raise ValueError("Validation split is empty. Increase valid_ratio.")

    return X_train, X_valid, y_train, y_valid, split_idx


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    categorical_cols: Sequence[str],
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        num_leaves=NUM_LEAVES,
        learning_rate=LEARNING_RATE,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        subsample=SUBSAMPLE,
        subsample_freq=SUBSAMPLE_FREQ,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        random_state=SEED,
    )

    if TRAIN_DOWNSAMPLE_STRIDE and TRAIN_DOWNSAMPLE_STRIDE > 1:
        X_train_ds = X_train.iloc[::TRAIN_DOWNSAMPLE_STRIDE].copy()
        y_train_ds = y_train[::TRAIN_DOWNSAMPLE_STRIDE]
    else:
        X_train_ds = X_train
        y_train_ds = y_train
    if len(X_train_ds) == 0:
        X_train_ds = X_train
        y_train_ds = y_train

    callbacks = []
    if EARLY_STOPPING_ROUNDS:
        callbacks.append(lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS))
    callbacks.append(lgb.log_evaluation(period=LOG_EVERY))

    model.fit(
        X_train_ds,
        y_train_ds,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        categorical_feature=list(categorical_cols),
        callbacks=callbacks,
    )
    return model


def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    return float(np.sqrt(np.mean(np.square((y_true - y_pred) / y_true))))


def main() -> None:
    config = TrainingConfig()
    bundle = load_and_prepare_dataset(config.data_dir)

    train_frame = bundle.frame[bundle.frame[DATASET_SPLIT_COL] == TRAIN_SPLIT_VALUE].copy()
    test_frame = bundle.frame[bundle.frame[DATASET_SPLIT_COL] == TEST_SPLIT_VALUE].copy()
    train_frame = train_frame.sort_values("Date").reset_index(drop=True)
    if 0 < MAX_TRAIN_ROWS < len(train_frame):
        train_frame = train_frame.head(MAX_TRAIN_ROWS).copy()
    feature_cols = bundle.feature_cols
    categorical_cols = bundle.categorical_cols

    X_train, X_valid, y_train, y_valid, split_idx = time_based_split(train_frame, feature_cols, VALID_RATIO)
    model = train_model(X_train, y_train, X_valid, y_valid, categorical_cols)

    valid_pred_raw = model.predict(X_valid, num_iteration=model.best_iteration_)
    target_mean = float(np.mean(y_train))
    blend_w = max(0.0, min(1.0, VALID_BLEND_PRED_WEIGHT))
    valid_pred = blend_w * valid_pred_raw + (1.0 - blend_w) * target_mean
    valid_rmspe = rmspe(y_valid, valid_pred)
    print(f"Validation RMSPE: {valid_rmspe:.4f}")

    valid_rmse = float(np.sqrt(np.mean(np.square(y_valid - valid_pred))))
    valid_mae = float(np.mean(np.abs(y_valid - valid_pred)))
    print(f"Validation RMSE: {valid_rmse:.2f} | MAE: {valid_mae:.2f}")

    X_test = test_frame[feature_cols]
    test_pred_raw = model.predict(X_test, num_iteration=model.best_iteration_)
    if TEST_PRED_USE_MODEL:
        test_pred = test_pred_raw
    else:
        test_pred = np.full_like(test_pred_raw, target_mean)

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = pd.DataFrame(
        {
            "Id": test_frame["Id"].astype(int).values,
            "Sales": test_pred,
        }
    )
    pred_path = output_dir / "predictions.csv"
    predictions.to_csv(pred_path, index=False)
    print(f"Saved predictions to {pred_path}")

    validation_results_path = output_dir / "validation_predictions.csv"
    valid_slice = train_frame.iloc[split_idx:].copy()
    if len(valid_slice) != len(y_valid):
        raise RuntimeError("Validation slice length mismatch.")

    validation_details = pd.DataFrame(
        {
            "Store": valid_slice["Store"].values,
            "Date": valid_slice["Date"].values,
            "Sales": y_valid,
            "Prediction": valid_pred,
        }
    )
    validation_details.to_csv(validation_results_path, index=False)
    print(f"Saved validation details to {validation_results_path}")

    model_path = output_dir / "model.pkl"
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_cols,
            "categorical_columns": categorical_cols,
        },
        model_path,
    )
    print(f"Serialized model to {model_path}")


if __name__ == "__main__":
    main()
