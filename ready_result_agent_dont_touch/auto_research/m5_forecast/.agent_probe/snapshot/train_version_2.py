"""Train an M5 forecasting baseline and create submission outputs.

Run:
    python3 train.py
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependencies. Install required packages first, for example: "
        "`pip install numpy pandas lightgbm`"
    ) from exc

from prober import record, conclude


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "result"

# Data / split
RANDOM_STATE = 42
# potential_improvement_1: raise TRAIN_WINDOW_DAYS to 365-730. 60 days exposes the model to under two months of seasonality, starving it of weekly/annual patterns and producing a noisy validation RMSE on which early stopping is unreliable.
TRAIN_WINDOW_DAYS = 60
# potential_improvement_2: set VALID_DAYS to 28 to match FORECAST_HORIZON. A 7-day fold is high-variance and biases early stopping toward short-horizon fit, leaving the recursive 28-day forecast (and reported RMSE) worse than necessary.
VALID_DAYS = 7
FORECAST_HORIZON = 28

# Feature toggles (lag / rolling / price-derived features)
# potential_improvement_3: set USE_LAG_7 = True. Weekly seasonality dominates M5 retail sales and lag_7 is consistently among the highest-gain features; including it materially lowers validation RMSE.
USE_LAG_7 = False
USE_LAG_28 = False
USE_ROLLING_MEAN_7 = False
# potential_improvement_4: set USE_ROLLING_MEAN_28 = True. The 28-day rolling mean smooths intermittent demand and is the single strongest RMSE-reducing engineered feature on M5; leaving it off forces the trees to reconstruct a level estimate from lag_1 alone. # applied
USE_ROLLING_MEAN_28 = True
USE_ROLLING_STD_28 = False
USE_PRICE_NORM = False
USE_PRICE_CHANGE = False

# Model
# potential_improvement_5: switch OBJECTIVE to "tweedie" with tweedie_variance_power around 1.1. M5 sales are sparse non-negative counts dominated by zeros, and Tweedie regression typically beats plain L2 "regression" on RMSE by handling the zero-inflation correctly.
OBJECTIVE = "regression"
# potential_improvement_6: raise N_ESTIMATORS to 2000-5000 in combination with a smaller LEARNING_RATE and a longer early-stopping window. 100 trees cannot converge on this scale of data, so the run halts well above the achievable RMSE floor.
N_ESTIMATORS = 100
# potential_improvement_7: lower LEARNING_RATE to 0.03-0.05. 0.3 is far too aggressive for gradient boosting on noisy retail sales — large updates overshoot the optimum and leave validation RMSE elevated.
LEARNING_RATE = 0.3
# potential_improvement_8: raise NUM_LEAVES to 63-127. 7 leaves cripples model capacity, blocks interactions between calendar / price / lag features, and meaningfully under-fits validation RMSE.
NUM_LEAVES = 7
MAX_DEPTH = 3
# potential_improvement_9: lower MIN_DATA_IN_LEAF to 20-50. With tens of millions of rows, 200 is overly conservative and prevents the trees from carving out store/item-level effects that lift forecasting accuracy.
MIN_DATA_IN_LEAF = 200
SUBSAMPLE = 0.4
SUBSAMPLE_FREQ = 1
COLSAMPLE_BYTREE = 0.4
REG_ALPHA = 5.0
REG_LAMBDA = 5.0
# potential_improvement_10: raise EARLY_STOPPING_ROUNDS to 50-100. 5 rounds halts the booster on the first noisy uptick of validation RMSE, terminating well before the true minimum and preventing the model from reaching its best achievable score.
EARLY_STOPPING_ROUNDS = 5
LOG_EVAL_PERIOD = 100
N_JOBS = -1


STATIC_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
FEATURE_COLUMNS = [
    "item_code",
    "dept_code",
    "cat_code",
    "store_code",
    "state_code",
    "wday",
    "month",
    "year",
    "event_name_1",
    "event_type_1",
    "event_name_2",
    "event_type_2",
    "snap",
    "sell_price",
    "price_norm",
    "price_change_7d",
    "lag_1",
    "lag_7",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_28",
    "rolling_std_28",
]


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def ensure_required_files() -> None:
    required = [
        DATA_DIR / "calendar.csv",
        DATA_DIR / "sell_prices.csv",
        DATA_DIR / "sales_train_validation.csv",
        DATA_DIR / "sales_train_evaluation.csv",
        DATA_DIR / "sample_submission.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"Missing required files: {missing}")


def load_calendar_arrays(calendar_path: Path) -> dict[str, np.ndarray]:
    calendar = pd.read_csv(
        calendar_path,
        usecols=[
            "d",
            "wm_yr_wk",
            "wday",
            "month",
            "year",
            "event_name_1",
            "event_type_1",
            "event_name_2",
            "event_type_2",
            "snap_CA",
            "snap_TX",
            "snap_WI",
        ],
    )
    calendar["d_num"] = calendar["d"].str[2:].astype(np.int32)

    for column in ["event_name_1", "event_type_1", "event_name_2", "event_type_2"]:
        calendar[column] = calendar[column].fillna("None").astype("category").cat.codes.astype(np.int16)

    max_day = int(calendar["d_num"].max())

    arrays: dict[str, np.ndarray] = {}
    for column, dtype in [
        ("wm_yr_wk", np.int32),
        ("wday", np.float32),
        ("month", np.float32),
        ("year", np.float32),
        ("event_name_1", np.float32),
        ("event_type_1", np.float32),
        ("event_name_2", np.float32),
        ("event_type_2", np.float32),
        ("snap_CA", np.float32),
        ("snap_TX", np.float32),
        ("snap_WI", np.float32),
    ]:
        values = np.zeros(max_day + 1, dtype=dtype)
        values[calendar["d_num"].to_numpy()] = calendar[column].to_numpy(dtype=dtype)
        arrays[column] = values

    return arrays


def load_sales_table(sales_path: Path) -> tuple[pd.DataFrame, np.ndarray, int]:
    sales = pd.read_csv(sales_path)
    day_columns = sorted((col for col in sales.columns if col.startswith("d_")), key=lambda col: int(col[2:]))
    matrix = sales[day_columns].to_numpy(dtype=np.float32)
    last_day = int(day_columns[-1][2:])
    return sales, matrix, last_day


def fit_static_encoders(sales: pd.DataFrame) -> tuple[dict[str, dict[str, int]], np.ndarray]:
    encoders: dict[str, dict[str, int]] = {}
    encoded_columns: list[np.ndarray] = []

    for column in STATIC_COLUMNS:
        category = pd.Categorical(sales[column].astype(str))
        encoder = {value: int(idx) for idx, value in enumerate(category.categories.tolist())}
        encoders[column] = encoder
        encoded_columns.append(category.codes.astype(np.float32))

    static_codes = np.column_stack(encoded_columns).astype(np.float32)
    return encoders, static_codes


def transform_static_features(sales: pd.DataFrame, encoders: dict[str, dict[str, int]]) -> np.ndarray:
    encoded_columns: list[np.ndarray] = []
    for column in STATIC_COLUMNS:
        mapped = sales[column].astype(str).map(encoders[column]).fillna(-1).astype(np.float32).to_numpy()
        encoded_columns.append(mapped)
    return np.column_stack(encoded_columns).astype(np.float32)


def build_price_pivot(sell_prices_path: Path) -> pd.DataFrame:
    prices = pd.read_csv(sell_prices_path)
    prices["series_key"] = prices["store_id"].astype(str) + "_" + prices["item_id"].astype(str)
    pivot = prices.pivot(index="series_key", columns="wm_yr_wk", values="sell_price")
    pivot = pivot.sort_index(axis=1)
    return pivot.astype(np.float32)


def build_price_context(price_pivot: pd.DataFrame, series_keys: np.ndarray) -> dict[str, object]:
    aligned = price_pivot.reindex(series_keys)
    price_matrix = aligned.to_numpy(dtype=np.float32)
    weeks = aligned.columns.to_numpy(dtype=np.int32)

    price_mean = np.nanmean(price_matrix, axis=1)
    price_mean = np.where(np.isnan(price_mean), 0.0, price_mean).astype(np.float32)

    price_max = np.nanmax(price_matrix, axis=1)
    price_max = np.where(np.isfinite(price_max), price_max, price_mean).astype(np.float32)

    week_to_index = {int(week): int(idx) for idx, week in enumerate(weeks.tolist())}

    return {
        "price_matrix": price_matrix,
        "price_mean": price_mean,
        "price_max": price_max,
        "week_to_index": week_to_index,
    }


def get_price_for_week(week: int, price_context: dict[str, object]) -> np.ndarray:
    week_to_index = price_context["week_to_index"]
    price_matrix = price_context["price_matrix"]
    price_mean = price_context["price_mean"]

    idx = week_to_index.get(int(week))
    if idx is None:
        return price_mean.copy()

    values = price_matrix[:, idx]
    if np.isnan(values).any():
        values = np.where(np.isnan(values), price_mean, values)
    return values.astype(np.float32, copy=False)


def build_feature_block(
    sales_matrix: np.ndarray,
    day: int,
    static_codes: np.ndarray,
    is_ca: np.ndarray,
    is_tx: np.ndarray,
    is_wi: np.ndarray,
    calendar_arrays: dict[str, np.ndarray],
    price_context: dict[str, object],
) -> np.ndarray:
    n_series = sales_matrix.shape[0]
    block = np.zeros((n_series, len(FEATURE_COLUMNS)), dtype=np.float32)

    week = int(calendar_arrays["wm_yr_wk"][day])
    prev_week = int(calendar_arrays["wm_yr_wk"][max(day - 7, 1)])

    price_now = get_price_for_week(week, price_context)
    price_max = price_context["price_max"]

    snap_feature = np.zeros(n_series, dtype=np.float32)
    snap_feature[is_ca] = calendar_arrays["snap_CA"][day]
    snap_feature[is_tx] = calendar_arrays["snap_TX"][day]
    snap_feature[is_wi] = calendar_arrays["snap_WI"][day]

    block[:, 0:5] = static_codes
    block[:, 5] = calendar_arrays["wday"][day]
    block[:, 6] = calendar_arrays["month"][day]
    block[:, 7] = calendar_arrays["year"][day]
    block[:, 8] = calendar_arrays["event_name_1"][day]
    block[:, 9] = calendar_arrays["event_type_1"][day]
    block[:, 10] = calendar_arrays["event_name_2"][day]
    block[:, 11] = calendar_arrays["event_type_2"][day]
    block[:, 12] = snap_feature
    block[:, 13] = price_now

    if USE_PRICE_NORM:
        price_norm = np.zeros_like(price_now)
        np.divide(price_now, price_max, out=price_norm, where=price_max > 0)
        block[:, 14] = price_norm

    if USE_PRICE_CHANGE:
        price_prev = get_price_for_week(prev_week, price_context)
        price_change = np.zeros_like(price_now)
        np.divide(price_now - price_prev, price_prev, out=price_change, where=price_prev > 0)
        block[:, 15] = price_change

    block[:, 16] = sales_matrix[:, day - 2]  # lag_1 — always on
    if USE_LAG_7:
        block[:, 17] = sales_matrix[:, day - 8]
    if USE_LAG_28:
        block[:, 18] = sales_matrix[:, day - 29]
    if USE_ROLLING_MEAN_7:
        block[:, 19] = sales_matrix[:, day - 8 : day - 1].mean(axis=1)
    if USE_ROLLING_MEAN_28:
        block[:, 20] = sales_matrix[:, day - 29 : day - 1].mean(axis=1)
    if USE_ROLLING_STD_28:
        block[:, 21] = sales_matrix[:, day - 29 : day - 1].std(axis=1)

    return block


def build_training_arrays(
    sales_matrix: np.ndarray,
    train_start_day: int,
    train_end_day: int,
    static_codes: np.ndarray,
    is_ca: np.ndarray,
    is_tx: np.ndarray,
    is_wi: np.ndarray,
    calendar_arrays: dict[str, np.ndarray],
    price_context: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_series = sales_matrix.shape[0]
    num_days = train_end_day - train_start_day + 1
    total_rows = n_series * num_days

    X = np.empty((total_rows, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.empty(total_rows, dtype=np.float32)
    day_index = np.empty(total_rows, dtype=np.int32)

    for day_idx, day in enumerate(range(train_start_day, train_end_day + 1)):
        block = build_feature_block(
            sales_matrix=sales_matrix,
            day=day,
            static_codes=static_codes,
            is_ca=is_ca,
            is_tx=is_tx,
            is_wi=is_wi,
            calendar_arrays=calendar_arrays,
            price_context=price_context,
        )
        start = day_idx * n_series
        end = (day_idx + 1) * n_series

        X[start:end, :] = block
        y[start:end] = sales_matrix[:, day - 1]
        day_index[start:end] = day

    return X, y, day_index


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
):
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency `lightgbm`. Install requirements first, for example: "
            "`pip install lightgbm pandas numpy`"
        ) from exc

    model = lgb.LGBMRegressor(
        objective=OBJECTIVE,
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        min_data_in_leaf=MIN_DATA_IN_LEAF,
        subsample=SUBSAMPLE,
        subsample_freq=SUBSAMPLE_FREQ,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True), lgb.log_evaluation(LOG_EVAL_PERIOD)],
    )

    if hasattr(model, "feature_names_in_"):
        delattr(model, "feature_names_in_")

    valid_pred = np.maximum(predict_with_suppressed_feature_name_warning(model, X_valid), 0.0)
    valid_rmse = float(np.sqrt(np.mean((y_valid - valid_pred) ** 2)))
    return model, valid_rmse


def predict_with_suppressed_feature_name_warning(model, features: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        return model.predict(features)


def ensure_future_capacity(sales_matrix: np.ndarray, final_day: int) -> np.ndarray:
    if sales_matrix.shape[1] >= final_day:
        return sales_matrix
    pad_width = final_day - sales_matrix.shape[1]
    return np.pad(sales_matrix, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)


def recursive_forecast(
    model,
    sales_matrix: np.ndarray,
    last_known_day: int,
    static_codes: np.ndarray,
    is_ca: np.ndarray,
    is_tx: np.ndarray,
    is_wi: np.ndarray,
    calendar_arrays: dict[str, np.ndarray],
    price_context: dict[str, object],
) -> np.ndarray:
    sales_matrix = ensure_future_capacity(sales_matrix, last_known_day + FORECAST_HORIZON)
    n_series = sales_matrix.shape[0]
    predictions = np.zeros((n_series, FORECAST_HORIZON), dtype=np.float32)

    for step, day in enumerate(range(last_known_day + 1, last_known_day + FORECAST_HORIZON + 1)):
        block = build_feature_block(
            sales_matrix=sales_matrix,
            day=day,
            static_codes=static_codes,
            is_ca=is_ca,
            is_tx=is_tx,
            is_wi=is_wi,
            calendar_arrays=calendar_arrays,
            price_context=price_context,
        )
        pred = np.maximum(predict_with_suppressed_feature_name_warning(model, block), 0.0).astype(np.float32)
        sales_matrix[:, day - 1] = pred
        predictions[:, step] = pred

    return predictions


def make_submission(
    sample_submission_path: Path,
    validation_ids: np.ndarray,
    evaluation_ids: np.ndarray,
    validation_pred: np.ndarray,
    evaluation_pred: np.ndarray,
) -> pd.DataFrame:
    forecast_columns = [f"F{idx}" for idx in range(1, FORECAST_HORIZON + 1)]

    validation_output = pd.DataFrame(validation_pred, columns=forecast_columns)
    validation_output.insert(0, "id", validation_ids)

    evaluation_output = pd.DataFrame(evaluation_pred, columns=forecast_columns)
    evaluation_output.insert(0, "id", evaluation_ids)

    all_predictions = pd.concat([validation_output, evaluation_output], axis=0, ignore_index=True)
    sample = pd.read_csv(sample_submission_path)

    submission = sample[["id"]].merge(all_predictions, on="id", how="left")
    submission[forecast_columns] = submission[forecast_columns].fillna(0.0)
    return submission


def train() -> None:
    ensure_required_files()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading calendar and sales tables ...")
    calendar_arrays = load_calendar_arrays(DATA_DIR / "calendar.csv")
    sales_validation, matrix_validation, last_day_validation = load_sales_table(DATA_DIR / "sales_train_validation.csv")
    sales_evaluation, matrix_evaluation, last_day_evaluation = load_sales_table(DATA_DIR / "sales_train_evaluation.csv")

    logger.info("Preparing static encoders and price context ...")
    encoders, static_codes_validation = fit_static_encoders(sales_validation)
    static_codes_evaluation = transform_static_features(sales_evaluation, encoders)

    state_validation = sales_validation["state_id"].astype(str).to_numpy()
    is_ca_validation = state_validation == "CA"
    is_tx_validation = state_validation == "TX"
    is_wi_validation = state_validation == "WI"

    state_evaluation = sales_evaluation["state_id"].astype(str).to_numpy()
    is_ca_evaluation = state_evaluation == "CA"
    is_tx_evaluation = state_evaluation == "TX"
    is_wi_evaluation = state_evaluation == "WI"

    series_keys_validation = (
        sales_validation["store_id"].astype(str) + "_" + sales_validation["item_id"].astype(str)
    ).to_numpy()
    series_keys_evaluation = (
        sales_evaluation["store_id"].astype(str) + "_" + sales_evaluation["item_id"].astype(str)
    ).to_numpy()

    price_pivot = build_price_pivot(DATA_DIR / "sell_prices.csv")
    price_context_validation = build_price_context(price_pivot, series_keys_validation)
    price_context_evaluation = build_price_context(price_pivot, series_keys_evaluation)

    minimum_day_for_features = 29
    train_start_day = max(minimum_day_for_features, last_day_validation - TRAIN_WINDOW_DAYS + 1)
    train_end_day = last_day_validation

    if train_end_day - train_start_day + 1 <= VALID_DAYS:
        raise SystemExit("Not enough history to create train/validation split.")

    logger.info(
        "Building training arrays from d_%d to d_%d ...",
        train_start_day,
        train_end_day,
    )
    X, y, day_index = build_training_arrays(
        sales_matrix=matrix_validation,
        train_start_day=train_start_day,
        train_end_day=train_end_day,
        static_codes=static_codes_validation,
        is_ca=is_ca_validation,
        is_tx=is_tx_validation,
        is_wi=is_wi_validation,
        calendar_arrays=calendar_arrays,
        price_context=price_context_validation,
    )

    valid_start_day = train_end_day - VALID_DAYS + 1
    train_mask = day_index < valid_start_day
    valid_mask = day_index >= valid_start_day

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]

    logger.info("Training LightGBM model ...")
    model, valid_rmse = train_lightgbm(X_train, y_train, X_valid, y_valid)
    logger.info("Validation RMSE: %.6f", valid_rmse)

    evals = getattr(model, "evals_result_", None) or {}
    if evals:
        eval_set_name = next(iter(evals))
        metric_series = evals[eval_set_name]
        if metric_series:
            inner_metric = next(iter(metric_series))
            for epoch_idx, metric_value in enumerate(metric_series[inner_metric], start=1):
                record(epoch_idx, metric_value)

    logger.info("Generating recursive 28-day forecasts for validation and evaluation ids ...")
    validation_pred = recursive_forecast(
        model=model,
        sales_matrix=matrix_validation.copy(),
        last_known_day=last_day_validation,
        static_codes=static_codes_validation,
        is_ca=is_ca_validation,
        is_tx=is_tx_validation,
        is_wi=is_wi_validation,
        calendar_arrays=calendar_arrays,
        price_context=price_context_validation,
    )
    evaluation_pred = recursive_forecast(
        model=model,
        sales_matrix=matrix_evaluation.copy(),
        last_known_day=last_day_evaluation,
        static_codes=static_codes_evaluation,
        is_ca=is_ca_evaluation,
        is_tx=is_tx_evaluation,
        is_wi=is_wi_evaluation,
        calendar_arrays=calendar_arrays,
        price_context=price_context_evaluation,
    )

    submission = make_submission(
        sample_submission_path=DATA_DIR / "sample_submission.csv",
        validation_ids=sales_validation["id"].to_numpy(),
        evaluation_ids=sales_evaluation["id"].to_numpy(),
        validation_pred=validation_pred,
        evaluation_pred=evaluation_pred,
    )

    submission_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(submission_path, index=False)

    metrics = {
        "metric": "rmse",
        "validation_rmse": valid_rmse,
        "train_window_days": int(train_end_day - train_start_day + 1),
        "valid_days": VALID_DAYS,
        "n_series": int(matrix_validation.shape[0]),
        "n_train_rows": int(X_train.shape[0]),
        "n_valid_rows": int(X_valid.shape[0]),
        "n_features": len(FEATURE_COLUMNS),
    }
    metrics_path = OUTPUT_DIR / "training_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    importance_path = OUTPUT_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    logger.info("Saved submission to %s", submission_path)
    logger.info("Saved metrics to %s", metrics_path)
    logger.info("Saved feature importance to %s", importance_path)


def main() -> None:
    train()
    conclude()


if __name__ == "__main__":
    main()
