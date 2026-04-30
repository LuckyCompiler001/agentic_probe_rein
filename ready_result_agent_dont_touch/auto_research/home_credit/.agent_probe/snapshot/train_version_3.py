"""Train a Home Credit default-risk baseline model.

This script follows the README task definition:
- Binary classification (`TARGET`)
- ROC-AUC validation metric
- LightGBM baseline on `application_train.csv`

Run:
    python3 train.py
"""

from __future__ import annotations

import json
import logging
import re

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from prober import record, conclude


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / 'data'
OUTPUT_DIR = SCRIPT_DIR / 'result'

# Data / split
# potential_improvement_1: set TRAIN_ROW_LIMIT to 0 (or None) to train on the full ~307k application_train rows; capping at 2000 starves the model of signal and is the single largest depressor of validation AUC. # applied
TRAIN_ROW_LIMIT = None
# potential_improvement_2: set TEST_SIZE to 0.2 (instead of 0.5) so ~80% of rows go to training; with the current 50/50 split LightGBM sees half the signal it could and AUC suffers correspondingly.
TEST_SIZE = 0.5
RANDOM_STATE = 42
# potential_improvement_3: set INCLUDE_AUXILIARY = True to merge bureau / previous_application / POS / installments / credit_card aggregates; these tables historically lift Home Credit AUC by ~0.02-0.03 over application-only features.
INCLUDE_AUXILIARY = False
KEEP_ID_AS_FEATURE = True

# Feature inclusion / exclusion
# potential_improvement_4: set DROP_FEATURE_COLUMNS = [] — EXT_SOURCE_1/2/3 are by far the strongest predictors in this dataset (each contributes ~0.01-0.02 AUC); dropping them is the second-largest performance hit in this script. # applied
DROP_FEATURE_COLUMNS = []

# Encoding / missing-value handling
DUMMY_NA = False
# potential_improvement_5: set FILLNA_VALUE = np.nan so LightGBM can use its native missing-value handling and split direction; replacing NaN with 0.0 collapses missing/zero into the same value and degrades split quality, costing measurable AUC.
FILLNA_VALUE = 0.0
# potential_improvement_6: set APPLY_DAYS_SANITIZE = True so the 365243 sentinel in DAYS_EMPLOYED (and other DAYS_ columns) becomes NaN rather than a huge outlier; leaving the sentinel in place biases tree splits and reduces validation AUC.
APPLY_DAYS_SANITIZE = False

# Model
# potential_improvement_7: raise N_ESTIMATORS to 2000-5000 (relying on EARLY_STOPPING_ROUNDS to halt) — 50 trees is far below the optimum for this objective and forces the probe to terminate before AUC plateaus.
N_ESTIMATORS = 50
# potential_improvement_8: lower LEARNING_RATE to 0.02-0.05; combined with more boosting rounds this is the standard high-AUC recipe for Home Credit and routinely beats lr=0.2 by 0.005-0.015 AUC.
LEARNING_RATE = 0.2
# potential_improvement_9: raise NUM_LEAVES to 31-63 (and set MAX_DEPTH to -1 or 8) so the trees can model the rich tabular interactions in this dataset; 8 leaves at depth 4 underfits and caps achievable AUC.
NUM_LEAVES = 8
MAX_DEPTH = 4
MIN_DATA_IN_LEAF = 100
SUBSAMPLE = 0.6
SUBSAMPLE_FREQ = 1
COLSAMPLE_BYTREE = 0.6
REG_ALPHA = 5.0
REG_LAMBDA = 5.0
# potential_improvement_10: raise EARLY_STOPPING_ROUNDS to 100-200 so training is not cut short by short-term noise on the validation curve; patience of 10 frequently halts before the true AUC peak.
EARLY_STOPPING_ROUNDS = 10
USE_SCALE_POS_WEIGHT = False
USE_STRATIFY = False
N_JOBS = -1


def sanitize_days_columns(frame: pd.DataFrame) -> pd.DataFrame:
    day_cols = [column for column in frame.columns if column.startswith('DAYS_')]
    for column in day_cols:
        frame[column] = frame[column].replace(365243, np.nan)
    return frame


def aggregate_numeric_table(table: pd.DataFrame, key: str, prefix: str) -> pd.DataFrame:
    if key not in table.columns:
        raise ValueError(f'Missing group key `{key}` in table `{prefix}`')

    table = sanitize_days_columns(table.copy())
    numeric_columns = [
        column
        for column in table.select_dtypes(include=[np.number]).columns
        if column != key
    ]

    if not numeric_columns:
        grouped = table[[key]].drop_duplicates().set_index(key)
    else:
        grouped = table.groupby(key)[numeric_columns].agg(['mean', 'max', 'min', 'sum'])
        grouped.columns = [f'{prefix}_{column}_{stat}'.upper() for column, stat in grouped.columns]

    grouped[f'{prefix}_ROW_COUNT'.upper()] = table.groupby(key).size()
    return grouped


def build_auxiliary_features(data_dir: Path) -> dict[str, pd.DataFrame]:
    logger.info('Building auxiliary aggregated features ...')
    features: dict[str, pd.DataFrame] = {}

    previous_application = pd.read_csv(data_dir / 'previous_application.csv')
    features['PREV'] = aggregate_numeric_table(previous_application, key='SK_ID_CURR', prefix='PREV')

    pos_cash_balance = pd.read_csv(data_dir / 'POS_CASH_balance.csv')
    features['POS_CASH'] = aggregate_numeric_table(pos_cash_balance, key='SK_ID_CURR', prefix='POS')

    installments = pd.read_csv(data_dir / 'installments_payments.csv')
    features['INSTAL'] = aggregate_numeric_table(installments, key='SK_ID_CURR', prefix='INSTAL')

    credit_card = pd.read_csv(data_dir / 'credit_card_balance.csv')
    features['CC'] = aggregate_numeric_table(credit_card, key='SK_ID_CURR', prefix='CC')

    bureau = pd.read_csv(data_dir / 'bureau.csv')
    bureau_balance = pd.read_csv(data_dir / 'bureau_balance.csv')
    bureau_balance_agg = aggregate_numeric_table(bureau_balance, key='SK_ID_BUREAU', prefix='BB')
    bureau_with_balance = bureau.merge(bureau_balance_agg, left_on='SK_ID_BUREAU', right_index=True, how='left')
    features['BUREAU'] = aggregate_numeric_table(bureau_with_balance, key='SK_ID_CURR', prefix='BUREAU')

    return features


def encode_features(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = pd.concat([train_frame, test_frame], axis=0, ignore_index=True)
    if DROP_FEATURE_COLUMNS:
        merged = merged.drop(columns=[c for c in DROP_FEATURE_COLUMNS if c in merged.columns], errors='ignore')
    if APPLY_DAYS_SANITIZE:
        merged = sanitize_days_columns(merged)
    categorical_columns = [
        column
        for column in merged.columns
        if pd.api.types.is_object_dtype(merged[column].dtype)
        or pd.api.types.is_string_dtype(merged[column].dtype)
    ]
    merged = pd.get_dummies(merged, columns=categorical_columns, dummy_na=DUMMY_NA)
    merged.columns = sanitize_feature_names(merged.columns)
    merged = merged.replace([np.inf, -np.inf], np.nan).fillna(FILLNA_VALUE)

    train_encoded = merged.iloc[: len(train_frame)].copy()
    test_encoded = merged.iloc[len(train_frame) :].copy()
    return train_encoded, test_encoded


def sanitize_feature_names(columns: pd.Index) -> list[str]:
    sanitized_columns: list[str] = []
    seen_names: dict[str, int] = {}

    for column in columns:
        normalized = re.sub(r'[^0-9A-Za-z_]+', '_', str(column)).strip('_')
        if not normalized:
            normalized = 'feature'
        if normalized[0].isdigit():
            normalized = f'F_{normalized}'

        duplicate_count = seen_names.get(normalized, 0)
        seen_names[normalized] = duplicate_count + 1
        if duplicate_count:
            normalized = f'{normalized}_{duplicate_count}'

        sanitized_columns.append(normalized)

    return sanitized_columns


def attach_auxiliary_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    auxiliary: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for feature_name, feature_frame in auxiliary.items():
        logger.info('Joining %s features (%d columns)', feature_name, feature_frame.shape[1])
        train_frame = train_frame.merge(feature_frame, left_on='SK_ID_CURR', right_index=True, how='left')
        test_frame = test_frame.merge(feature_frame, left_on='SK_ID_CURR', right_index=True, how='left')
    return train_frame, test_frame


def train() -> None:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Missing dependency `lightgbm`. Install required packages first: '
            '`pip install lightgbm pandas scikit-learn`'
        ) from exc

    data_dir = DATA_DIR
    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_dir / 'application_train.csv'
    test_path = data_dir / 'application_test.csv'
    sample_submission_path = data_dir / 'sample_submission.csv'

    logger.info('Loading application tables ...')
    train_application = pd.read_csv(train_path)
    test_application = pd.read_csv(test_path)

    if TRAIN_ROW_LIMIT and TRAIN_ROW_LIMIT > 0:
        train_application = train_application.head(TRAIN_ROW_LIMIT).copy()

    y = train_application['TARGET'].astype(int)
    train_ids = train_application['SK_ID_CURR'].copy()
    test_ids = test_application['SK_ID_CURR'].copy()

    train_features = train_application.drop(columns=['TARGET'])
    test_features = test_application.copy()

    if INCLUDE_AUXILIARY:
        auxiliary = build_auxiliary_features(data_dir)
        train_features, test_features = attach_auxiliary_features(train_features, test_features, auxiliary)

    logger.info('Encoding features ...')
    train_features, test_features = encode_features(train_features, test_features)

    if KEEP_ID_AS_FEATURE:
        X = train_features.copy()
        X_test = test_features.copy()
    else:
        X = train_features.drop(columns=['SK_ID_CURR'], errors='ignore')
        X_test = test_features.drop(columns=['SK_ID_CURR'], errors='ignore')

    stratify_values = y if USE_STRATIFY else None
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_values,
    )

    if USE_SCALE_POS_WEIGHT:
        pos_count = int(y_train.sum())
        neg_count = int((1 - y_train).sum())
        scale_pos_weight = neg_count / max(pos_count, 1)
    else:
        scale_pos_weight = 1.0

    logger.info('Training LightGBM baseline ...')
    model = lgb.LGBMClassifier(
        objective='binary',
        metric='auc',
        boosting_type='gbdt',
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
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True), lgb.log_evaluation(100)],
    )

    val_pred = model.predict_proba(X_valid)[:, 1]
    val_auc = float(roc_auc_score(y_valid, val_pred))
    logger.info('Validation ROC-AUC: %.6f', val_auc)

    evals_result = getattr(model, 'evals_result_', None) or {}
    eval_dict = evals_result.get('valid_0') or (next(iter(evals_result.values())) if evals_result else {})
    auc_history = eval_dict.get('auc') if isinstance(eval_dict, dict) else None
    if auc_history:
        for iteration, auc_value in enumerate(auc_history, start=1):
            record(iteration, auc_value)
    else:
        record(1, val_auc)

    test_pred = model.predict_proba(X_test)[:, 1]

    if sample_submission_path.exists():
        submission = pd.read_csv(sample_submission_path)
        if 'SK_ID_CURR' not in submission.columns:
            submission.insert(0, 'SK_ID_CURR', test_ids.values)
        submission['TARGET'] = test_pred
    else:
        submission = pd.DataFrame({'SK_ID_CURR': test_ids.values, 'TARGET': test_pred})

    submission_path = output_dir / 'submission.csv'
    submission.to_csv(submission_path, index=False)

    metrics_path = output_dir / 'validation_metrics.json'
    metrics = {
        'metric': 'roc_auc',
        'value': val_auc,
        'n_features': int(X.shape[1]),
        'n_train_rows': int(X_train.shape[0]),
        'n_valid_rows': int(X_valid.shape[0]),
        'include_auxiliary': bool(INCLUDE_AUXILIARY),
    }
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    feature_importance = pd.DataFrame(
        {
            'feature': X.columns,
            'importance_gain': model.booster_.feature_importance(importance_type='gain'),
            'importance_split': model.booster_.feature_importance(importance_type='split'),
        }
    ).sort_values('importance_gain', ascending=False)
    feature_importance.to_csv(output_dir / 'feature_importance.csv', index=False)

    logger.info('Saved submission: %s', submission_path)
    logger.info('Saved metrics: %s', metrics_path)

def main() -> None:
    try:
        train()
    finally:
        conclude()


if __name__ == '__main__':
    main()
