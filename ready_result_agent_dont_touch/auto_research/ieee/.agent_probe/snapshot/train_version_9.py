"""Train a LightGBM baseline for IEEE-CIS fraud detection.

Run:
    python3 train.py
"""

from __future__ import annotations

import json
import logging

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
# potential_improvement_1: set TRAIN_ROW_LIMIT to 0 (or None) to train on the full ~590k transactions instead of just 15k — fraud is rare (~3.5%), so the tiny subset starves the model of positives and severely depresses validation ROC-AUC. # applied
TRAIN_ROW_LIMIT = None
# potential_improvement_2: set USE_IDENTITY_FEATURES = True — the identity table (DeviceType, DeviceInfo, id_01..id_38) carries strong fraud signal and adding it typically lifts validation ROC-AUC by 0.01-0.03. # applied
USE_IDENTITY_FEATURES = True
# potential_improvement_3: set VALID_TIME_QUANTILE to 0.8 — using only the last 20% as validation gives 33% more training rows (and therefore more positives) which directly raises validation ROC-AUC for time-ordered fraud data.
VALID_TIME_QUANTILE = 0.6
RANDOM_STATE = 42

# Feature inclusion / encoding
# potential_improvement_4: set DROP_FEATURE_PREFIXES = [] — V/C/D/M columns are the most predictive engineered features in this dataset (Vesta's count/distance/match aggregations); dropping all of them throws away the bulk of the signal.
DROP_FEATURE_PREFIXES = []
USE_AMOUNT_LOG = False
# potential_improvement_5: set FILLNA_VALUE = np.nan (i.e. skip the fillna) — LightGBM handles missingness natively and treats NaN as an informative split direction; replacing NaN with 0.0 collides with real zeros and destroys that signal. # applied
FILLNA_VALUE = np.nan

# Model
# potential_improvement_6: raise N_ESTIMATORS to 3000-5000 — combined with a lower learning rate and early stopping, this lets the booster actually converge; 30 trees is far too few to reach a competitive ROC-AUC on this dataset. # applied
N_ESTIMATORS = 3000
# potential_improvement_7: lower LEARNING_RATE to 0.03-0.05 — 0.3 is far too aggressive for boosting on a noisy imbalanced classification problem and causes the model to overshoot good splits, capping validation ROC-AUC.
LEARNING_RATE = 0.3
# potential_improvement_8: raise NUM_LEAVES to 127 (or 255) — 4 leaves per tree is drastically underfit on hundreds of features; richer trees capture the high-order interactions that drive fraud detection ROC-AUC.
NUM_LEAVES = 4
MAX_DEPTH = 3
# potential_improvement_9: lower MIN_CHILD_SAMPLES to 20 — at 300 the model cannot create leaves around the rare fraud class (which has very few rows per region), leaving signal unmodelled and hurting ROC-AUC.
MIN_CHILD_SAMPLES = 300
SUBSAMPLE = 0.4
SUBSAMPLE_FREQ = 1
COLSAMPLE_BYTREE = 0.4
REG_ALPHA = 5.0
REG_LAMBDA = 5.0
# potential_improvement_10: raise EARLY_STOPPING_ROUNDS to 150-200 — patience of 4 stops training the moment ROC-AUC plateaus briefly, well before the true optimum, especially once the learning rate is reduced. # applied
EARLY_STOPPING_ROUNDS = 50
USE_SCALE_POS_WEIGHT = False
LOG_EVAL_PERIOD = 500
N_JOBS = -1


def load_and_merge_tables(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info('Loading transaction tables ...')
    train_transaction = pd.read_csv(data_dir / 'train_transaction.csv')
    test_transaction = pd.read_csv(data_dir / 'test_transaction.csv')

    if USE_IDENTITY_FEATURES:
        logger.info('Loading and merging identity tables ...')
        train_identity = pd.read_csv(data_dir / 'train_identity.csv')
        test_identity = pd.read_csv(data_dir / 'test_identity.csv')
        train_merged = train_transaction.merge(train_identity, on='TransactionID', how='left')
        test_merged = test_transaction.merge(test_identity, on='TransactionID', how='left')
    else:
        train_merged = train_transaction
        test_merged = test_transaction

    return train_merged, test_merged


def drop_columns_by_prefix(frame: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    if not prefixes:
        return frame
    drop_cols = [c for c in frame.columns if any(c.startswith(p) for p in prefixes)]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
    return frame


def encode_categorical_columns(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_columns = sorted(set(train_frame.columns) | set(test_frame.columns))
    train_frame = train_frame.reindex(columns=all_columns)
    test_frame = test_frame.reindex(columns=all_columns)

    categorical_columns = [
        column
        for column in all_columns
        if pd.api.types.is_object_dtype(train_frame[column].dtype)
        or pd.api.types.is_string_dtype(train_frame[column].dtype)
        or isinstance(train_frame[column].dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(test_frame[column].dtype)
        or pd.api.types.is_string_dtype(test_frame[column].dtype)
        or isinstance(test_frame[column].dtype, pd.CategoricalDtype)
    ]

    logger.info('Encoding %d categorical columns ...', len(categorical_columns))
    for column in categorical_columns:
        combined = pd.concat([train_frame[column], test_frame[column]], axis=0)
        combined = combined.astype('string').fillna('__MISSING__')

        codes, _ = pd.factorize(combined, sort=False)
        train_frame[column] = codes[: len(train_frame)].astype(np.int32)
        test_frame[column] = codes[len(train_frame) :].astype(np.int32)

    return train_frame, test_frame


def preprocess_features(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_copy = train_frame.copy()
    test_copy = test_frame.copy()

    train_copy = drop_columns_by_prefix(train_copy, DROP_FEATURE_PREFIXES)
    test_copy = drop_columns_by_prefix(test_copy, DROP_FEATURE_PREFIXES)

    if USE_AMOUNT_LOG and 'TransactionAmt' in train_copy.columns:
        train_copy['TransactionAmt'] = np.log1p(train_copy['TransactionAmt'].clip(lower=0))
        test_copy['TransactionAmt'] = np.log1p(test_copy['TransactionAmt'].clip(lower=0))

    train_copy, test_copy = encode_categorical_columns(train_copy, test_copy)

    train_copy = train_copy.replace([np.inf, -np.inf], np.nan).fillna(FILLNA_VALUE)
    test_copy = test_copy.replace([np.inf, -np.inf], np.nan).fillna(FILLNA_VALUE)

    for column in train_copy.columns:
        if pd.api.types.is_float_dtype(train_copy[column]):
            train_copy[column] = train_copy[column].astype(np.float32)
            test_copy[column] = test_copy[column].astype(np.float32)
        elif pd.api.types.is_integer_dtype(train_copy[column]):
            train_copy[column] = train_copy[column].astype(np.int32)
            test_copy[column] = test_copy[column].astype(np.int32)

    return train_copy, test_copy


def time_based_validation_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, float]:
    time_threshold = float(features['TransactionDT'].quantile(VALID_TIME_QUANTILE))
    train_mask = features['TransactionDT'] <= time_threshold
    valid_mask = features['TransactionDT'] > time_threshold

    if valid_mask.sum() == 0 or train_mask.sum() == 0:
        logger.warning('Time split failed (empty side), falling back to stratified random split.')
        X_train, X_valid, y_train, y_valid = train_test_split(
            features,
            target,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=target,
        )
        return X_train, X_valid, y_train, y_valid, float('nan')

    X_train = features.loc[train_mask].copy()
    y_train = target.loc[train_mask].copy()
    X_valid = features.loc[valid_mask].copy()
    y_valid = target.loc[valid_mask].copy()

    return X_train, X_valid, y_train, y_valid, time_threshold


def train() -> None:
    try:
        import lightgbm as lgb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            'Missing dependency `lightgbm`. Install required packages first: '
            '`pip install lightgbm pandas scikit-learn`'
        ) from exc

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_merged, test_merged = load_and_merge_tables(DATA_DIR)

    if TRAIN_ROW_LIMIT and TRAIN_ROW_LIMIT > 0:
        train_merged = train_merged.head(TRAIN_ROW_LIMIT).copy()

    y = train_merged['isFraud'].astype(int)
    test_ids = test_merged['TransactionID'].copy()

    train_features = train_merged.drop(columns=['isFraud'])
    test_features = test_merged.copy()

    logger.info('Preprocessing features ...')
    train_features, test_features = preprocess_features(train_features, test_features)

    X = train_features.drop(columns=['TransactionID'], errors='ignore')
    X_test = test_features.drop(columns=['TransactionID'], errors='ignore')

    X_train, X_valid, y_train, y_valid, time_threshold = time_based_validation_split(X, y)

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
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        num_leaves=NUM_LEAVES,
        max_depth=MAX_DEPTH,
        min_child_samples=MIN_CHILD_SAMPLES,
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
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True), lgb.log_evaluation(LOG_EVAL_PERIOD)],
    )

    eval_history = getattr(model, 'evals_result_', {}) or {}
    valid_history = eval_history.get('valid_0') or eval_history.get('validation_0') or {}
    auc_history = valid_history.get('auc') or valid_history.get('AUC') or []
    for epoch_idx, auc_value in enumerate(auc_history, start=1):
        record(epoch_idx, auc_value)

    val_pred = model.predict_proba(X_valid)[:, 1]
    val_auc = float(roc_auc_score(y_valid, val_pred))
    logger.info('Validation ROC-AUC: %.6f', val_auc)

    logger.info('Generating test predictions ...')
    test_pred = model.predict_proba(X_test)[:, 1]

    sample_submission_path = DATA_DIR / 'sample_submission.csv'
    if sample_submission_path.exists():
        submission = pd.read_csv(sample_submission_path)
        if 'TransactionID' not in submission.columns:
            submission.insert(0, 'TransactionID', test_ids.values)
        submission['isFraud'] = test_pred
    else:
        submission = pd.DataFrame({'TransactionID': test_ids.values, 'isFraud': test_pred})

    submission_path = OUTPUT_DIR / 'submission.csv'
    submission.to_csv(submission_path, index=False)

    metrics = {
        'metric': 'roc_auc',
        'value': val_auc,
        'n_features': int(X.shape[1]),
        'n_train_rows': int(X_train.shape[0]),
        'n_valid_rows': int(X_valid.shape[0]),
        'best_iteration': int(model.best_iteration_ or 0),
        'time_split_quantile': VALID_TIME_QUANTILE,
        'time_split_threshold': time_threshold,
    }
    metrics_path = OUTPUT_DIR / 'validation_metrics.json'
    with metrics_path.open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)

    feature_importance = pd.DataFrame(
        {
            'feature': X.columns,
            'importance_gain': model.booster_.feature_importance(importance_type='gain'),
            'importance_split': model.booster_.feature_importance(importance_type='split'),
        }
    ).sort_values('importance_gain', ascending=False)
    feature_importance_path = OUTPUT_DIR / 'feature_importance.csv'
    feature_importance.to_csv(feature_importance_path, index=False)

    logger.info('Saved submission: %s', submission_path)
    logger.info('Saved metrics: %s', metrics_path)
    logger.info('Saved feature importance: %s', feature_importance_path)


def main() -> None:
    try:
        train()
    finally:
        conclude()


if __name__ == '__main__':
    main()
