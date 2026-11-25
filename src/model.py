"""
Step 2 of the implementation plan: baseline eligibility model (XGBoost).

Loads the processed split produced by notebook 01 / src/data_pipeline.py,
trains an XGBoost classifier, and exposes predictions and probabilities
for the audit modules (fairness, explainability, reliability, robustness).
"""

import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from .tasks import INCOME, TaskSpec

CATEGORICAL_COLS = list(INCOME.categorical_cols)
NUMERIC_COLS = list(INCOME.numeric_cols)
FEATURE_COLS = INCOME.feature_cols


def load_processed(processed_dir: str, task: TaskSpec = INCOME):
    """Read train.csv / test.csv written by the preprocessing notebook and split into X, y, group."""
    train = pd.read_csv(os.path.join(processed_dir, "train.csv"))
    test = pd.read_csv(os.path.join(processed_dir, "test.csv"))

    X_train, X_test = train[task.feature_cols].copy(), test[task.feature_cols].copy()
    y_train, y_test = train["label"].astype(int), test["label"].astype(int)
    g_train, g_test = train["group"], test["group"]
    return X_train, X_test, y_train, y_test, g_train, g_test


def to_model_frame(X_train: pd.DataFrame, X_test: pd.DataFrame, task: TaskSpec = INCOME):
    """
    Cast categorical columns to pandas 'category' dtype using one shared set of
    categories (union of train and test), so XGBoost's native categorical
    support sees identical category codes in both splits. Native categorical
    handling avoids one-hot encoding the high-cardinality OCCP (528) and POBP
    (221) columns.
    """
    X_train, X_test = X_train.copy(), X_test.copy()
    for col in task.categorical_cols:
        combined = pd.concat([X_train[col], X_test[col]]).astype(int)
        dtype = pd.CategoricalDtype(categories=sorted(combined.unique()))
        X_train[col] = X_train[col].astype(int).astype(dtype)
        X_test[col] = X_test[col].astype(int).astype(dtype)
    return X_train, X_test


def build_dtypes(X_train: pd.DataFrame, X_test: pd.DataFrame, task: TaskSpec = INCOME) -> dict:
    """Categorical dtypes (union of train and test categories), reusable on new data."""
    dtypes = {}
    for col in task.categorical_cols:
        combined = pd.concat([X_train[col], X_test[col]]).astype(int)
        dtypes[col] = pd.CategoricalDtype(categories=sorted(combined.unique()))
    return dtypes


def apply_dtypes(df: pd.DataFrame, dtypes: dict, task: TaskSpec = INCOME) -> pd.DataFrame:
    """
    Turn raw feature values (e.g. an uploaded CSV or one applicant) into the frame the
    model expects. Non-numeric values and category codes never seen in training become
    missing (NaN), which XGBoost handles natively.
    """
    X = pd.DataFrame(index=df.index)
    for col in task.numeric_cols:
        X[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    for col in task.categorical_cols:
        cats = {int(c): i for i, c in enumerate(dtypes[col].categories)}
        num = pd.to_numeric(df[col], errors="coerce")
        codes = num.map(lambda v: cats.get(int(round(v)), -1) if pd.notna(v) else -1)
        X[col] = pd.Categorical.from_codes(codes.to_numpy(dtype=int), dtype=dtypes[col])
    return X[task.feature_cols]


def parse_binary_label(series: pd.Series):
    """Return (labels as 0/1 ints, None) or (None, reason it cannot be used)."""
    s = pd.to_numeric(series, errors="coerce")
    if s.isna().any():
        return None, f"{int(s.isna().sum())} label values are missing or not numbers"
    if not set(s.unique()) <= {0, 1}:
        return None, "labels must be 0 (not eligible) or 1 (eligible)"
    return s.astype(int).to_numpy(), None


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }
