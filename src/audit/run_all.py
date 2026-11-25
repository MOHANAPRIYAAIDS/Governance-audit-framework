"""
Run every audit module on one model and one dataset, then aggregate the GRS.

Modules that need true labels (fairness, reliability, robustness accuracy, oversight)
are skipped when y_true is None and reported as not assessed, never guessed.
"""

import numpy as np
import pandas as pd

from .data_quality import audit_data_quality_result
from .explainability import audit_explainability
from .fairness import audit_fairness
from .grs import compute_grs
from .oversight import DEFAULT_REVIEW_THRESHOLD, audit_oversight
from .reliability import audit_reliability
from ..tasks import INCOME, TaskSpec
from .robustness import audit_robustness

MIN_GROUP_SIZE = 30


def default_groups(X: pd.DataFrame, task: TaskSpec = INCOME) -> dict:
    return {c: X[c].astype(float) for c in task.protected}


def default_flip_values(X: pd.DataFrame, task: TaskSpec = INCOME) -> dict:
    out = {}
    for c in task.flip_features:
        counts = X[c].astype(float).value_counts()
        out[c] = sorted(int(v) for v in counts[counts >= MIN_GROUP_SIZE].index)
    return out


def run_all_audits(
    model, X: pd.DataFrame, y_true=None, raw_df: pd.DataFrame = None, precomputed: dict = None,
    review_threshold=DEFAULT_REVIEW_THRESHOLD, decision_threshold=0.5,
    n_explain_rows=10000, weights=None, seed=42, task: TaskSpec = INCOME,
):
    """
    X: model-ready frame (categorical dtypes applied). raw_df: the untouched feature table,
    used for data quality (auditing cleaned data would hide the issues). precomputed:
    {dimension_key: ModuleResult} to use instead of recomputing (e.g. data quality
    from the raw extract). Returns (results, grs, probabilities).
    """
    p = model.predict_proba(X)[:, 1]
    pred = (p >= decision_threshold).astype(int)
    results = dict(precomputed or {})

    if "data_quality" not in results:
        results["data_quality"] = audit_data_quality_result(raw_df) if raw_df is not None else None

    results["explainability"] = audit_explainability(
        model, X, sensitive=list(task.sensitive), flip_values=default_flip_values(X, task),
        max_rows=n_explain_rows, decision_threshold=decision_threshold,
    )
    results["robustness"] = audit_robustness(
        model, X, y_true=y_true, decision_threshold=decision_threshold, seed=seed, task=task
    )

    if y_true is not None:
        try:
            results["fairness"] = audit_fairness(y_true, pred, default_groups(X, task))
        except ValueError:
            results["fairness"] = None  # fewer than two groups with enough rows
        results["reliability"] = audit_reliability(y_true, p, default_groups(X, task), decision_threshold=decision_threshold)
        results["oversight"] = audit_oversight(y_true, p, review_threshold, decision_threshold)
    else:
        for k in ("fairness", "reliability", "oversight"):
            results[k] = None

    return results, compute_grs(results, weights), p
