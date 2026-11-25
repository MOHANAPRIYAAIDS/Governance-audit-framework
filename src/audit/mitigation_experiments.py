"""Reproducible fairness-mitigation experiments for the ACS income baseline.

These experiments compare alternatives; they do not overwrite the baseline model.
Proxy association is a univariate screening heuristic, not proof of proxy use.
"""

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import mutual_info_score
from xgboost import XGBClassifier

from ..model import evaluate, load_processed
from ..tasks import INCOME
from .explainability import counterfactual_flip_rate, shap_values
from .fairness import audit_fairness

PROTECTED_FEATURES = ("RAC1P", "SEX")
PROXY_CANDIDATES_PER_PROTECTED = 1
SHAP_SAMPLE_SIZE = 5000


def _discretize_feature(values: pd.Series, numeric: bool) -> pd.Series:
    if numeric:
        bins = min(10, int(values.nunique()))
        return pd.qcut(values, q=bins, duplicates="drop").astype(str)
    return values.astype(str)


def screen_proxy_associations(
    X_train: pd.DataFrame,
    protected_features: tuple = PROTECTED_FEATURES,
) -> dict:
    """Rank train-set feature associations with each protected attribute using MI."""
    output = {}
    for protected in protected_features:
        if protected not in X_train.columns:
            raise ValueError(f"Protected feature {protected!r} is missing from X_train")
        protected_values = X_train[protected].astype(str)
        protected_entropy = mutual_info_score(protected_values, protected_values)
        rows = []
        for feature in X_train.columns:
            if feature == protected:
                continue
            values = _discretize_feature(
                X_train[feature], numeric=feature in INCOME.numeric_cols
            )
            mi = mutual_info_score(protected_values, values)
            rows.append(
                {
                    "feature": feature,
                    "mutual_information": float(mi),
                    "protected_entropy_fraction": (
                        float(mi / protected_entropy) if protected_entropy else 0.0
                    ),
                }
            )
        output[protected] = sorted(
            rows, key=lambda row: row["mutual_information"], reverse=True
        )
    return output


def group_reweighing_sample_weights(
    protected_values: pd.Series, labels: pd.Series
) -> pd.Series:
    """Return classic group-label reweighing weights, aligned to the input index."""
    if not protected_values.index.equals(labels.index):
        raise ValueError("protected_values and labels must have identical indexes")
    frame = pd.DataFrame(
        {"group": protected_values.astype(str), "label": labels.astype(int)},
        index=labels.index,
    )
    group_counts = frame["group"].value_counts()
    label_counts = frame["label"].value_counts()
    joint_counts = frame.groupby(["group", "label"])["label"].transform("size")
    weights = (
        frame["group"].map(group_counts)
        * frame["label"].map(label_counts)
        / (len(frame) * joint_counts)
    ).astype(float)
    weights /= weights.mean()
    weights.name = "sample_weight"
    return weights


def _model_frames(
    X_train: pd.DataFrame, X_test: pd.DataFrame, dropped_features: set
) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept = [column for column in X_train.columns if column not in dropped_features]
    if not kept:
        raise ValueError("At least one feature must remain for model training")
    categorical = [column for column in INCOME.categorical_cols if column in kept]
    train, test = X_train[kept].copy(), X_test[kept].copy()
    for column in categorical:
        combined = pd.concat([train[column], test[column]]).astype(int)
        dtype = pd.CategoricalDtype(categories=sorted(combined.unique()))
        train[column] = train[column].astype(int).astype(dtype)
        test[column] = test[column].astype(int).astype(dtype)
    return train, test


def _new_model(random_state: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        tree_method="hist",
        enable_categorical=True,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def _shap_direction_summary(
    model: XGBClassifier, X_test: pd.DataFrame, sample_size: int, random_state: int
) -> list:
    sample = X_test.sample(min(sample_size, len(X_test)), random_state=random_state)
    contributions, _ = shap_values(model, sample)
    rows = []
    for feature in contributions.columns:
        values = contributions[feature]
        rows.append(
            {
                "feature": feature,
                "mean_absolute_shap": float(values.abs().mean()),
                "mean_signed_shap": float(values.mean()),
                "positive_contribution_share": float((values > 0).mean()),
                "negative_contribution_share": float((values < 0).mean()),
            }
        )
    return sorted(rows, key=lambda row: row["mean_absolute_shap"], reverse=True)


def run_fairness_mitigation_experiments(
    processed_dir: str,
    baseline_model_path: str,
    output_path: str,
    random_state: int = 42,
) -> dict:
    """Run baseline, feature-ablation, reweighing, SHAP, and what-if comparisons."""
    X_train, X_test, y_train, y_test, _, _ = load_processed(processed_dir)
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    proxy_screen = screen_proxy_associations(X_train)
    proxy_candidates = list(
        dict.fromkeys(
            row["feature"]
            for rows in proxy_screen.values()
            for row in rows[:PROXY_CANDIDATES_PER_PROTECTED]
        )
    )

    _, baseline_test = _model_frames(X_train, X_test, set())
    baseline = XGBClassifier()
    baseline.load_model(baseline_model_path)

    variants = [
        {
            "name": "baseline",
            "dropped": set(),
            "weight_attribute": None,
            "model": baseline,
            "X_test": baseline_test,
        }
    ]
    variant_specs = [
        ("without_protected_features", set(PROTECTED_FEATURES), None),
        (
            "without_protected_features_and_top_proxy_candidates",
            set(PROTECTED_FEATURES) | set(proxy_candidates),
            None,
        ),
        ("reweighted_by_RAC1P", set(), "RAC1P"),
        ("reweighted_by_SEX", set(), "SEX"),
    ]

    for name, dropped, weight_attribute in variant_specs:
        train_frame, test_frame = _model_frames(X_train, X_test, dropped)
        model = _new_model(random_state)
        fit_kwargs = {}
        if weight_attribute is not None:
            fit_kwargs["sample_weight"] = group_reweighing_sample_weights(
                X_train[weight_attribute], y_train
            ).to_numpy()
        model.fit(train_frame, y_train, **fit_kwargs)
        variants.append(
            {"name": name, "dropped": dropped, "weight_attribute": weight_attribute,
             "model": model, "X_test": test_frame}
        )

    group_values = {
        "RAC1P": X_test["RAC1P"].to_numpy(),
        "SEX": X_test["SEX"].to_numpy(),
    }
    race_counts = X_test["RAC1P"].astype(int).value_counts()
    race_flip_values = sorted(race_counts[race_counts >= 30].index.tolist())
    results = {}
    for variant in variants:
        model = variant["model"]
        X_eval = variant["X_test"]
        probabilities = model.predict_proba(X_eval)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        fairness = audit_fairness(y_test, predictions, group_values)
        flips = {}
        for feature, values in (("SEX", [1, 2]), ("RAC1P", race_flip_values)):
            if feature in X_eval.columns:
                flips[feature] = counterfactual_flip_rate(model, X_eval, feature, values)
        shap_summary = _shap_direction_summary(
            model, X_eval, SHAP_SAMPLE_SIZE, random_state
        )
        total_importance = sum(row["mean_absolute_shap"] for row in shap_summary)
        sensitive_importance = sum(
            row["mean_absolute_shap"]
            for row in shap_summary
            if row["feature"] in INCOME.sensitive
        )
        results[variant["name"]] = {
            "dropped_features": sorted(variant["dropped"]),
            "reweighing_attribute": variant["weight_attribute"],
            "performance": evaluate(y_test, predictions, probabilities),
            "fairness": fairness.metrics,
            "fairness_risk": fairness.risk,
            "fairness_status": fairness.status,
            "fairness_findings": fairness.findings,
            "counterfactual_flip_rates": flips,
            "sensitive_feature_importance_share": (
                float(sensitive_importance / total_importance) if total_importance else 0.0
            ),
            "shap_direction_summary": shap_summary,
        }

    payload = {
        "task": "predicting income > $50,000; not insurance eligibility",
        "sample_counts": {"train": len(X_train), "test": len(X_test)},
        "proxy_screen_method": (
            "Univariate mutual information on training data; numeric features "
            "were quantile-binned. Association is a screening signal, not proof of proxy use."
        ),
        "proxy_associations": proxy_screen,
        "proxy_candidates_removed": proxy_candidates,
        "reweighing_method": (
            "Group-label reweighing applied separately to RAC1P and SEX, using training rows only."
        ),
        "shap_note": (
            "TreeSHAP contributions are on the log-odds scale. Positive signed SHAP pushes "
            "toward the >$50,000 class; summaries use a deterministic test sample."
        ),
        "variants": results,
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=lambda value: value.item())
    return payload
