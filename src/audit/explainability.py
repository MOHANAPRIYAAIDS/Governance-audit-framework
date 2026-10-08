"""
Explainability audit module (survey section 4.3).

  - SHAP importance:     I_j = mean over rows of |phi_j|       (phi = TreeSHAP contribution)
  - Importance share:    share_j = I_j / sum_k I_k
  - Sensitive Feature Influence: sum of share_j over the sensitive features
  - Counterfactual flip rate: share of decisions that change when ONLY a protected
    attribute is changed (a model-level version of the single-applicant check the
    dashboard will offer)

Contributions come from XGBoost's built-in exact TreeSHAP (pred_contribs), on the
log-odds scale. Categorical columns stay single features, so OCCP and POBP are not
fragmented across dummy columns.

SHAP is an explanation, not proof of discrimination: read it next to the fairness
module. Thresholds are documented defaults to be sensitivity-tested.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from .common import ModuleResult, risk_from_violation, status_from_risk

SENSITIVE_FEATURES = ["RAC1P", "SEX", "POBP"]

DEFAULT_THRESHOLDS = {
    "sensitive_share": 0.10,
    "counterfactual_flip_rate": 0.05,
}


def shap_values(model, X: pd.DataFrame):
    """Exact TreeSHAP contributions (log-odds). Returns (DataFrame n x features, bias array)."""
    contribs = model.get_booster().predict(xgb.DMatrix(X, enable_categorical=True), pred_contribs=True)
    return pd.DataFrame(contribs[:, :-1], columns=X.columns, index=X.index), contribs[:, -1]


def global_importance(shap_df: pd.DataFrame) -> pd.DataFrame:
    imp = shap_df.abs().mean().rename("mean_abs_shap").to_frame()
    imp["share"] = imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()
    imp["rank"] = imp["mean_abs_shap"].rank(ascending=False).astype(int)
    return imp.sort_values("mean_abs_shap", ascending=False)


def _with_value(X: pd.DataFrame, feature: str, value) -> pd.DataFrame:
    Xc = X.copy()
    Xc[feature] = pd.Categorical([value] * len(X), categories=X[feature].cat.categories)
    return Xc


def counterfactual_flip_rate(model, X: pd.DataFrame, feature: str, values, decision_threshold=0.5) -> dict:
    """
    For each alternative value, set `feature` to it for every row and compare the decision
    with the original. Rows already holding that value are not counted. `any_flip_rate` is
    the share of rows whose decision changes under at least one alternative value.
    """
    base = model.predict_proba(X)[:, 1] >= decision_threshold
    current = X[feature].astype(float).to_numpy()
    flipped_any = np.zeros(len(X), dtype=bool)
    per_value = {}
    for v in values:
        pred = model.predict_proba(_with_value(X, feature, v))[:, 1] >= decision_threshold
        applicable = current != v
        changed = (pred != base) & applicable
        flipped_any |= changed
        per_value[str(v)] = float(changed.sum() / max(applicable.sum(), 1))
    return {"any_flip_rate": float(flipped_any.mean()), "per_value": per_value}


def explain_row(shap_df: pd.DataFrame, X: pd.DataFrame, idx, top: int = 5) -> pd.DataFrame:
    """Top feature contributions for one applicant (positive pushes towards eligible)."""
    out = pd.DataFrame({
        "value": X.loc[idx].astype(float),
        "contribution": shap_df.loc[idx],
    })
    return out.reindex(out["contribution"].abs().sort_values(ascending=False).index).head(top)


def audit_explainability(
    model,
    X: pd.DataFrame,
    sensitive=None,
    flip_values: dict = None,
    thresholds=None,
    max_rows: int = None,
    decision_threshold: float = 0.5,
    random_state: int = 42,
) -> ModuleResult:
    """flip_values maps a protected feature -> the values to try (e.g. {"SEX": [1, 2]})."""
    sensitive = sensitive or SENSITIVE_FEATURES
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if max_rows and len(X) > max_rows:
        X = X.sample(max_rows, random_state=random_state)

    shap_df, _ = shap_values(model, X)
    imp = global_importance(shap_df)
    sens_total = float(imp.loc[sensitive, "share"].sum())

    risks = {"sensitive_share": risk_from_violation(sens_total, thresholds["sensitive_share"])}
    flips = {}
    for feat, vals in (flip_values or {}).items():
        flips[feat] = counterfactual_flip_rate(model, X, feat, vals, decision_threshold)
        risks[f"flip_{feat}"] = risk_from_violation(
            flips[feat]["any_flip_rate"], thresholds["counterfactual_flip_rate"]
        )
    risk = max(risks.values())

    top = imp.head(3)
    findings = [
        "Most influential features: "
        + ", ".join(f"{f} ({r.share:.1%})" for f, r in top.iterrows()) + "."
    ]
    if sens_total > thresholds["sensitive_share"]:
        findings.append(
            f"Sensitive features {sensitive} carry {sens_total:.1%} of total SHAP importance "
            f"(limit {thresholds['sensitive_share']:.0%}). Read with the fairness module: "
            "this shows influence, not proof of discrimination."
        )
    for feat, f in flips.items():
        if f["any_flip_rate"] > thresholds["counterfactual_flip_rate"]:
            findings.append(
                f"{f['any_flip_rate']:.1%} of decisions change when only {feat} is changed "
                f"(limit {thresholds['counterfactual_flip_rate']:.0%})."
            )

    return ModuleResult(
        name="Explainability",
        metrics={
            "sensitive_share_total": sens_total,
            "sensitive_features": {
                f: {"share": float(imp.loc[f, "share"]), "rank": int(imp.loc[f, "rank"])} for f in sensitive
            },
            "top_features": [(f, float(s)) for f, s in imp["share"].head(5).items()],
            "counterfactual": flips,
            "risks": risks,
        },
        risk=risk,
        status=status_from_risk(risk),
        findings=findings,
        details={"importance": imp, "shap_values": shap_df, "X": X},
    )
