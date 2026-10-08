"""
Fairness audit module (survey section 4.2).

For each protected attribute, groups are compared on:
  - Demographic Parity Difference = max selection rate - min selection rate
  - Disparate Impact Ratio        = min selection rate / max selection rate
                                    (four-fifths rule: below 0.8 is a flag)
  - Equal Opportunity Difference  = max TPR - min TPR
  - Equalized Odds (FPR part)     = max FPR - min FPR   (TPR part is the line above)

Groups with fewer than MIN_GROUP_SIZE rows are excluded from the comparison and
reported, because rates on tiny groups are noise (e.g. RAC1P=4 in this data).

Thresholds are documented defaults to be sensitivity-tested, not legal limits.
The attribute's risk is the worst (max) of its metric risks, so one failing
metric is never averaged away.
"""

import numpy as np
import pandas as pd

from .common import ModuleResult, risk_from_violation, status_from_risk

MIN_GROUP_SIZE = 30

DEFAULT_THRESHOLDS = {
    "demographic_parity_difference": 0.10,
    "equal_opportunity_difference": 0.10,
    "equalized_odds_fpr_difference": 0.10,
    "disparate_impact_gap": 0.20,  # 1 - DIR; 0.20 corresponds to the 0.8 four-fifths rule
}


def group_table(y_true, y_pred, group) -> pd.DataFrame:
    df = pd.DataFrame({
        "y_true": np.asarray(y_true).astype(int),
        "y_pred": np.asarray(y_pred).astype(int),
        "g": np.asarray(group),
    })
    rows = []
    for g, d in df.groupby("g"):
        pos, neg = d.y_true == 1, d.y_true == 0
        rows.append({
            "group": g,
            "n": len(d),
            "selection_rate": d.y_pred.mean(),
            "true_rate": d.y_true.mean(),
            "tpr": d.y_pred[pos].mean() if pos.any() else np.nan,
            "fpr": d.y_pred[neg].mean() if neg.any() else np.nan,
            "accuracy": (d.y_true == d.y_pred).mean(),
        })
    return pd.DataFrame(rows).set_index("group")


def _spread(series: pd.Series):
    s = series.dropna()
    return float(s.max() - s.min()), s.idxmax(), s.idxmin()


def audit_attribute(y_true, y_pred, group, name, min_group_size=MIN_GROUP_SIZE, thresholds=None):
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    table = group_table(y_true, y_pred, group)
    eligible = table[table.n >= min_group_size]
    excluded = table[table.n < min_group_size]
    if len(eligible) < 2:
        raise ValueError(f"{name}: fewer than 2 groups have at least {min_group_size} rows")

    dpd, hi, lo = _spread(eligible.selection_rate)
    sr_max, sr_min = eligible.selection_rate.max(), eligible.selection_rate.min()
    dir_ = float(sr_min / sr_max) if sr_max > 0 else 1.0
    eod, _, _ = _spread(eligible.tpr)
    fprd, _, _ = _spread(eligible.fpr)

    values = {
        "demographic_parity_difference": dpd,
        "equal_opportunity_difference": eod,
        "equalized_odds_fpr_difference": fprd,
        "disparate_impact_gap": 1.0 - dir_,
    }
    risks = {k: risk_from_violation(v, thresholds[k]) for k, v in values.items()}

    findings = []
    if values["demographic_parity_difference"] > thresholds["demographic_parity_difference"]:
        findings.append(
            f"{name}: demographic parity difference {dpd:.3f} exceeds {thresholds['demographic_parity_difference']:.2f} "
            f"(group {hi} selected {sr_max:.1%}, group {lo} selected {sr_min:.1%})."
        )
    if values["disparate_impact_gap"] > thresholds["disparate_impact_gap"]:
        findings.append(f"{name}: disparate impact ratio {dir_:.3f} is below the four-fifths threshold (0.8).")
    if eod > thresholds["equal_opportunity_difference"]:
        findings.append(f"{name}: equal opportunity difference {eod:.3f} exceeds {thresholds['equal_opportunity_difference']:.2f}.")
    if fprd > thresholds["equalized_odds_fpr_difference"]:
        findings.append(f"{name}: false-positive-rate difference {fprd:.3f} exceeds {thresholds['equalized_odds_fpr_difference']:.2f}.")
    if len(excluded):
        findings.append(
            f"{name}: groups {list(excluded.index)} excluded (fewer than {min_group_size} rows); "
            "fairness cannot be assessed for them."
        )

    return {
        "metrics": {
            "demographic_parity_difference": dpd,
            "disparate_impact_ratio": dir_,
            "equal_opportunity_difference": eod,
            "equalized_odds_fpr_difference": fprd,
            "highest_selection_group": str(hi),
            "lowest_selection_group": str(lo),
        },
        "risks": risks,
        "risk": max(risks.values()),
        "table": table,
        "excluded": list(excluded.index),
        "findings": findings,
    }


def audit_fairness(y_true, y_pred, groups: dict, min_group_size=MIN_GROUP_SIZE, thresholds=None) -> ModuleResult:
    """groups maps attribute name -> array of group labels aligned with y_true/y_pred."""
    per_attr = {
        name: audit_attribute(y_true, y_pred, g, name, min_group_size, thresholds)
        for name, g in groups.items()
    }
    risk = max(a["risk"] for a in per_attr.values())
    return ModuleResult(
        name="Fairness",
        metrics={name: a["metrics"] | {"risks": a["risks"], "attribute_risk": a["risk"]} for name, a in per_attr.items()},
        risk=risk,
        status=status_from_risk(risk),
        findings=[f for a in per_attr.values() for f in a["findings"]],
        details={name: a["table"] for name, a in per_attr.items()},
    )
