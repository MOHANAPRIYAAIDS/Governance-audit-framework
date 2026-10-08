"""
Compliance, accountability and human-oversight module (survey section 4.5).

IMPORTANT: in Phase 1 this module is a SIMULATION. Folktables has no policy rules,
review records or decision logs, so:

  - Human oversight is simulated with a stated policy: any decision whose confidence
    (max(p, 1 - p)) is below `review_threshold` is sent to a human reviewer.
      Human Review Rate         = share of decisions reviewed
      High-Risk Escalation Rate = share of the model's ERRORS that were sent to review
      Residual error rate       = share of all decisions that are wrong AND not reviewed
  - Compliance Rate and Audit Traceability Rate need real inputs (policy rules and a
    decision log). They are computed only when supplied; otherwise they are reported as
    "not assessed" and do not enter the risk. Phase 2 (synthetic welfare data) is where
    those inputs will exist.

Thresholds are documented defaults to be sensitivity-tested.
"""

import numpy as np
import pandas as pd

from .common import ModuleResult, risk_from_violation, status_from_risk

DEFAULT_REVIEW_THRESHOLD = 0.70
DEFAULT_THRESHOLDS = {"residual_error_rate": 0.10, "compliance_gap": 0.05, "traceability_gap": 0.05}


def confidence(p, decision_threshold=0.5):
    p = np.asarray(p, float)
    return np.where(p >= decision_threshold, p, 1 - p)


def oversight_curve(y_true, p, review_thresholds=None, decision_threshold=0.5) -> pd.DataFrame:
    """Review rate, errors caught and residual error as the review threshold moves."""
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(p, float) >= decision_threshold).astype(int)
    conf, err = confidence(p, decision_threshold), pred != y
    rows = []
    for t in (review_thresholds if review_thresholds is not None else np.round(np.arange(0.5, 1.0001, 0.05), 2)):
        reviewed = conf < t
        rows.append({
            "review_threshold": float(t),
            "review_rate": float(reviewed.mean()),
            "errors_caught": float((reviewed & err).sum() / max(err.sum(), 1)),
            "residual_error_rate": float((err & ~reviewed).mean()),
        })
    return pd.DataFrame(rows)


def audit_oversight(
    y_true, p, review_threshold=DEFAULT_REVIEW_THRESHOLD, decision_threshold=0.5,
    X=None, policy_rules=None, decision_log=None, required_fields=None, thresholds=None,
) -> ModuleResult:
    """
    policy_rules: list of (name, fn(X, y_pred) -> bool array, True = compliant), optional.
    decision_log + required_fields: a DataFrame and the columns that must be filled, optional.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    y = np.asarray(y_true).astype(int)
    pred = (np.asarray(p, float) >= decision_threshold).astype(int)
    conf, err = confidence(p, decision_threshold), pred != y
    reviewed = conf < review_threshold

    review_rate = float(reviewed.mean())
    catch = float((reviewed & err).sum() / max(err.sum(), 1))
    residual = float((err & ~reviewed).mean())

    risks = {"residual_error_rate": risk_from_violation(residual, thr["residual_error_rate"])}
    metrics = {
        "simulated": True, "review_threshold": review_threshold,
        "human_review_rate": review_rate, "high_risk_escalation_rate": catch,
        "residual_unreviewed_error_rate": residual,
        "compliance_rate": None, "audit_traceability_rate": None,
    }
    findings = [
        f"SIMULATED policy: decisions with confidence below {review_threshold:.0%} go to a human. "
        f"That reviews {review_rate:.1%} of decisions and catches {catch:.1%} of the model's errors; "
        f"{residual:.1%} of all decisions are wrong and unreviewed."
    ]
    if residual > thr["residual_error_rate"]:
        findings.append(f"Unreviewed error rate {residual:.1%} exceeds {thr['residual_error_rate']:.0%}: raise the review threshold or improve the model.")

    not_assessed = []
    if policy_rules and X is not None:
        ok = np.ones(len(pred), bool)
        for _, fn in policy_rules:
            ok &= np.asarray(fn(X, pred), bool)
        metrics["compliance_rate"] = float(ok.mean())
        risks["compliance"] = risk_from_violation(1 - metrics["compliance_rate"], thr["compliance_gap"])
        findings.append(f"Policy compliance rate {metrics['compliance_rate']:.1%}.")
    else:
        not_assessed.append("compliance rate (no policy rules supplied)")
    if decision_log is not None and required_fields:
        metrics["audit_traceability_rate"] = float(decision_log[required_fields].notna().all(axis=1).mean())
        risks["traceability"] = risk_from_violation(1 - metrics["audit_traceability_rate"], thr["traceability_gap"])
        findings.append(f"Audit traceability rate {metrics['audit_traceability_rate']:.1%}.")
    else:
        not_assessed.append("audit traceability rate (no decision log supplied)")
    if not_assessed:
        findings.append("Not assessed in Phase 1: " + "; ".join(not_assessed) + ".")

    metrics["risks"] = risks
    risk = max(risks.values())
    return ModuleResult(
        name="Human oversight",
        metrics=metrics, risk=risk, status=status_from_risk(risk), findings=findings,
        details={"curve": oversight_curve(y, p, decision_threshold=decision_threshold), "simulated": True},
    )
