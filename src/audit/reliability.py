"""
Prediction reliability audit module (survey section 4.4).

  - Brier Score = mean((p - y)^2)                          lower is better
  - ECE         = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|
    where confidence = max(p, 1 - p) and the predictions are split into equal-width
    confidence bins (the standard definition from Guo et al. 2017).

Also reported: calibration per protected group (a model can be well calibrated overall
and poorly calibrated for one group), and the share of low-confidence predictions,
which is what the oversight module sends to a human.

Thresholds are documented defaults to be sensitivity-tested.
"""

import numpy as np
import pandas as pd

from .common import ModuleResult, risk_from_violation, status_from_risk

# ECE over 10 bins is biased upward on small samples, so per-group calibration needs far
# more rows than a simple rate comparison does. 500 keeps roughly 50 rows per bin.
MIN_GROUP_SIZE = 500
LOW_CONFIDENCE = 0.70

DEFAULT_THRESHOLDS = {
    "ece": 0.05,
    "brier": 0.20,  # a no-information model scores about 0.25 on a balanced target
    "group_ece_gap": 0.05,
}


def brier_score(y_true, p) -> float:
    y, p = np.asarray(y_true, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def ece_table(y_true, p, n_bins=10, decision_threshold=0.5) -> pd.DataFrame:
    y, p = np.asarray(y_true).astype(int), np.asarray(p, float)
    pred = (p >= decision_threshold).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        n = int(m.sum())
        rows.append({
            "bin_low": b / n_bins, "bin_high": (b + 1) / n_bins, "n": n,
            "confidence": conf[m].mean() if n else np.nan,
            "accuracy": correct[m].mean() if n else np.nan,
        })
    t = pd.DataFrame(rows)
    t["gap"] = (t["accuracy"] - t["confidence"]).abs()
    return t


def expected_calibration_error(y_true, p, n_bins=10, decision_threshold=0.5) -> float:
    t = ece_table(y_true, p, n_bins, decision_threshold)
    total = t["n"].sum()
    return float((t["n"] / total * t["gap"].fillna(0)).sum()) if total else 0.0


def reliability_curve(y_true, p, n_bins=10) -> pd.DataFrame:
    """Predicted probability of 'eligible' against the observed share eligible, per bin."""
    y, p = np.asarray(y_true, float), np.asarray(p, float)
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            rows.append({"mean_predicted": p[m].mean(), "observed": y[m].mean(), "n": int(m.sum())})
    return pd.DataFrame(rows)


def audit_reliability(
    y_true, p, groups: dict = None, n_bins=10, thresholds=None,
    min_group_size=MIN_GROUP_SIZE, decision_threshold=0.5,
) -> ModuleResult:
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    y, p = np.asarray(y_true).astype(int), np.asarray(p, float)
    brier = brier_score(y, p)
    ece = expected_calibration_error(y, p, n_bins, decision_threshold)
    pred = (p >= decision_threshold).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    low_share = float((conf < LOW_CONFIDENCE).mean())

    risks = {
        "ece": risk_from_violation(ece, thr["ece"]),
        "brier": risk_from_violation(brier, thr["brier"]),
    }
    findings = [
        f"ECE {ece:.3f}, Brier score {brier:.3f}; {low_share:.1%} of predictions have "
        f"confidence below {LOW_CONFIDENCE:.0%}."
    ]
    if ece > thr["ece"]:
        findings.append(f"ECE {ece:.3f} exceeds {thr['ece']:.2f}: predicted confidence does not match accuracy.")
    if brier > thr["brier"]:
        findings.append(f"Brier score {brier:.3f} exceeds {thr['brier']:.2f}.")

    group_metrics, group_tables = {}, {}
    for attr, g in (groups or {}).items():
        g = np.asarray(g)
        rows = []
        for val in pd.unique(g[~pd.isna(g)]):
            m = g == val
            if m.sum() >= min_group_size:
                rows.append({"group": val, "n": int(m.sum()),
                             "ece": expected_calibration_error(y[m], p[m], n_bins, decision_threshold),
                             "brier": brier_score(y[m], p[m])})
        if len(rows) >= 2:
            t = pd.DataFrame(rows).set_index("group")
            gap = float(t["ece"].max() - t["ece"].min())
            risks[f"group_ece_gap_{attr}"] = risk_from_violation(gap, thr["group_ece_gap"])
            group_metrics[attr] = {"gap": gap, "groups": {str(k): {"n": int(r.n), "ece": float(r.ece)} for k, r in t.iterrows()}}
            group_tables[attr] = t
            if gap > thr["group_ece_gap"]:
                findings.append(
                    f"{attr}: calibration error differs by {gap:.3f} across groups "
                    f"(limit {thr['group_ece_gap']:.2f}); confidence is less trustworthy for some groups."
                )

    risk = max(risks.values())
    return ModuleResult(
        name="Reliability",
        metrics={"brier": brier, "ece": ece, "accuracy": float((pred == y).mean()),
                 "low_confidence_share": low_share, "group_ece": group_metrics, "risks": risks},
        risk=risk,
        status=status_from_risk(risk),
        findings=findings,
        details={"ece_table": ece_table(y, p, n_bins, decision_threshold),
                 "curve": reliability_curve(y, p, n_bins), "group_tables": group_tables},
    )
