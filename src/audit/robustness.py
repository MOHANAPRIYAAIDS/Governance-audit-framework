"""
Robustness audit module (survey section 4.4).

  - Performance Degradation   PD  = M_original - M_perturbed
  - Relative degradation      RPD = PD / M_original
with M = accuracy (F1 is reported too). Three controlled perturbations are applied to the
model's input features:
  noise    Gaussian noise on AGEP and WKHP, scaled to a fraction of each column's spread
  missing  random cells set to missing
  swap     random categorical cells replaced by another applicant's value

The risk uses the "moderate" level of each perturbation. Without true labels (e.g. an
unlabelled upload) the share of decisions that change is used instead.
Thresholds are documented defaults to be sensitivity-tested.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ..model import CATEGORICAL_COLS, NUMERIC_COLS
from .common import ModuleResult, risk_from_violation, status_from_risk

VALID_RANGE = {"AGEP": (17, 95), "WKHP": (1, 99)}
LEVELS = {"noise": [0.1, 0.2, 0.5], "missing": [0.05, 0.10, 0.20], "swap": [0.02, 0.05, 0.10]}
MODERATE = {"noise": 0.2, "missing": 0.10, "swap": 0.05}
DESCRIPTIONS = {
    "noise": "noise on age and hours (fraction of spread)",
    "missing": "fraction of input values missing",
    "swap": "fraction of categorical values replaced",
}
DEFAULT_THRESHOLDS = {"relative_degradation": 0.05, "flip_rate": 0.10}


def perturb(X: pd.DataFrame, kind: str, level: float, rng: np.random.Generator) -> pd.DataFrame:
    Xp, n = X.copy(), len(X)
    if kind == "noise":
        for c in NUMERIC_COLS:
            lo, hi = VALID_RANGE[c]
            Xp[c] = (X[c] + rng.normal(0, level * float(X[c].std()), n)).round().clip(lo, hi)
    elif kind == "missing":
        for c in X.columns:
            Xp.loc[rng.random(n) < level, c] = np.nan
    elif kind == "swap":
        for c in CATEGORICAL_COLS:
            mask = rng.random(n) < level
            if mask.any():
                Xp.loc[mask, c] = X[c].to_numpy()[rng.integers(0, n, int(mask.sum()))]
    else:
        raise ValueError(f"unknown perturbation: {kind}")
    return Xp


def audit_robustness(
    model, X: pd.DataFrame, y_true=None, levels=None, moderate=None,
    n_repeats=3, seed=42, decision_threshold=0.5, thresholds=None,
) -> ModuleResult:
    levels = levels or LEVELS
    moderate = moderate or MODERATE
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    labelled = y_true is not None
    y = np.asarray(y_true).astype(int) if labelled else None

    base = (model.predict_proba(X)[:, 1] >= decision_threshold).astype(int)
    m0_acc = accuracy_score(y, base) if labelled else np.nan
    m0_f1 = f1_score(y, base) if labelled else np.nan

    rows = []
    for kind, lvls in levels.items():
        for level in lvls:
            acc, f1s, flips = [], [], []
            for r in range(n_repeats):
                rng = np.random.default_rng(seed + 1000 * r + int(level * 1000))
                pred = (model.predict_proba(perturb(X, kind, level, rng))[:, 1] >= decision_threshold).astype(int)
                flips.append(float((pred != base).mean()))
                if labelled:
                    acc.append(accuracy_score(y, pred))
                    f1s.append(f1_score(y, pred))
            a = float(np.mean(acc)) if labelled else np.nan
            rows.append({
                "perturbation": kind, "level": level, "accuracy": a,
                "f1": float(np.mean(f1s)) if labelled else np.nan,
                "pd_accuracy": m0_acc - a if labelled else np.nan,
                "rpd_accuracy": (m0_acc - a) / m0_acc if labelled else np.nan,
                "flip_rate": float(np.mean(flips)),
            })
    grid = pd.DataFrame(rows)

    risks, findings = {}, []
    for kind in levels:
        sub = grid[grid.perturbation == kind]
        row = sub.iloc[(sub.level - moderate[kind]).abs().argmin()]
        if labelled:
            risks[kind] = risk_from_violation(row.rpd_accuracy, thr["relative_degradation"])
            findings.append(
                f"{kind} ({row.level:g}): accuracy falls {row.rpd_accuracy:.1%} relative; "
                f"{row.flip_rate:.1%} of decisions change."
            )
        else:
            risks[kind] = risk_from_violation(row.flip_rate, thr["flip_rate"])
            findings.append(f"{kind} ({row.level:g}): {row.flip_rate:.1%} of decisions change (no labels supplied).")

    risk = max(risks.values())
    return ModuleResult(
        name="Robustness",
        metrics={"baseline_accuracy": float(m0_acc), "baseline_f1": float(m0_f1), "risks": risks,
                 "moderate_levels": moderate},
        risk=risk,
        status=status_from_risk(risk),
        findings=findings,
        details={"grid": grid},
    )
