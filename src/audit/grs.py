"""
Governance Risk Score (survey section 5.6).

  GRS = sum_i w_i * r_i,   with each r_i in [0, 1] and the weights summing to 1.

Only dimensions that were actually assessed take part; the weights of the assessed
dimensions are rescaled to sum to 1, and the dimensions left out are listed.
The average can hide one bad dimension, so the profile and the list of failing
dimensions are always returned next to the score.

The weights are an assumption, not a finding. Equal weights are the default and
grs_sensitivity() shows the score under other schemes.
"""

from dataclasses import dataclass

import pandas as pd

DIMENSIONS = {
    "data_quality": "Data quality",
    "fairness": "Fairness",
    "explainability": "Explainability",
    "reliability": "Reliability",
    "robustness": "Robustness",
    "oversight": "Human oversight",
}

EQUAL_WEIGHTS = {d: 1 / len(DIMENSIONS) for d in DIMENSIONS}

WEIGHT_SCHEMES = {
    "equal": EQUAL_WEIGHTS,
    "fairness_heavy": {"data_quality": 0.14, "fairness": 0.30, "explainability": 0.14,
                       "reliability": 0.14, "robustness": 0.14, "oversight": 0.14},
    "technical_heavy": {"data_quality": 0.20, "fairness": 0.10, "explainability": 0.10,
                        "reliability": 0.20, "robustness": 0.20, "oversight": 0.20},
    "governance_heavy": {"data_quality": 0.10, "fairness": 0.25, "explainability": 0.20,
                         "reliability": 0.10, "robustness": 0.10, "oversight": 0.25},
}


def band(score: float) -> str:
    if score < 0.25:
        return "Low"
    if score < 0.50:
        return "Moderate"
    if score < 0.75:
        return "High"
    return "Critical"


@dataclass
class GRSResult:
    score: float
    band: str
    profile: pd.DataFrame
    not_assessed: list
    failing: list
    worst_dimension: str


def compute_grs(results: dict, weights: dict = None) -> GRSResult:
    """results maps a dimension key to a ModuleResult, or None if it was not assessed."""
    w = {**EQUAL_WEIGHTS, **(weights or {})}
    assessed = [d for d in DIMENSIONS if results.get(d) is not None]
    if not assessed:
        raise ValueError("no dimension was assessed")
    total = sum(w[d] for d in assessed)
    if total <= 0:
        raise ValueError("weights of the assessed dimensions sum to zero")

    rows = []
    for d in assessed:
        wn = w[d] / total
        r = results[d]
        rows.append({"dimension": DIMENSIONS[d], "key": d, "risk": r.risk, "status": r.status,
                     "weight": wn, "contribution": wn * r.risk})
    profile = pd.DataFrame(rows).set_index("key")
    score = float(profile["contribution"].sum())
    return GRSResult(
        score=score,
        band=band(score),
        profile=profile,
        not_assessed=[DIMENSIONS[d] for d in DIMENSIONS if d not in assessed],
        failing=[DIMENSIONS[d] for d in assessed if results[d].status == "fail"],
        worst_dimension=DIMENSIONS[profile["risk"].idxmax()],
    )


def grs_sensitivity(results: dict, schemes: dict = None) -> pd.DataFrame:
    rows = []
    for name, w in (schemes or WEIGHT_SCHEMES).items():
        g = compute_grs(results, w)
        rows.append({"weight_scheme": name, "grs": g.score, "band": g.band})
    return pd.DataFrame(rows).set_index("weight_scheme")
