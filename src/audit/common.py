"""
Shared output format for every audit module.

Each module (data quality, fairness, explainability, reliability, robustness,
oversight) returns a ModuleResult so the Governance Risk Score and the
dashboard can consume them the same way.
"""

import json
from dataclasses import dataclass, field

PASS, WARN, FAIL = "pass", "warn", "fail"


def risk_from_violation(value: float, threshold: float) -> float:
    """
    Map a "bad when larger" metric to a risk in [0, 1].
    0 at no violation, 0.5 exactly at the threshold, 1 at twice the threshold.
    Thresholds are documented defaults, not validated limits.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    return float(min(max(value / (2 * threshold), 0.0), 1.0))


def status_from_risk(risk: float, warn_at: float = 0.5, fail_at: float = 0.75) -> str:
    if risk >= fail_at:
        return FAIL
    if risk >= warn_at:
        return WARN
    return PASS


@dataclass
class ModuleResult:
    name: str
    metrics: dict
    risk: float
    status: str
    findings: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"{self.name}: risk {self.risk:.3f} ({self.status})"]
        lines += [f"  - {f}" for f in self.findings]
        return "\n".join(lines)


def _jsonable(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def save_result(result: ModuleResult, path: str, extra: dict = None) -> None:
    """Write name, risk, status, findings and metrics (not the heavy details) as JSON."""
    out = {
        "name": result.name,
        "risk": result.risk,
        "status": result.status,
        "findings": result.findings,
        "metrics": result.metrics,
        **(extra or {}),
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=_jsonable)


def load_result(path: str) -> ModuleResult:
    with open(path) as f:
        d = json.load(f)
    return ModuleResult(
        name=d["name"], metrics=d.get("metrics", {}), risk=d["risk"],
        status=d["status"], findings=d.get("findings", []),
    )
