"""
Data Quality audit module (survey section 4.1).

Implements the three data-quality metrics defined in the paper:
  - Missing Value Rate = missing observations / total observations
  - Duplicate Rate      = duplicate records / total records
  - Data Completeness   = 1 - (total missing values / total observations)

This module only MEASURES data quality. It must be run on the RAW features
(before src/data_pipeline.clean_features drops nulls/duplicates) — auditing
already-cleaned data would hide exactly the issues the module exists to find.
"""

from dataclasses import dataclass, field

import pandas as pd

from .common import ModuleResult, risk_from_violation, status_from_risk


@dataclass
class DataQualityReport:
    n_rows: int
    n_columns: int
    missing_value_rate_overall: float
    missing_value_rate_by_column: pd.Series
    duplicate_rate: float
    completeness: float
    risk_score: float
    details: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Rows x Columns:            {self.n_rows} x {self.n_columns}",
            f"Missing Value Rate (overall): {self.missing_value_rate_overall:.4f}",
            f"Duplicate Rate:               {self.duplicate_rate:.4f}",
            f"Data Completeness:            {self.completeness:.4f}",
            f"Data Quality Risk Score:      {self.risk_score:.4f}  (0 = clean, 1 = worst)",
            "",
            "Missing Value Rate by column (non-zero only):",
        ]
        nonzero = self.missing_value_rate_by_column[self.missing_value_rate_by_column > 0]
        if nonzero.empty:
            lines.append("  (none)")
        else:
            for col, rate in nonzero.sort_values(ascending=False).items():
                lines.append(f"  {col:<20s} {rate:.4f}")
        return "\n".join(lines)


def missing_value_rate(df: pd.DataFrame) -> tuple[float, pd.Series]:
    """Overall rate = total missing cells / total cells. Per-column rate for diagnosis."""
    total_cells = df.shape[0] * df.shape[1]
    missing_by_col = df.isna().sum()
    overall = missing_by_col.sum() / total_cells if total_cells else 0.0
    rate_by_col = missing_by_col / df.shape[0] if df.shape[0] else missing_by_col * 0.0
    return overall, rate_by_col


def duplicate_rate(df: pd.DataFrame) -> float:
    """Proportion of rows that are exact duplicates of another row."""
    if df.shape[0] == 0:
        return 0.0
    n_duplicates = df.duplicated(keep="first").sum()
    return n_duplicates / df.shape[0]


def completeness(df: pd.DataFrame) -> float:
    """1 - (total missing values / total observations). Closer to 1 is better."""
    total_cells = df.shape[0] * df.shape[1]
    if total_cells == 0:
        return 1.0
    total_missing = df.isna().sum().sum()
    return 1.0 - (total_missing / total_cells)


def data_quality_risk_score(missing_rate: float, dup_rate: float) -> float:
    """
    Simple normalized risk score in [0, 1] combining the two "badness" signals
    (completeness is just 1 - missing_rate, so it isn't counted twice).
    Equal-weighted average — document/justify this choice in your report,
    and treat it as a candidate default rather than a fixed rule.
    """
    return (missing_rate + dup_rate) / 2.0


def audit_data_quality(df: pd.DataFrame) -> DataQualityReport:
    """Run the full data-quality audit on a raw (uncleaned) feature dataframe."""
    overall_missing, by_col = missing_value_rate(df)
    dup_rate = duplicate_rate(df)
    complete = completeness(df)
    risk = data_quality_risk_score(overall_missing, dup_rate)

    return DataQualityReport(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        missing_value_rate_overall=overall_missing,
        missing_value_rate_by_column=by_col,
        duplicate_rate=dup_rate,
        completeness=complete,
        risk_score=risk,
    )


DEFAULT_THRESHOLDS = {"missing_value_rate": 0.05, "duplicate_rate": 0.05}


def audit_data_quality_result(df: pd.DataFrame, thresholds=None) -> ModuleResult:
    """
    The same three metrics as audit_data_quality, in the shared ModuleResult format
    so the Governance Risk Score can use them. Risk is the worse of the missing-value
    and duplicate risks (0.5 at the threshold). Run it on RAW data.
    """
    thr = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    rep = audit_data_quality(df)
    risks = {
        "missing_value_rate": risk_from_violation(rep.missing_value_rate_overall, thr["missing_value_rate"]),
        "duplicate_rate": risk_from_violation(rep.duplicate_rate, thr["duplicate_rate"]),
    }
    risk = max(risks.values())
    findings = [
        f"{rep.n_rows:,} rows audited: missing value rate {rep.missing_value_rate_overall:.2%}, "
        f"duplicate rate {rep.duplicate_rate:.2%}, completeness {rep.completeness:.2%}."
    ]
    worst = rep.missing_value_rate_by_column.sort_values(ascending=False)
    if worst.iloc[0] > 0:
        findings.append(f"Most missing column: {worst.index[0]} ({worst.iloc[0]:.2%}).")
    return ModuleResult(
        name="Data quality",
        metrics={
            "n_rows": rep.n_rows,
            "missing_value_rate": rep.missing_value_rate_overall,
            "duplicate_rate": rep.duplicate_rate,
            "completeness": rep.completeness,
            "risks": risks,
        },
        risk=risk,
        status=status_from_risk(risk),
        findings=findings,
        details={"report": rep},
    )


if __name__ == "__main__":
    from src.data_pipeline import build_income_task, load_acs_income

    acs_data = load_acs_income()
    features, label, group = build_income_task(acs_data)

    report = audit_data_quality(features)
    print(report.summary())
