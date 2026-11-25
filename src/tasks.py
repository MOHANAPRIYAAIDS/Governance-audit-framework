"""
Task definitions: which columns are numeric or categorical, which attributes the
fairness module slices by, and which features count as sensitive.

The audit modules default to INCOME, so the dashboard and notebooks 01 to 08 are
unchanged. PUBLIC_COVERAGE is the second Phase 1 task (notebook 09).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TaskSpec:
    name: str
    numeric_cols: tuple
    categorical_cols: tuple
    valid_range: dict = field(default_factory=dict)   # numeric column -> (low, high) after noise
    protected: tuple = ()                             # fairness module slices by these
    sensitive: tuple = ()                             # explainability sensitive-feature set
    flip_features: tuple = ()                         # counterfactual check changes these

    @property
    def feature_cols(self) -> list:
        return list(self.numeric_cols) + list(self.categorical_cols)


INCOME = TaskSpec(
    name="income",
    numeric_cols=("AGEP", "WKHP"),
    categorical_cols=("COW", "SCHL", "MAR", "OCCP", "POBP", "RELSHIPP", "SEX", "RAC1P"),
    valid_range={"AGEP": (17, 95), "WKHP": (1, 99)},
    protected=("RAC1P", "SEX"),
    sensitive=("RAC1P", "SEX", "POBP"),
    flip_features=("SEX", "RAC1P"),
)

PUBLIC_COVERAGE = TaskSpec(
    name="public_coverage",
    numeric_cols=("AGEP", "PINCP"),
    categorical_cols=("SCHL", "MAR", "SEX", "DIS", "ESP", "CIT", "MIG", "MIL", "ANC",
                      "NATIVITY", "DEAR", "DEYE", "DREM", "ESR", "ST", "FER", "RAC1P"),
    valid_range={"AGEP": (0, 95), "PINCP": (-20000, 30000)},
    protected=("RAC1P", "SEX", "DIS"),
    sensitive=("RAC1P", "SEX", "DIS", "CIT", "NATIVITY"),
    flip_features=("SEX", "RAC1P", "DIS"),
)
