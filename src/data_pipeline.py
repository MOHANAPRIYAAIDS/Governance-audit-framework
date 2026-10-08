"""
Step 1 of the implementation plan: Dataset & Preprocessing.

Loads the Folktables ACS Income task (US Census PUMS data) and produces
a cleaned train/test split ready for the baseline eligibility model.

ACSIncome's built-in binary target (PINCP > $50,000) is used as a stand-in
"eligibility" label for Phase 1 technical validation, per the survey's
two-phase evaluation strategy (Folktables -> technical validation,
MGNREGA/PLFS-inspired synthetic data -> welfare-domain validation in Phase 2).

Census PUMS renamed the "relationship to householder" column from RELP to
RELSHIPP starting with the 2019 data dictionary (the category codes changed
too, not just the name). folktables' built-in ACSIncome task still hardcodes
RELP, so it breaks on 2019+ data with a KeyError. build_income_task() below
reconstructs the same task definition but picks the relationship column name
that matches the requested year, so both older (<=2018) and current (2019+)
extracts work.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from folktables import ACSDataSource, BasicProblem
from folktables.acs import adult_filter
from sklearn.model_selection import train_test_split

# RAC1P (recoded detailed race code) is used as the protected attribute for
# the fairness module later. SEX is available as an alternative if needed.
PROTECTED_ATTRIBUTE = "RAC1P"

# First ACS PUMS data dictionary year that uses RELSHIPP instead of RELP.
RELSHIPP_INTRODUCED_YEAR = 2019


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    group_train: pd.Series
    group_test: pd.Series


def load_acs_income(state: str = "CA", year: str = "2022", survey: str = "person") -> pd.DataFrame:
    """
    Download (or read locally cached) ACS PUMS microdata for one state/year.

    First call needs internet access; folktables caches the raw CSVs under
    its own data directory afterwards, so repeat runs are offline and fast.
    """
    data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey=survey)
    acs_data = data_source.get_data(states=[state], download=True)
    return acs_data


def _relationship_column(year: str) -> str:
    """RELSHIPP for 2019+ data dictionaries, RELP for 2018 and earlier."""
    return "RELSHIPP" if int(year) >= RELSHIPP_INTRODUCED_YEAR else "RELP"


def _make_income_task(year: str) -> BasicProblem:
    """
    Rebuild folktables' ACSIncome task definition, swapping in whichever
    relationship-to-householder column name matches the requested year.
    Everything else (target, threshold, group, row filters) is identical
    to the upstream ACSIncome task.
    """
    return BasicProblem(
        features=[
            "AGEP",
            "COW",
            "SCHL",
            "MAR",
            "OCCP",
            "POBP",
            _relationship_column(year),
            "WKHP",
            "SEX",
            "RAC1P",
        ],
        target="PINCP",
        target_transform=lambda x: x > 50000,
        group="RAC1P",
        preprocess=adult_filter,
        postprocess=lambda x: np.nan_to_num(x, -1),
    )


def build_income_task(acs_data: pd.DataFrame, year: str = "2022"):
    """
    Apply the ACSIncome task definition (feature selection, target
    thresholding, and basic row filters folktables applies internally,
    e.g. AGEP > 16, WKHP > 0, PINCP > 100).

    `year` must match the year the data was downloaded for, so the correct
    relationship-column name (RELP vs. RELSHIPP) is used.

    This installed version of folktables (0.0.12) returns `label` and
    `group` as single-column DataFrames rather than Series (see
    folktables/folktables.py: BasicProblem.df_to_pandas). They are squeezed
    to Series here so the rest of the pipeline can treat them uniformly;
    all three outputs share a common 0-based RangeIndex already, since
    folktables rebuilds every output from a bare numpy array internally.
    """
    income_task = _make_income_task(year)
    features, label, group = income_task.df_to_pandas(acs_data)
    label = label.squeeze("columns").rename("label")
    group = group.squeeze("columns").rename(PROTECTED_ATTRIBUTE)
    return features, label, group


def clean_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning step used to prepare data for model training.

    This performs the ACTION (drop duplicates, drop remaining nulls); the
    MEASUREMENT of how much was affected belongs in
    src/audit/data_quality.py so the audit numbers are computed the same
    way regardless of whether cleaning has already run.
    """
    return features.drop_duplicates().dropna()


def train_test_split_with_group(
    features: pd.DataFrame,
    label: pd.Series,
    group: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    """Stratified split on the label, keeping the protected-attribute column aligned."""
    idx_train, idx_test = train_test_split(
        features.index, test_size=test_size, random_state=random_state, stratify=label
    )
    return DatasetSplit(
        X_train=features.loc[idx_train],
        X_test=features.loc[idx_test],
        y_train=label.loc[idx_train],
        y_test=label.loc[idx_test],
        group_train=group.loc[idx_train],
        group_test=group.loc[idx_test],
    )


def prepare_dataset(state: str = "CA", year: str = "2022") -> DatasetSplit:
    """Full Step 1 pipeline: download -> task features -> clean -> split."""
    acs_data = load_acs_income(state=state, year=year)
    features, label, group = build_income_task(acs_data, year=year)

    cleaned = clean_features(features)
    label = label.loc[cleaned.index]
    group = group.loc[cleaned.index]

    return train_test_split_with_group(cleaned, label, group)


if __name__ == "__main__":
    split = prepare_dataset()
    print(f"Train shape: {split.X_train.shape}, Test shape: {split.X_test.shape}")
    print(f"Positive rate (train): {split.y_train.mean():.3f}")
    print(f"Group distribution (train):\n{split.group_train.value_counts()}")
