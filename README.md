# Governance Auditing Framework for AI-Based Social Welfare Eligibility Decision Systems

M.Tech Data Science final year project (Phase 1). Implements a multi-dimensional
governance audit pipeline for an AI-based welfare eligibility classifier:

1. Data Quality
2. Fairness
3. Explainability
4. Prediction Reliability
5. Robustness
6. Compliance & Accountability
7. Overall Governance Risk Score (GRS)

## Phase 1 dataset

Folktables ACS Income (US Census PUMS) — used for technical validation of the
audit modules before Phase 2 validation on MGNREGA/PLFS-inspired synthetic data.

## Project layout

```
data/              raw and processed datasets
src/
  data_pipeline.py     dataset loading + preprocessing
  model.py             baseline XGBoost eligibility model
  audit/
    data_quality.py
    fairness.py
    explainability.py
    reliability.py
    robustness.py
    compliance.py
    governance_score.py
results/           audit outputs, plots, reports
notebooks/         exploratory notebooks
```

## Setup

```bash
pip install -r requirements.txt
```

## Run order

1. `notebooks/01` to `08`, in order (each saves its result to `results/`)
2. Dashboard, from the project root:

```bash
streamlit run app/dashboard.py
```

The dashboard has three tabs: a single-applicant form, a CSV upload (add a `label`
column for the full audit) and the model-level audit with the Governance Risk Score.
Phase 1 caveats: the label is an income proxy, and the human-oversight module is simulated.
