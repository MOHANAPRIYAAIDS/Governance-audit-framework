"""
Governance audit dashboard for the baseline welfare-eligibility model.

Run from the project root:   streamlit run app/dashboard.py

Tabs:
  Applicant    one person in, one decision out, with the checks that matter for that decision
  Batch CSV    many people in; a full audit if the file has a 'label' column
  Model audit  the audit of the baseline model on its held-out test set and the GRS

Phase 1 caveats shown in the app: the label is an income proxy (not real eligibility),
the oversight module is a simulation, and code labels follow the Census PUMS dictionary.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier

from src.audit.common import load_result
from src.audit.explainability import explain_row, shap_values
from src.audit.grs import DIMENSIONS, WEIGHT_SCHEMES, compute_grs, grs_sensitivity
from src.audit.oversight import DEFAULT_REVIEW_THRESHOLD
from src.audit.run_all import run_all_audits
from src.model import FEATURE_COLS, apply_dtypes, build_dtypes, load_processed

PROCESSED = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")

FIELD_NAMES = {
    "AGEP": "Age", "COW": "Class of worker", "SCHL": "Education", "MAR": "Marital status",
    "OCCP": "Occupation code", "POBP": "Place of birth code", "RELSHIPP": "Relationship to householder",
    "WKHP": "Hours worked per week", "SEX": "Sex", "RAC1P": "Race",
}
COW = {1: "Employee of private for-profit", 2: "Employee of private not-for-profit",
       3: "Local government employee", 4: "State government employee", 5: "Federal government employee",
       6: "Self-employed, not incorporated", 7: "Self-employed, incorporated",
       8: "Working without pay in family business"}
MAR = {1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never married or under 15"}
SEX = {1: "Male", 2: "Female"}
RAC1P = {1: "White alone", 2: "Black or African American alone", 3: "American Indian alone",
         4: "Alaska Native alone", 5: "American Indian and Alaska Native tribes",
         6: "Asian alone", 7: "Native Hawaiian and Other Pacific Islander alone",
         8: "Some other race alone", 9: "Two or more races"}
SCHL = {1: "No schooling", 2: "Nursery or preschool", 3: "Kindergarten", 15: "Grade 12, no diploma",
        16: "Regular high school diploma", 17: "GED or alternative credential",
        18: "Some college, less than 1 year", 19: "1 or more years of college, no degree",
        20: "Associate's degree", 21: "Bachelor's degree", 22: "Master's degree",
        23: "Professional degree", 24: "Doctorate"}
SCHL.update({c: f"Grade {c - 3}" for c in range(4, 15)})
RELSHIPP = {20: "Reference person", 21: "Opposite-sex spouse", 22: "Opposite-sex unmarried partner",
            23: "Same-sex spouse", 24: "Same-sex unmarried partner", 25: "Biological son or daughter",
            26: "Adopted son or daughter", 27: "Stepson or stepdaughter", 28: "Brother or sister",
            29: "Father or mother", 30: "Grandchild", 31: "Parent-in-law", 32: "Son/daughter-in-law",
            33: "Other relative", 34: "Roommate or housemate", 35: "Foster child",
            36: "Other nonrelative", 37: "Institutionalized group quarters",
            38: "Non-institutionalized group quarters"}
OPTION_LABELS = {"COW": COW, "SCHL": SCHL, "MAR": MAR, "RELSHIPP": RELSHIPP, "SEX": SEX, "RAC1P": RAC1P}
STATUS_ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}

st.set_page_config(page_title="Welfare AI Governance Audit", layout="wide")


@st.cache_resource(show_spinner="Loading model and data...")
def load_assets():
    X_train, X_test, y_train, y_test, _, _ = load_processed(PROCESSED)
    dtypes = build_dtypes(X_train, X_test)
    model = XGBClassifier()
    model.load_model(os.path.join(RESULTS, "baseline_xgb.json"))
    return model, dtypes, X_train, apply_dtypes(X_test, dtypes), y_test.reset_index(drop=True)


@st.cache_resource(show_spinner="Auditing the model on the test set (about a minute the first time)...")
def test_set_audit():
    model, dtypes, X_train, Xte, yte = load_assets()
    pre = {}
    dq = os.path.join(RESULTS, "data_quality_result.json")
    raw = None
    if os.path.exists(dq):
        pre["data_quality"] = load_result(dq)
    else:
        raw = Xte.astype(float)
    results, grs, p = run_all_audits(model, Xte, y_true=yte.to_numpy(), raw_df=raw, precomputed=pre)
    return results, p


def sidebar_settings():
    st.sidebar.header("Settings")
    thr = st.sidebar.slider("Decision threshold", 0.30, 0.70, 0.50, 0.01)
    review = st.sidebar.slider("Send to human review if confidence is below", 0.50, 0.95,
                               float(DEFAULT_REVIEW_THRESHOLD), 0.05)
    st.sidebar.subheader("GRS weights")
    st.sidebar.caption("Relative importance of each dimension. They are rescaled to sum to 1.")
    weights = {k: st.sidebar.slider(v, 0.0, 1.0, 1.0, 0.05, key=f"w_{k}") for k, v in DIMENSIONS.items()}
    st.sidebar.caption("Model-level audit numbers use a 0.5 decision threshold and a 0.70 review threshold.")
    return thr, review, weights


def show_grs(results, weights):
    grs = compute_grs(results, weights if sum(weights.values()) > 0 else None)
    c1, c2, c3 = st.columns(3)
    c1.metric("Governance Risk Score", f"{grs.score:.2f}")
    c2.metric("Risk band", grs.band)
    c3.metric("Worst dimension", grs.worst_dimension)
    if grs.failing:
        st.error("Failing dimensions: " + ", ".join(grs.failing) + ". The score is an average and can hide these.")
    if grs.not_assessed:
        st.info("Not assessed: " + ", ".join(grs.not_assessed))
    prof = grs.profile[["dimension", "risk", "status", "weight"]].copy()
    prof["status"] = prof["status"].map(STATUS_ICON)
    st.dataframe(prof.set_index("dimension").round(3), width="stretch")
    st.bar_chart(grs.profile.set_index("dimension")["risk"])
    return grs


def module_details(results):
    for key, r in results.items():
        if r is None:
            continue
        with st.expander(f"{DIMENSIONS[key]}: risk {r.risk:.2f} ({STATUS_ICON[r.status]})"):
            for f in r.findings:
                st.write("- " + f)
            d = r.details
            if key == "fairness":
                for attr, t in d.items():
                    st.caption(f"Group table: {attr}")
                    st.dataframe(t.round(3), width="stretch")
            elif key == "explainability":
                st.caption("Share of total SHAP importance (log-odds scale)")
                st.bar_chart(d["importance"]["share"])
            elif key == "reliability":
                st.caption("Calibration: predicted probability vs observed share eligible")
                st.line_chart(d["curve"].set_index("mean_predicted")["observed"])
                st.dataframe(d["ece_table"].round(3), width="stretch")
            elif key == "robustness":
                st.dataframe(d["grid"].round(4), width="stretch")
            elif key == "oversight":
                st.caption("SIMULATED: review rate against errors caught")
                st.dataframe(d["curve"].round(3), width="stretch")


# ------------------------------------------------------------------ applicant tab
def applicant_form(X_train):
    codes = {c: X_train[c].astype(int).value_counts().index.tolist() for c in ("OCCP", "POBP")}
    with st.form("applicant"):
        cols = st.columns(3)
        row = {}
        row["AGEP"] = cols[0].number_input(FIELD_NAMES["AGEP"], 17, 95, 35)
        row["WKHP"] = cols[1].number_input(FIELD_NAMES["WKHP"], 1, 99, 40)
        for i, col in enumerate(("COW", "SCHL", "MAR", "RELSHIPP", "SEX", "RAC1P")):
            opts = OPTION_LABELS[col]
            keys = list(opts)
            default = {"COW": 1, "SCHL": 21, "MAR": 1, "RELSHIPP": 20, "SEX": 1, "RAC1P": 1}[col]
            row[col] = cols[(i + 2) % 3].selectbox(
                FIELD_NAMES[col], keys, index=keys.index(default), format_func=lambda c, o=opts: f"{o[c]} ({c})")
        row["OCCP"] = cols[2].selectbox(FIELD_NAMES["OCCP"], codes["OCCP"], help="Census occupation code (OCCP)")
        row["POBP"] = cols[0].selectbox(FIELD_NAMES["POBP"], codes["POBP"], help="Census place-of-birth code (POBP)")
        submitted = st.form_submit_button("Assess applicant")
    return row, submitted


def stability_check(model, dtypes, row, thr, n=40, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame([row] * n)
    df["AGEP"] = (df["AGEP"] + rng.integers(-2, 3, n)).clip(17, 95)
    df["WKHP"] = (df["WKHP"] + rng.integers(-5, 6, n)).clip(1, 99)
    pred = model.predict_proba(apply_dtypes(df, dtypes))[:, 1] >= thr
    base = model.predict_proba(apply_dtypes(pd.DataFrame([row]), dtypes))[0, 1] >= thr
    return float((pred == base).mean())


def counterfactual_table(model, dtypes, row, thr):
    base_p = model.predict_proba(apply_dtypes(pd.DataFrame([row]), dtypes))[0, 1]
    rows = []
    for attr, opts in (("SEX", SEX), ("RAC1P", RAC1P)):
        for code, name in opts.items():
            if code == row[attr]:
                continue
            alt = {**row, attr: code}
            p = model.predict_proba(apply_dtypes(pd.DataFrame([alt]), dtypes))[0, 1]
            rows.append({"change": f"{FIELD_NAMES[attr]} to {name}", "probability": p,
                         "decision": "Eligible" if p >= thr else "Not eligible",
                         "decision_changes": (p >= thr) != (base_p >= thr)})
    return pd.DataFrame(rows)


def applicant_tab(thr, review, weights):
    model, dtypes, X_train, _, _ = load_assets()
    st.write("Enter an applicant using the same fields the model was trained on.")
    row, submitted = applicant_form(X_train)
    if not submitted:
        return
    Xrow = apply_dtypes(pd.DataFrame([row]), dtypes)
    p = float(model.predict_proba(Xrow)[0, 1])
    eligible = p >= thr
    conf = max(p, 1 - p)

    st.subheader("This applicant's decision")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted", "Eligible" if eligible else "Not eligible")
    c2.metric("Probability eligible", f"{p:.1%}")
    c3.metric("Confidence", f"{conf:.1%}")
    c4.metric("Human review", "Required" if conf < review else "Not required")

    sh, _ = shap_values(model, Xrow)
    exp = explain_row(sh, Xrow, Xrow.index[0], top=8)
    exp.index = [FIELD_NAMES[i] for i in exp.index]
    st.caption("What pushed this decision (log-odds; positive pushes towards eligible)")
    st.bar_chart(exp["contribution"])
    st.dataframe(exp.round(3), width="stretch")

    stab = stability_check(model, dtypes, row, thr)
    cf = counterfactual_table(model, dtypes, row, thr)
    changed = cf[cf.decision_changes]
    s1, s2 = st.columns(2)
    s1.metric("Stability", f"{stab:.0%} of nearby inputs give the same decision",
              help="Age changed by up to 2 years and weekly hours by up to 5.")
    s2.metric("Decisions that change if only sex or race changes", f"{len(changed)} of {len(cf)}")
    with st.expander("Counterfactual detail"):
        st.dataframe(cf.round(3), width="stretch")
        st.caption("Race group 4 has almost no training data, so its row is unreliable.")

    flags = []
    if conf < review:
        flags.append("confidence is below the review threshold")
    if stab < 0.9:
        flags.append("the decision is unstable for small input changes")
    if len(changed):
        flags.append("the decision changes if only sex or race is changed")
    if flags:
        st.warning("Recommended handling: refer to a human reviewer because " + "; ".join(flags) + ".")
    else:
        st.success("Recommended handling: no decision-level flags. Automated decision is acceptable under these checks.")

    st.subheader("Model-level audit context")
    results, _ = test_set_audit()
    show_grs(results, weights)


# ------------------------------------------------------------------ batch tab
def batch_tab(thr, review, weights):
    model, dtypes, _, _, _ = load_assets()
    st.write("Upload a CSV with these columns: " + ", ".join(FEATURE_COLS)
             + ". Add a `label` column (1 = eligible, 0 = not) to run the full audit.")
    up = st.file_uploader("CSV file", type="csv")
    if up is None:
        return
    df = pd.read_csv(up)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing)
                 + (". Note: use RELSHIPP (2019 and later), not RELP." if "RELSHIPP" in missing else ""))
        return
    X = apply_dtypes(df[FEATURE_COLS], dtypes)
    y = df["label"].astype(int).to_numpy() if "label" in df.columns else None
    with st.spinner("Auditing..."):
        results, _, p = run_all_audits(model, X, y_true=y, raw_df=df[FEATURE_COLS],
                                       review_threshold=review, decision_threshold=thr)
    out = df.copy()
    out["probability_eligible"] = p
    out["decision"] = np.where(p >= thr, "Eligible", "Not eligible")
    out["confidence"] = np.maximum(p, 1 - p)
    out["human_review"] = out["confidence"] < review
    st.subheader(f"{len(out):,} applicants")
    c1, c2 = st.columns(2)
    c1.metric("Predicted eligible", f"{(p >= thr).mean():.1%}")
    c2.metric("Flagged for human review", f"{out['human_review'].mean():.1%}")
    st.dataframe(out.head(200), width="stretch")
    st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")
    if y is None:
        st.info("No `label` column: fairness, reliability and human-oversight audits need true outcomes and were skipped.")
    st.subheader("Audit of this file")
    show_grs(results, weights)
    module_details(results)


# ------------------------------------------------------------------ model audit tab
def model_tab(weights):
    results, _ = test_set_audit()
    st.write("Audit of the baseline XGBoost model on its held-out test set (38,642 people).")
    st.caption("Phase 1 caveats: the label is income above $50,000, a stand-in for eligibility; "
               "the human-oversight result is a simulation.")
    show_grs(results, weights)
    st.subheader("Does the score depend on the weights?")
    st.dataframe(grs_sensitivity(results, WEIGHT_SCHEMES).round(3), width="stretch")
    module_details(results)


def main():
    st.title("Welfare eligibility AI: governance audit")
    thr, review, weights = sidebar_settings()
    t1, t2, t3 = st.tabs(["Applicant", "Batch CSV", "Model audit"])
    with t1:
        applicant_tab(thr, review, weights)
    with t2:
        batch_tab(thr, review, weights)
    with t3:
        model_tab(weights)


main()
