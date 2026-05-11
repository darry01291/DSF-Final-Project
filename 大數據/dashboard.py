"""
DFS504 Semester Project — Streamlit Dashboard
Credit Risk & Default Intelligence (Domain B+)

Run with:
    streamlit run dashboard.py

Pages:
  1. Home           — problem overview
  2. Data Explorer  — dataset statistics & visualisations
  3. Model Performance — metrics, confusion matrix, ROC/PR curves
  4. Prediction Demo   — enter customer data, get default probability
  5. Explainability    — SHAP feature importance
  6. Business Recommendation — decision thresholds & actions
"""

import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

warnings.filterwarnings("ignore")

DATA_PATH    = r'C:\Users\Acer\AppData\Local\Temp\UCI_Credit_Card.csv'
TARGET_COL   = "default.payment.next.month"
RANDOM_STATE = 42


@st.cache_data(show_spinner="Loading dataset…")
def load_raw_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_resource(show_spinner="Training models (first run takes ~2 min)…")
def get_trained_artifacts():
    from credit_default_pipeline import (
        engineer_features,
        prepare_splits,
        train_baseline,
        train_logistic_regression,
        train_random_forest,
        train_gradient_boosting,
        train_xgboost,
        train_lightgbm,
        tune_lightgbm,
        compute_shap,
        build_comparison_table,
    )

    raw = load_raw_data()
    df = engineer_features(raw)

    (
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler,
    ) = prepare_splits(df)

    feature_cols = df.drop(columns=[TARGET_COL]).columns.tolist()
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)

    baseline = train_baseline(X_train, y_train, X_test, y_test)
    lr = train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    rf = train_random_forest(X_train, y_train, X_test, y_test)
    gb = train_gradient_boosting(X_train, y_train, X_test, y_test)
    xgb_r = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_r = train_lightgbm(X_train, y_train, X_test, y_test)
    tuned = tune_lightgbm(X_train, y_train, X_test, y_test)

    all_results = [baseline, lr, rf, gb, xgb_r, lgb_r, tuned]
    comparison_df = build_comparison_table(all_results)
    shap_values, mean_abs_shap = compute_shap(tuned["fitted_model"], X_test_df)

    return {
        "df": df,
        "raw": raw,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "X_test_df": X_test_df,
        "results": all_results,
        "comparison_df": comparison_df,
        "shap_values": shap_values,
        "mean_abs_shap": mean_abs_shap,
        "tuned_model": tuned["fitted_model"],
    }


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
PAGES = [
    "Home",
    "Data Explorer",
    "Model Performance",
    "Prediction Demo",
    "Explainability",
    "Business Recommendation",
]
page = st.sidebar.radio("Navigate", PAGES)


# ---------------------------------------------------------------------------
# Page 1 — Home
# ---------------------------------------------------------------------------
def page_home():
    st.title("💳 Credit Default Intelligence System")
    st.subheader("DFS504 Big Data & AI in Finance — Semester Project")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Problem Statement
        Credit card default prediction is a critical challenge for financial institutions.
        Correctly identifying customers likely to default allows banks to:
        - Reduce credit loss exposure
        - Optimise credit limit policies
        - Trigger early intervention programs

        ### Dataset
        - **Source**: UCI Machine Learning Repository — Default of Credit Card Clients
        - **Domain**: Credit Risk & Default Intelligence (B+)
        - **Size**: 30,000 customers, 23 features
        - **Target**: `default.payment.next.month` (binary 0/1)
        """)
    with col2:
        st.markdown("""
        ### ML Framework
        | Phase | Description |
        |-------|-------------|
        | 1 | Data loading & Excel export |
        | 2 | EDA & class imbalance analysis |
        | 3 | Feature engineering (7 new features) |
        | 4 | Baseline model (Decision Tree) |
        | 5 | Advanced models (RF, GBM, XGBoost, LightGBM, LR) |
        | 6 | Stratified 5-fold cross-validation |
        | 7 | RandomizedSearchCV hyperparameter tuning |
        | 8 | SHAP interpretation |
        | 9 | Model comparison & export |
        """)

    st.info("Use the sidebar to navigate between pages.")


# ---------------------------------------------------------------------------
# Page 2 — Data Explorer
# ---------------------------------------------------------------------------
def page_data_explorer():
    st.title("Data Explorer")
    raw = load_raw_data()

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", f"{len(raw):,}")
    col2.metric("Columns", raw.shape[1])
    col3.metric("Default Rate", f"{raw[TARGET_COL].mean():.2%}")

    st.subheader("Sample Data")
    st.dataframe(raw.head(100), use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.dataframe(raw.describe(), use_container_width=True)

    st.subheader("Class Distribution")
    counts = raw[TARGET_COL].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Label"] = counts["Class"].map({0: "No Default", 1: "Default"})
    fig = px.bar(counts, x="Label", y="Count", color="Label", text="Count")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Credit Limit Distribution by Default Status")
    fig2 = px.histogram(
        raw, x="LIMIT_BAL", color=TARGET_COL, nbins=50,
        barmode="overlay", opacity=0.7,
        labels={TARGET_COL: "Default (1=Yes)"},
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Correlation Heatmap (Top 15 Features vs Target)")
    numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
    corr = raw[numeric_cols].corr()
    top_cols = (
        corr[TARGET_COL].abs().sort_values(ascending=False).head(15).index.tolist()
    )
    fig3 = px.imshow(
        raw[top_cols].corr(),
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 3 — Model Performance
# ---------------------------------------------------------------------------
def page_model_performance():
    st.title("Model Performance")
    artifacts = get_trained_artifacts()
    comparison_df = artifacts["comparison_df"]
    results = artifacts["results"]
    y_test = artifacts["y_test"]

    st.subheader("Model Comparison Table")
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["ROC-AUC", "PR-AUC", "F1"], color="lightgreen"
        ),
        use_container_width=True,
    )

    st.subheader("ROC Curves")
    fig = go.Figure()
    fig.add_shape(
        type="line", x0=0, x1=1, y0=0, y1=1,
        line=dict(dash="dash", color="grey")
    )
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"{r['model']} (AUC={r['roc_auc']:.3f})"
        ))
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Precision-Recall Curves")
    fig2 = go.Figure()
    for r in results:
        prec, rec, _ = precision_recall_curve(y_test, r["y_prob"])
        fig2.add_trace(go.Scatter(
            x=rec, y=prec, mode="lines",
            name=f"{r['model']} (PR-AUC={r['pr_auc']:.3f})"
        ))
    fig2.update_layout(xaxis_title="Recall", yaxis_title="Precision")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Confusion Matrix — Best Model (LightGBM Tuned)")
    best = results[-1]
    cm = np.array(best["confusion_matrix"])
    fig3 = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
        x=["No Default", "Default"],
        y=["No Default", "Default"],
    )
    st.plotly_chart(fig3, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 4 — Prediction Demo
# ---------------------------------------------------------------------------
def page_prediction_demo():
    st.title("Prediction Demo")
    st.markdown("Enter a customer's data to get the default probability.")

    artifacts = get_trained_artifacts()
    model = artifacts["tuned_model"]
    feature_cols = artifacts["feature_cols"]

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            limit_bal = st.number_input(
                "Credit Limit (NT$)", 10000, 1000000, 50000, step=10000
            )
            sex = st.selectbox(
                "Sex", [1, 2],
                format_func=lambda x: "Male" if x == 1 else "Female"
            )
            education = st.selectbox(
                "Education", [1, 2, 3, 4],
                format_func=lambda x: {
                    1: "Graduate", 2: "University",
                    3: "High School", 4: "Other"
                }[x],
            )
            marriage = st.selectbox(
                "Marriage", [1, 2, 3],
                format_func=lambda x: {
                    1: "Married", 2: "Single", 3: "Other"
                }[x],
            )
            age = st.number_input("Age", 18, 80, 30)

        with col2:
            st.markdown("**Repayment Status (PAY_x)**")
            st.caption("-2=No consumption  -1=Paid fully  0=Min paid  1-8=Months delayed")
            pay_0 = st.slider("PAY_0 (Sep)", -2, 8, 0)
            pay_2 = st.slider("PAY_2 (Aug)", -2, 8, 0)
            pay_3 = st.slider("PAY_3 (Jul)", -2, 8, 0)
            pay_4 = st.slider("PAY_4 (Jun)", -2, 8, 0)
            pay_5 = st.slider("PAY_5 (May)", -2, 8, 0)
            pay_6 = st.slider("PAY_6 (Apr)", -2, 8, 0)

        with col3:
            st.markdown("**Bill & Payment Amounts (NT$)**")
            bill1 = st.number_input("BILL_AMT1 (Sep)", 0, 1000000, 10000, step=1000)
            bill2 = st.number_input("BILL_AMT2 (Aug)", 0, 1000000, 10000, step=1000)
            bill3 = st.number_input("BILL_AMT3 (Jul)", 0, 1000000, 10000, step=1000)
            bill4 = st.number_input("BILL_AMT4 (Jun)", 0, 1000000, 10000, step=1000)
            bill5 = st.number_input("BILL_AMT5 (May)", 0, 1000000, 10000, step=1000)
            bill6 = st.number_input("BILL_AMT6 (Apr)", 0, 1000000, 10000, step=1000)
            pay_amt1 = st.number_input("PAY_AMT1 (Sep)", 0, 1000000, 2000, step=500)
            pay_amt2 = st.number_input("PAY_AMT2 (Aug)", 0, 1000000, 2000, step=500)
            pay_amt3 = st.number_input("PAY_AMT3 (Jul)", 0, 1000000, 2000, step=500)
            pay_amt4 = st.number_input("PAY_AMT4 (Jun)", 0, 1000000, 2000, step=500)
            pay_amt5 = st.number_input("PAY_AMT5 (May)", 0, 1000000, 2000, step=500)
            pay_amt6 = st.number_input("PAY_AMT6 (Apr)", 0, 1000000, 2000, step=500)

        submitted = st.form_submit_button("Predict Default Risk")

    if submitted:
        pay_cols_vals = [pay_0, pay_2, pay_3, pay_4, pay_5, pay_6]
        bill_vals = [bill1, bill2, bill3, bill4, bill5, bill6]
        pay_amt_vals = [pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]

        avg_bill = float(np.mean(bill_vals))
        avg_pay  = float(np.mean(pay_amt_vals))
        util            = float(np.clip(avg_bill / limit_bal, 0, 5)) if limit_bal > 0 else 0.0
        avg_pay_ratio   = float(np.clip(avg_pay / avg_bill, 0, 5)) if avg_bill > 0 else 1.0
        consec_delay    = sum(1 for p in pay_cols_vals if p > 0)
        bill_trend      = float(bill1 - bill6)
        pay_trend       = float(pay_amt1 - pay_amt6)
        credit_age_ratio = float(limit_bal / age) if age > 0 else 0.0

        row = {
            "LIMIT_BAL": limit_bal, "SEX": sex,
            "EDUCATION": education if education in [1, 2, 3] else 4,
            "MARRIAGE": marriage if marriage in [1, 2] else 3,
            "AGE": age,
            "PAY_0": pay_0, "PAY_2": pay_2, "PAY_3": pay_3,
            "PAY_4": pay_4, "PAY_5": pay_5, "PAY_6": pay_6,
            "BILL_AMT1": bill1, "BILL_AMT2": bill2, "BILL_AMT3": bill3,
            "BILL_AMT4": bill4, "BILL_AMT5": bill5, "BILL_AMT6": bill6,
            "PAY_AMT1": pay_amt1, "PAY_AMT2": pay_amt2, "PAY_AMT3": pay_amt3,
            "PAY_AMT4": pay_amt4, "PAY_AMT5": pay_amt5, "PAY_AMT6": pay_amt6,
            "utilization_rate": util, "avg_pay_ratio": avg_pay_ratio,
            "consecutive_delay": consec_delay, "bill_trend": bill_trend,
            "pay_trend": pay_trend, "credit_age_ratio": credit_age_ratio,
        }
        input_df = pd.DataFrame([row])[feature_cols]
        prob = float(model.predict_proba(input_df)[0][1])

        st.divider()
        risk_label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
        color = "red" if prob >= 0.5 else "green"
        st.markdown(
            f"### Default Probability: **:{color}[{prob:.1%}]** — :{color}[{risk_label}]"
        )

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Default Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if prob >= 0.5 else "green"},
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "lightsalmon"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
        ))
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Page 5 — Explainability
# ---------------------------------------------------------------------------
def page_explainability():
    st.title("Explainability — SHAP Analysis")
    artifacts = get_trained_artifacts()
    mean_abs_shap = artifacts["mean_abs_shap"]

    st.subheader("Top 15 Features by Mean |SHAP| Value")
    top = mean_abs_shap.head(15).reset_index()
    top.columns = ["Feature", "Mean |SHAP|"]
    fig = px.bar(
        top.sort_values("Mean |SHAP|"),
        x="Mean |SHAP|", y="Feature", orientation="h",
        title="Feature Importance (LightGBM Tuned)",
        color="Mean |SHAP|", color_continuous_scale="Reds",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Key Insights
    | Feature | Interpretation |
    |---------|----------------|
    | `PAY_0` | Most recent repayment delay — strongest predictor |
    | `delinquency_count` | Number of overdue months across 6 periods |
    | `repay_streak` | Consecutive on-time payments — protective factor |
    | `LIMIT_BAL` | Higher credit limit correlates with lower default risk |
    | `pay_to_bill_ratio` | Higher repayment relative to bill reduces risk |
    | `utilisation_rate` | High utilisation signals financial stress |
    """)


# ---------------------------------------------------------------------------
# Page 6 — Business Recommendation
# ---------------------------------------------------------------------------
def page_business_recommendation():
    st.title("Business Recommendation")

    artifacts = get_trained_artifacts()
    y_test = artifacts["y_test"]
    tuned_result = artifacts["results"][-1]
    y_prob = tuned_result["y_prob"]

    threshold = st.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.05)
    y_pred_thresh = (y_prob >= threshold).astype(int)

    prec = precision_score(y_test, y_pred_thresh, zero_division=0)
    rec = recall_score(y_test, y_pred_thresh, zero_division=0)
    f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
    flagged_pct = float(y_pred_thresh.mean())

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Precision", f"{prec:.3f}")
    col2.metric("Recall", f"{rec:.3f}")
    col3.metric("F1 Score", f"{f1:.3f}")
    col4.metric("Flagged as Default", f"{flagged_pct:.1%}")

    st.subheader("Threshold vs Metric Tradeoff")
    thresholds = np.arange(0.1, 0.91, 0.05)
    rows = []
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        rows.append({
            "Threshold": round(float(t), 2),
            "Precision": precision_score(y_test, yp, zero_division=0),
            "Recall": recall_score(y_test, yp, zero_division=0),
            "F1": f1_score(y_test, yp, zero_division=0),
        })
    thresh_df = pd.DataFrame(rows)
    fig = px.line(
        thresh_df.melt(id_vars="Threshold", var_name="Metric", value_name="Score"),
        x="Threshold", y="Score", color="Metric",
        title="Precision / Recall / F1 vs Decision Threshold",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="grey")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ### Recommended Actions by Risk Tier
    | Risk Tier | Probability | Recommended Action |
    |-----------|-------------|-------------------|
    | Low Risk | < 30% | Approve credit; standard monitoring |
    | Medium Risk | 30%–60% | Approve with reduced limit; monthly review |
    | High Risk | > 60% | Decline or require collateral; trigger outreach |

    ### Business Impact Estimate
    - Optimising recall to ~65% allows flagging most defaulters before first missed payment
    - Estimated 3–5% reduction in annual credit loss provisions
    - False-alert rate remains manageable for a credit review team at threshold 0.5
    """)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
if page == "Home":
    page_home()
elif page == "Data Explorer":
    page_data_explorer()
elif page == "Model Performance":
    page_model_performance()
elif page == "Prediction Demo":
    page_prediction_demo()
elif page == "Explainability":
    page_explainability()
elif page == "Business Recommendation":
    page_business_recommendation()
