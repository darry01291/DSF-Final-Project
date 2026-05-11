"""
DFS504 Semester Project — Credit Risk & Default Intelligence
Domain B+: UCI CC Default Clients
Target: default.payment.next.month

Pipeline phases:
  1. Data loading & Excel export
  2. EDA
  3. Feature engineering
  4. Baseline model
  5. Advanced models (RF, GBM, XGBoost, LightGBM, Logistic Regression)
  6. Cross-validation
  7. Hyperparameter tuning
  8. SHAP interpretation
  9. Results export
"""

import zipfile
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
    RandomizedSearchCV,
)
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import lightgbm as lgb
import shap
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ZIP_PATH = r"C:\Users\Acer\Desktop\大數據\數據 zip.zip"
EXCEL_SAVE_PATH = r"C:\Users\Acer\Desktop\大數據\UCI_Credit_Card.xlsx"
CSV_NAME_IN_ZIP = "UCI_Credit_Card.csv"
TARGET_COL = "default.payment.next.month"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Phase 1 — Data Loading & Excel Export
# ---------------------------------------------------------------------------
def load_data(zip_path: str, csv_name: str) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as archive:
        with archive.open(csv_name) as csv_file:
            dataframe = pd.read_csv(csv_file)
    print(f"[Phase 1] Loaded {len(dataframe):,} rows × {dataframe.shape[1]} columns")
    return dataframe


def export_to_excel(dataframe: pd.DataFrame, save_path: str) -> None:
    dataframe.to_excel(save_path, index=False, engine="openpyxl")
    print(f"[Phase 1] Excel saved → {save_path}")


# ---------------------------------------------------------------------------
# Phase 2 — EDA
# ---------------------------------------------------------------------------
def run_eda(dataframe: pd.DataFrame) -> dict:
    print("\n[Phase 2] EDA Summary")
    print(f"  Shape : {dataframe.shape}")
    missing = dataframe.isnull().sum()
    if missing.any():
        print(f"  Missing values:\n{missing[missing > 0]}")
    else:
        print("  Missing values: none")

    default_rate = dataframe[TARGET_COL].mean()
    print(f"  Default rate: {default_rate:.2%}  (class imbalance present)")

    figures = {}

    counts = dataframe[TARGET_COL].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Label"] = counts["Class"].map({0: "No Default", 1: "Default"})
    figures["class_dist"] = px.bar(
        counts,
        x="Label",
        y="Count",
        title="Class Distribution",
        color="Label",
        text="Count",
    )

    figures["limit_dist"] = px.histogram(
        dataframe,
        x="LIMIT_BAL",
        color=TARGET_COL,
        nbins=50,
        title="Credit Limit Distribution by Default Status",
        barmode="overlay",
        opacity=0.7,
        labels={TARGET_COL: "Default"},
    )

    numeric_cols = dataframe.select_dtypes(include=np.number).columns.tolist()
    corr = dataframe[numeric_cols].corr()
    top_corr_cols = (
        corr[TARGET_COL].abs().sort_values(ascending=False).head(15).index.tolist()
    )
    corr_sub = dataframe[top_corr_cols].corr()
    figures["corr_heatmap"] = px.imshow(
        corr_sub,
        text_auto=".2f",
        title="Correlation Heatmap (Top 15 Features vs Target)",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )

    return figures


# ---------------------------------------------------------------------------
# Phase 3 — Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    df = dataframe.copy()

    bill_cols    = ["BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6"]
    pay_amt_cols = ["PAY_AMT1",  "PAY_AMT2",  "PAY_AMT3",  "PAY_AMT4",  "PAY_AMT5",  "PAY_AMT6"]
    delay_cols   = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    avg_bill = df[bill_cols].mean(axis=1)
    avg_pay  = df[pay_amt_cols].mean(axis=1)

    df["utilization_rate"]  = np.where(df["LIMIT_BAL"] > 0, avg_bill / df["LIMIT_BAL"], 0).clip(0, 5)
    df["avg_pay_ratio"]     = np.where(avg_bill > 0, avg_pay / avg_bill, 1.0).clip(0, 5)
    df["consecutive_delay"] = (df[delay_cols] > 0).sum(axis=1)
    df["bill_trend"]        = df["BILL_AMT1"] - df["BILL_AMT6"]
    df["pay_trend"]         = df["PAY_AMT1"]  - df["PAY_AMT6"]
    df["credit_age_ratio"]  = df["LIMIT_BAL"] / df["AGE"]

    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"]  = df["MARRIAGE"].replace({0: 3})

    df = df.drop(columns=["ID"], errors="ignore")

    print(f"\n[Phase 3] Feature engineering complete — {df.shape[1]} columns total")
    return df


# ---------------------------------------------------------------------------
# Phase 4 & 5 — Model Training Helpers
# ---------------------------------------------------------------------------
def prepare_splits(df: pd.DataFrame):
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n[Phase 4] Train: {X_train.shape}  Test: {X_test.shape}")
    print(f"  Train default rate: {y_train.mean():.2%}")
    print(f"  Test  default rate: {y_test.mean():.2%}")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, scaler


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_prob = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else y_pred.astype(float)
    )

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print(f"\n  {name}")
    print(f"    ROC-AUC : {roc_auc:.4f}")
    print(f"    PR-AUC  : {pr_auc:.4f}")
    print(f"    F1      : {f1:.4f}")
    print(
        f"    Precision/Recall (class 1): "
        f"{report['1']['precision']:.3f} / {report['1']['recall']:.3f}"
    )

    return {
        "model": name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "y_prob": y_prob,
        "fitted_model": model,
    }


def train_baseline(X_train, y_train, X_test, y_test) -> dict:
    print("\n[Phase 4] Baseline — Decision Tree (max_depth=5, class_weight=balanced)")
    model = DecisionTreeClassifier(
        max_depth=5, class_weight="balanced", random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return evaluate_model("Baseline Decision Tree", model, X_test, y_test)


def train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test) -> dict:
    print("\n[Phase 5] Logistic Regression")
    model = LogisticRegression(
        max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)
    return evaluate_model("Logistic Regression", model, X_test_scaled, y_test)


def train_random_forest(X_train, y_train, X_test, y_test) -> dict:
    print("\n[Phase 5] Random Forest")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return evaluate_model("Random Forest", model, X_test, y_test)


def train_gradient_boosting(X_train, y_train, X_test, y_test) -> dict:
    print("\n[Phase 5] Gradient Boosting")
    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return evaluate_model("Gradient Boosting", model, X_test, y_test)


def train_xgboost(X_train, y_train, X_test, y_test) -> dict:
    print("\n[Phase 5] XGBoost")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return evaluate_model("XGBoost", model, X_test, y_test)


def train_lightgbm(X_train, y_train, X_test, y_test) -> dict:
    print("\n[Phase 5] LightGBM")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return evaluate_model("LightGBM", model, X_test, y_test)


# ---------------------------------------------------------------------------
# Phase 6 — Cross-Validation
# ---------------------------------------------------------------------------
def run_cross_validation(models: dict, X_train, y_train) -> pd.DataFrame:
    print("\n[Phase 6] Stratified 5-Fold Cross-Validation (ROC-AUC)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, model in models.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
        )
        rows.append({"model": name, "cv_mean": scores.mean(), "cv_std": scores.std()})
        print(f"  {name}: {scores.mean():.4f} ± {scores.std():.4f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Phase 7 — Hyperparameter Tuning (LightGBM)
# ---------------------------------------------------------------------------
def tune_lightgbm(X_train, y_train, X_test, y_test) -> dict:
    print("\n[Phase 7] RandomizedSearchCV — LightGBM (20 iterations, 5-fold)")
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    param_dist = {
        "n_estimators": [200, 300, 500],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "num_leaves": [31, 63, 127],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }
    base = lgb.LGBMClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    search = RandomizedSearchCV(
        base,
        param_distributions=param_dist,
        n_iter=20,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    print(f"  Best params : {search.best_params_}")
    print(f"  Best CV AUC : {search.best_score_:.4f}")
    return evaluate_model("LightGBM (Tuned)", search.best_estimator_, X_test, y_test)


# ---------------------------------------------------------------------------
# Phase 8 — SHAP Interpretation
# ---------------------------------------------------------------------------
def compute_shap(model, X_test_df: pd.DataFrame) -> tuple:
    print("\n[Phase 8] Computing SHAP values…")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_test_df.columns
    ).sort_values(ascending=False)
    print("  Top 10 features by mean |SHAP|:")
    print(mean_abs_shap.head(10).to_string())
    return shap_values, mean_abs_shap


# ---------------------------------------------------------------------------
# Phase 9 — Comparison & Figures
# ---------------------------------------------------------------------------
def build_comparison_table(results: list) -> pd.DataFrame:
    rows = [
        {
            "Model": r["model"],
            "ROC-AUC": round(r["roc_auc"], 4),
            "PR-AUC": round(r["pr_auc"], 4),
            "F1": round(r["f1"], 4),
            "Precision (Default)": round(r["precision_1"], 4),
            "Recall (Default)": round(r["recall_1"], 4),
        }
        for r in results
    ]
    df = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False)
    print("\n[Phase 9] Model Comparison (sorted by ROC-AUC):")
    print(df.to_string(index=False))
    return df


def build_roc_figure(results: list, X_test_df, y_test) -> go.Figure:
    from sklearn.metrics import roc_curve

    fig = go.Figure()
    fig.add_shape(
        type="line", x0=0, x1=1, y0=0, y1=1,
        line=dict(dash="dash", color="grey")
    )
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr, mode="lines",
                name=f"{r['model']} (AUC={r['roc_auc']:.3f})"
            )
        )
    fig.update_layout(
        title="ROC Curves — All Models",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
    )
    return fig


def build_shap_figure(mean_abs_shap: pd.Series) -> go.Figure:
    top = mean_abs_shap.head(15).reset_index()
    top.columns = ["Feature", "Mean |SHAP|"]
    return px.bar(
        top.sort_values("Mean |SHAP|"),
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title="Top 15 Features — Mean |SHAP| Value (LightGBM Tuned)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("DFS504 Credit Default Prediction Pipeline")
    print("=" * 60)

    raw_df = load_data(ZIP_PATH, CSV_NAME_IN_ZIP)
    export_to_excel(raw_df, EXCEL_SAVE_PATH)

    eda_figures = run_eda(raw_df)

    df = engineer_features(raw_df)

    (
        X_train, X_test, y_train, y_test,
        X_train_scaled, X_test_scaled, scaler,
    ) = prepare_splits(df)

    print("\n[Phase 4 & 5] Training models…")
    baseline_result = train_baseline(X_train, y_train, X_test, y_test)
    lr_result = train_logistic_regression(X_train_scaled, y_train, X_test_scaled, y_test)
    rf_result = train_random_forest(X_train, y_train, X_test, y_test)
    gb_result = train_gradient_boosting(X_train, y_train, X_test, y_test)
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
    lgb_result = train_lightgbm(X_train, y_train, X_test, y_test)

    cv_models = {
        "Random Forest": rf_result["fitted_model"],
        "Gradient Boosting": gb_result["fitted_model"],
        "XGBoost": xgb_result["fitted_model"],
        "LightGBM": lgb_result["fitted_model"],
    }
    cv_df = run_cross_validation(cv_models, X_train, y_train)

    tuned_lgb_result = tune_lightgbm(X_train, y_train, X_test, y_test)

    feature_cols = df.drop(columns=[TARGET_COL]).columns.tolist()
    X_test_df = pd.DataFrame(X_test, columns=feature_cols)
    shap_values, mean_abs_shap = compute_shap(tuned_lgb_result["fitted_model"], X_test_df)

    all_results = [
        baseline_result, lr_result, rf_result,
        gb_result, xgb_result, lgb_result, tuned_lgb_result,
    ]
    comparison_df = build_comparison_table(all_results)
    roc_fig = build_roc_figure(all_results, X_test_df, y_test)
    shap_fig = build_shap_figure(mean_abs_shap)

    with pd.ExcelWriter(
        EXCEL_SAVE_PATH, engine="openpyxl", mode="a", if_sheet_exists="replace"
    ) as writer:
        comparison_df.to_excel(writer, sheet_name="Model Comparison", index=False)
        cv_df.to_excel(writer, sheet_name="CV Results", index=False)
    print(f"\n[Phase 9] Results appended to {EXCEL_SAVE_PATH}")

    for fig in eda_figures.values():
        fig.show()
    roc_fig.show()
    shap_fig.show()

    best = comparison_df.iloc[0]
    print("\n" + "=" * 60)
    print(f"Pipeline complete. Best model: {best['Model']}")
    print(f"  ROC-AUC : {best['ROC-AUC']}")
    print(f"  PR-AUC  : {best['PR-AUC']}")
    print(f"  F1      : {best['F1']}")
    print("=" * 60)

    return {
        "df": df,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "results": all_results,
        "comparison_df": comparison_df,
        "cv_df": cv_df,
        "tuned_model": tuned_lgb_result["fitted_model"],
        "shap_values": shap_values,
        "mean_abs_shap": mean_abs_shap,
        "eda_figures": eda_figures,
        "roc_fig": roc_fig,
        "shap_fig": shap_fig,
        "feature_cols": feature_cols,
    }


if __name__ == "__main__":
    main()
