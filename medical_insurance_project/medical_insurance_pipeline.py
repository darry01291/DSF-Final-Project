"""
=====================================================================
AI Applications in Finance — Final Project (2026)
Task: Medical Insurance Charge Prediction (Supervised Regression)
=====================================================================

End-to-end, reproducible pipeline covering the five required stages:
    Stage 1 - Problem Definition & EDA
    Stage 2 - Data Preprocessing & Feature Engineering
    Stage 3 - Model Selection & Justification
    Stage 4 - Training, Tuning & Evaluation
    Stage 5 - Result Interpretation & Business Conclusion

Run:
    python medical_insurance_pipeline.py

Outputs (written to ./results):
    *.png  figures used in the slide deck / report
    metrics_summary.csv, model_comparison.csv, feature_importance.csv
    results_log.txt  (full numeric log)
"""

import os
import json
import warnings
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------- #
# Reproducibility & paths
# --------------------------------------------------------------------- #
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "medical_insurance.xlsx")
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

PALETTE = {"no": "#4C72B0", "yes": "#DD8452"}


def save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [figure] {name}")


# ===================================================================== #
# STAGE 1 — Problem Definition & Exploratory Data Analysis
# ===================================================================== #
def stage1_eda(df):
    print("\n" + "=" * 70)
    print("STAGE 1 — PROBLEM DEFINITION & EDA")
    print("=" * 70)

    print("\nBusiness question:")
    print("  Predict an individual's annual medical insurance CHARGES from")
    print("  demographic & health attributes, so the insurer can price")
    print("  premiums fairly and flag high-cost members for care management.")
    print("\nPrediction type: SUPERVISED REGRESSION (continuous target = charges)")

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumn types:")
    print(df.dtypes.to_string())
    print("\nMissing values:")
    print(df.isna().sum().to_string())
    print(f"\nExact duplicate rows: {df.duplicated().sum()}  "
          f"(unique rows = {len(df.drop_duplicates())})")

    print("\nNumeric summary:")
    print(df.describe().round(2).to_string())

    print("\nTarget (charges) distribution:")
    print(f"  mean   = {df['charges'].mean():,.0f}")
    print(f"  median = {df['charges'].median():,.0f}")
    print(f"  skew   = {df['charges'].skew():.3f}  -> right-skewed / heavy tail")
    print(f"  log-skew = {np.log(df['charges']).skew():.3f}  -> ~symmetric after log")

    print("\nCharges by smoker status:")
    print(df.groupby("smoker")["charges"].agg(["count", "mean", "median"]).round(0).to_string())

    # ---- Figure 1: target distribution (raw vs log) -----------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.histplot(df["charges"], bins=40, kde=True, color="#4C72B0", ax=ax[0])
    ax[0].set_title("Charges (raw) — right-skewed")
    ax[0].set_xlabel("Annual charges (USD)")
    sns.histplot(np.log(df["charges"]), bins=40, kde=True, color="#55A868", ax=ax[1])
    ax[1].set_title("log(Charges) — approx. symmetric")
    ax[1].set_xlabel("log(Annual charges)")
    fig.suptitle("Figure 1 — Target distribution", fontweight="bold")
    save(fig, "fig01_target_distribution.png")

    # ---- Figure 2: correlation matrix (numeric + encoded) -----------
    enc = df.copy()
    enc["smoker_bin"] = (enc["smoker"] == "yes").astype(int)
    enc["sex_bin"] = (enc["sex"] == "male").astype(int)
    corr = enc[["age", "bmi", "children", "smoker_bin", "sex_bin", "charges"]].corr()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=.5, ax=ax)
    ax.set_title("Figure 2 — Correlation matrix", fontweight="bold")
    save(fig, "fig02_correlation_matrix.png")
    print("\nCorrelation with charges (sorted):")
    print(corr["charges"].drop("charges").sort_values(ascending=False).round(3).to_string())

    # ---- Figure 3: charges vs age, coloured by smoker ---------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="age", y="charges", hue="smoker",
                    palette=PALETTE, alpha=.6, ax=ax)
    ax.set_title("Figure 3 — Charges vs Age (by smoker): 3 cost bands",
                 fontweight="bold")
    ax.set_xlabel("Age"); ax.set_ylabel("Annual charges (USD)")
    save(fig, "fig03_charges_vs_age_smoker.png")

    # ---- Figure 4: bmi vs charges, coloured by smoker ---------------
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker",
                    palette=PALETTE, alpha=.6, ax=ax)
    ax.axvline(30, ls="--", c="grey")
    ax.text(30.3, df["charges"].max() * .95, "BMI = 30 (obesity)", color="grey")
    ax.set_title("Figure 4 — Charges vs BMI: smoker+obese cost explodes",
                 fontweight="bold")
    ax.set_xlabel("BMI"); ax.set_ylabel("Annual charges (USD)")
    save(fig, "fig04_charges_vs_bmi_smoker.png")

    # ---- Figure 5: categorical boxplots -----------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    sns.boxplot(data=df, x="smoker", y="charges", palette=PALETTE, ax=ax[0])
    ax[0].set_title("by Smoker")
    sns.boxplot(data=df, x="region", y="charges", ax=ax[1])
    ax[1].set_title("by Region"); ax[1].tick_params(axis="x", rotation=20)
    sns.boxplot(data=df, x="children", y="charges", ax=ax[2])
    ax[2].set_title("by #Children")
    fig.suptitle("Figure 5 — Charges by categorical features", fontweight="bold")
    save(fig, "fig05_categorical_boxplots.png")

    return df


# ===================================================================== #
# STAGE 2 — Data Preprocessing & Feature Engineering
# ===================================================================== #
def stage2_preprocess(df):
    print("\n" + "=" * 70)
    print("STAGE 2 — DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)

    # --- 2.1 De-duplication (data-leakage prevention) ----------------
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"\nDropped exact duplicates: {before} -> {len(df)} rows.")
    print("  Rationale: the file is the 1,338-row Kaggle set duplicated ~2x.")
    print("  Keeping duplicates would leak identical rows across train/test")
    print("  and inflate test scores. De-dup BEFORE splitting is essential.")

    # --- 2.2 Feature engineering -------------------------------------
    # (a) smoker x obese interaction — strongest driver of the top cost band
    df["obese"] = (df["bmi"] >= 30).astype(int)
    df["smoker_bin"] = (df["smoker"] == "yes").astype(int)
    df["smoker_obese"] = df["smoker_bin"] * df["obese"]
    # (b) age^2 — medical cost rises non-linearly with age
    df["age2"] = df["age"] ** 2
    # (c) bmi band (clinical categories) kept as readable feature for EDA
    print("\nEngineered features:")
    print("  - smoker_obese : interaction; isolates the highest-cost segment")
    print("  - age2         : captures non-linear (accelerating) age effect")
    print("  - obese        : clinical BMI>=30 flag")
    print("\n  Mean charges by (smoker_obese):")
    print(df.groupby("smoker_obese")["charges"].mean().round(0).to_string())

    # --- 2.3 Define feature sets -------------------------------------
    numeric_raw = ["age", "bmi", "children"]
    numeric_eng = ["age2", "smoker_obese", "obese"]
    categorical = ["sex", "smoker", "region"]
    features = numeric_raw + numeric_eng + categorical
    target = "charges"

    X = df[features].copy()
    y = df[target].copy()

    # --- 2.4 Train / test split (random; no time dimension) ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE)
    print(f"\nSplit: train={len(X_train)}  test={len(X_test)}  (80/20, random)")
    print("  No time/index ordering -> random split is appropriate;")
    print("  stratification not needed for continuous target.")

    # --- 2.5 Preprocessing transformer -------------------------------
    # Scale numerics (needed for the linear baseline), one-hot categoricals.
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_raw + numeric_eng),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
        ]
    )
    print("\nPreprocessing: StandardScaler(numeric) + OneHot(drop-first, categorical)")
    print("  Scaling justified for Linear Regression (coef comparability);")
    print("  harmless for trees. One-hot: low-cardinality nominal features.")

    return (df, X_train, X_test, y_train, y_test, pre,
            numeric_raw + numeric_eng, categorical)


# ===================================================================== #
# STAGE 3 & 4 — Model Selection, Tuning, Evaluation
# ===================================================================== #
def evaluate(name, model, X_tr, X_te, y_tr, y_te, store):
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    row = {
        "model": name,
        "train_R2": r2_score(y_tr, pred_tr),
        "test_R2": r2_score(y_te, pred_te),
        "test_MAE": mean_absolute_error(y_te, pred_te),
        "test_RMSE": np.sqrt(mean_squared_error(y_te, pred_te)),
    }
    store.append(row)
    print(f"\n  {name}")
    print(f"    train R2 = {row['train_R2']:.4f}")
    print(f"    test  R2 = {row['test_R2']:.4f}")
    print(f"    test  MAE  = {row['test_MAE']:,.0f} USD")
    print(f"    test  RMSE = {row['test_RMSE']:,.0f} USD")
    return pred_te, row


def stage34_models(X_train, X_test, y_train, y_test, pre):
    print("\n" + "=" * 70)
    print("STAGE 3 & 4 — MODEL SELECTION, TUNING & EVALUATION")
    print("=" * 70)

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []

    # ---- Baseline: Linear Regression --------------------------------
    print("\n[Baseline] Linear Regression")
    print("  Why: interpretable, fast benchmark; charges are roughly linear")
    print("  in age and shift sharply by smoker -> linear+interaction = strong start.")
    lin = Pipeline([("pre", pre), ("model", LinearRegression())])
    lin.fit(X_train, y_train)
    cv_lin = cross_val_score(lin, X_train, y_train, cv=cv, scoring="r2")
    print(f"  5-fold CV R2 = {cv_lin.mean():.4f} +/- {cv_lin.std():.4f}")
    pred_lin, _ = evaluate("Linear Regression", lin, X_train, X_test,
                           y_train, y_test, results)

    # ---- Advanced: Gradient Boosting (tuned) ------------------------
    print("\n[Advanced] Gradient Boosting Regressor (GridSearchCV)")
    print("  Why: charges have heavy tails, threshold effects (BMI>=30) and")
    print("  interactions (smoker x bmi). Boosted trees model non-linear")
    print("  interactions automatically and are robust to feature scaling/outliers.")
    gbr = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
    grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [2, 3],
        "model__learning_rate": [0.05, 0.1],
        "model__subsample": [0.8, 1.0],
    }
    gs = GridSearchCV(gbr, grid, cv=cv, scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"  Best params: {gs.best_params_}")
    print(f"  Best CV R2 = {gs.best_score_:.4f}")
    best_gbr = gs.best_estimator_
    pred_gbr, _ = evaluate("Gradient Boosting", best_gbr, X_train, X_test,
                           y_train, y_test, results)

    # ---- Reference: Random Forest (tuned, secondary advanced) -------
    print("\n[Reference] Random Forest Regressor (GridSearchCV)")
    rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(random_state=RANDOM_STATE))])
    grid_rf = {
        "model__n_estimators": [300, 500],
        "model__max_depth": [None, 6, 10],
        "model__min_samples_leaf": [1, 3],
    }
    gs_rf = GridSearchCV(rf, grid_rf, cv=cv, scoring="r2", n_jobs=-1)
    gs_rf.fit(X_train, y_train)
    print(f"  Best params: {gs_rf.best_params_}")
    best_rf = gs_rf.best_estimator_
    pred_rf, _ = evaluate("Random Forest", best_rf, X_train, X_test,
                          y_train, y_test, results)

    res_df = pd.DataFrame(results)
    res_df["lift_R2_vs_baseline"] = res_df["test_R2"] - res_df.loc[0, "test_R2"]
    res_df["MAE_reduction_vs_baseline"] = res_df.loc[0, "test_MAE"] - res_df["test_MAE"]
    res_df.to_csv(os.path.join(RESULTS_DIR, "model_comparison.csv"), index=False)
    print("\nModel comparison:")
    print(res_df.round(3).to_string(index=False))

    # ---- Figure 6: model comparison bars ----------------------------
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.barplot(data=res_df, x="model", y="test_R2", ax=ax[0], palette="viridis")
    ax[0].set_title("Test R2 (higher = better)"); ax[0].set_ylim(0, 1)
    for i, v in enumerate(res_df["test_R2"]):
        ax[0].text(i, v + .02, f"{v:.3f}", ha="center")
    sns.barplot(data=res_df, x="model", y="test_MAE", ax=ax[1], palette="rocket")
    ax[1].set_title("Test MAE in USD (lower = better)")
    for i, v in enumerate(res_df["test_MAE"]):
        ax[1].text(i, v + 50, f"{v:,.0f}", ha="center")
    for a in ax:
        a.tick_params(axis="x", rotation=15); a.set_xlabel("")
    fig.suptitle("Figure 6 — Baseline vs Advanced models", fontweight="bold")
    save(fig, "fig06_model_comparison.png")

    # ---- Figure 7: predicted vs actual (best model) -----------------
    best_name = res_df.loc[res_df["test_R2"].idxmax(), "model"]
    best_pred = {"Linear Regression": pred_lin, "Gradient Boosting": pred_gbr,
                 "Random Forest": pred_rf}[best_name]
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_test, best_pred, alpha=.45, color="#4C72B0")
    lims = [0, max(y_test.max(), best_pred.max())]
    ax.plot(lims, lims, "r--")
    ax.set_xlabel("Actual charges (USD)"); ax.set_ylabel("Predicted charges (USD)")
    ax.set_title(f"Figure 7 — Predicted vs Actual ({best_name})", fontweight="bold")
    save(fig, "fig07_pred_vs_actual.png")

    # ---- Figure 8: residuals (best model) ---------------------------
    resid = y_test.values - best_pred
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].scatter(best_pred, resid, alpha=.45, color="#55A868")
    ax[0].axhline(0, ls="--", c="r")
    ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Residual")
    ax[0].set_title("Residuals vs Predicted")
    sns.histplot(resid, bins=40, kde=True, color="#55A868", ax=ax[1])
    ax[1].set_title("Residual distribution")
    fig.suptitle(f"Figure 8 — Residual diagnostics ({best_name})", fontweight="bold")
    save(fig, "fig08_residuals.png")

    return res_df, best_gbr, best_rf, best_name


# ===================================================================== #
# STAGE 5 — Interpretation: feature importance
# ===================================================================== #
def stage5_interpret(best_gbr, num_features, cat_features, X_train):
    print("\n" + "=" * 70)
    print("STAGE 5 — RESULT INTERPRETATION (feature importance)")
    print("=" * 70)

    pre = best_gbr.named_steps["pre"]
    model = best_gbr.named_steps["model"]
    cat_names = list(pre.named_transformers_["cat"].get_feature_names_out(cat_features))
    feat_names = num_features + cat_names
    imp = pd.DataFrame({"feature": feat_names,
                        "importance": model.feature_importances_}) \
        .sort_values("importance", ascending=False).reset_index(drop=True)
    imp.to_csv(os.path.join(RESULTS_DIR, "feature_importance.csv"), index=False)
    print("\nGradient Boosting feature importance:")
    print(imp.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=imp.head(10), x="importance", y="feature",
                palette="mako", ax=ax)
    ax.set_title("Figure 9 — Feature importance (Gradient Boosting)",
                 fontweight="bold")
    save(fig, "fig09_feature_importance.png")
    return imp


# ===================================================================== #
# MAIN
# ===================================================================== #
def main():
    log_path = os.path.join(RESULTS_DIR, "results_log.txt")
    with open(log_path, "w") as f, redirect_stdout(f):
        print("MEDICAL INSURANCE CHARGE PREDICTION — RESULTS LOG")
        print("Random seed:", RANDOM_STATE)

        df = pd.read_excel(DATA_PATH)
        df = stage1_eda(df)
        (df, X_train, X_test, y_train, y_test, pre,
         num_features, cat_features) = stage2_preprocess(df)
        res_df, best_gbr, best_rf, best_name = stage34_models(
            X_train, X_test, y_train, y_test, pre)
        imp = stage5_interpret(best_gbr, num_features, cat_features, X_train)

        # machine-readable summary
        summary = {
            "n_unique_rows": int(len(df)),
            "best_model": best_name,
            "results": res_df.round(4).to_dict(orient="records"),
            "top_features": imp.head(5).to_dict(orient="records"),
        }
        with open(os.path.join(RESULTS_DIR, "metrics_summary.json"), "w") as jf:
            json.dump(summary, jf, indent=2)
        res_df.round(4).to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"),
                               index=False)
        print("\nDONE. All artefacts written to ./results")

    # echo log to console too
    with open(log_path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
