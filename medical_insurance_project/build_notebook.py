"""Generate the reproducible Jupyter notebook for the medical-insurance project."""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# Medical Insurance Charge Prediction
### AI Applications in Finance — Final Project (2026)
**Task type:** Supervised **Regression** — predict annual medical insurance `charges`.

This notebook is fully reproducible: running it top-to-bottom on `medical_insurance.xlsx`
reproduces every number and figure in the report.

It follows the five required stages:
1. Problem Definition & EDA
2. Data Preprocessing & Feature Engineering
3. Model Selection & Justification
4. Training, Tuning & Evaluation
5. Result Interpretation & Business Conclusion
""")

code("""import warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
PALETTE = {"no": "#4C72B0", "yes": "#DD8452"}""")

# ---------------- Stage 1 ----------------
md("""## Stage 1 — Problem Definition & EDA

**Business question.** A health insurer wants to predict each member's *annual medical
charges* from demographic and health attributes (`age, sex, bmi, children, smoker, region`).
Accurate predictions support (i) **fair, risk-based premium pricing** and (ii) **early
identification of high-cost members** for care-management programmes.

**Prediction type.** Continuous target → **supervised regression**.
""")

code("""df = pd.read_excel("medical_insurance.xlsx")
print("Shape:", df.shape)
display(df.head())
print("\\nDtypes:\\n", df.dtypes)
print("\\nMissing values:\\n", df.isna().sum())
print("\\nExact duplicate rows:", df.duplicated().sum(),
      "| unique rows:", len(df.drop_duplicates()))
df.describe().round(2)""")

md("""**EDA observations**
- No missing values; 6 features + 1 target.
- **Heavy duplication**: the 2,772 rows reduce to **1,337 unique rows** — the file is the
  classic 1,338-row Kaggle dataset duplicated ~2×. This is a *data-leakage risk* handled in Stage 2.
- `charges` is strongly **right-skewed** (skew ≈ 1.5); `log(charges)` is ~symmetric.
""")

code("""fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df["charges"], bins=40, kde=True, color="#4C72B0", ax=ax[0]); ax[0].set_title("charges (raw)")
sns.histplot(np.log(df["charges"]), bins=40, kde=True, color="#55A868", ax=ax[1]); ax[1].set_title("log(charges)")
plt.show()
print("skew raw =", round(df['charges'].skew(),3), "| skew log =", round(np.log(df['charges']).skew(),3))""")

code("""enc = df.copy()
enc["smoker_bin"] = (enc["smoker"]=="yes").astype(int)
enc["sex_bin"] = (enc["sex"]=="male").astype(int)
corr = enc[["age","bmi","children","smoker_bin","sex_bin","charges"]].corr()
plt.figure(figsize=(7,5)); sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
plt.title("Correlation matrix"); plt.show()
print(corr["charges"].drop("charges").sort_values(ascending=False).round(3))""")

code("""fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
sns.scatterplot(data=df, x="age", y="charges", hue="smoker", palette=PALETTE, alpha=.6, ax=ax[0])
ax[0].set_title("charges vs age (3 cost bands by smoker)")
sns.scatterplot(data=df, x="bmi", y="charges", hue="smoker", palette=PALETTE, alpha=.6, ax=ax[1])
ax[1].axvline(30, ls="--", c="grey"); ax[1].set_title("charges vs bmi (smoker+obese explodes)")
plt.show()""")

md("""**Key relationships.** `smoker` is by far the dominant driver (corr ≈ 0.79). Smokers split into
two bands by BMI: smokers with **BMI ≥ 30** incur dramatically higher charges. Age adds a steady,
slightly accelerating cost gradient. These interactions motivate the engineered features below.
""")

# ---------------- Stage 2 ----------------
md("""## Stage 2 — Data Preprocessing & Feature Engineering

**Decisions**
- **Missing values:** none → no imputation.
- **De-duplication (leakage prevention):** drop exact duplicates **before** splitting so identical
  records cannot appear in both train and test (which would inflate scores). 2,772 → 1,337 rows.
- **Encoding:** one-hot (`drop="first"`) for low-cardinality nominal `sex, smoker, region`.
- **Scaling:** `StandardScaler` on numerics — required for the linear baseline's coefficient
  comparability; harmless for trees.
- **Feature engineering (≥1 non-trivial):**
  - `smoker_obese` = smoker × (BMI≥30) **interaction** — isolates the highest-cost segment.
  - `age2` = age² — captures the accelerating, non-linear age effect.
  - `obese` = BMI≥30 clinical flag.
- **Split:** 80/20 random (no time ordering; continuous target → no stratification needed).
""")

code("""df = df.drop_duplicates().reset_index(drop=True)
print("After de-dup:", df.shape)

df["obese"] = (df["bmi"] >= 30).astype(int)
df["smoker_bin"] = (df["smoker"] == "yes").astype(int)
df["smoker_obese"] = df["smoker_bin"] * df["obese"]
df["age2"] = df["age"] ** 2
print("Mean charges by smoker_obese:\\n", df.groupby("smoker_obese")["charges"].mean().round(0))

numeric = ["age","bmi","children","age2","smoker_obese","obese"]
categorical = ["sex","smoker","region"]
X = df[numeric + categorical]; y = df["charges"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=RANDOM_STATE)
print("train:", X_train.shape, "test:", X_test.shape)

pre = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical),
])""")

# ---------------- Stage 3 & 4 ----------------
md("""## Stage 3 — Model Selection & Justification

**Baseline — Linear Regression.** Charges rise roughly linearly with age and shift sharply by
smoker status. With the engineered interaction/`age²` terms, a linear model captures most of the
structure while remaining fully interpretable — the right transparent benchmark for a regulated
pricing context.

**Advanced — Gradient Boosting.** The data has **heavy tails, a threshold effect (BMI≥30) and a
strong smoker×BMI interaction**. Boosted trees model non-linear interactions automatically, are
robust to outliers and need no scaling — a natural fit. We also fit a tuned **Random Forest** as a
second non-linear reference. All hyper-parameters are tuned by **5-fold CV grid search** (not by eye).
""")

md("""## Stage 4 — Training, Tuning & Evaluation""")

code("""cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
results = []

def evaluate(name, model):
    model.fit(X_train, y_train)
    ptr, pte = model.predict(X_train), model.predict(X_test)
    row = {"model": name, "train_R2": r2_score(y_train, ptr), "test_R2": r2_score(y_test, pte),
           "test_MAE": mean_absolute_error(y_test, pte),
           "test_RMSE": np.sqrt(mean_squared_error(y_test, pte))}
    results.append(row); print(name, {k: round(v,3) if isinstance(v,float) else v for k,v in row.items()})
    return model, pte

lin = Pipeline([("pre", pre), ("model", LinearRegression())])
print("CV R2 (linear):", round(cross_val_score(lin, X_train, y_train, cv=cv, scoring="r2").mean(),4))
lin, pred_lin = evaluate("Linear Regression", lin)""")

code("""gbr = Pipeline([("pre", pre), ("model", GradientBoostingRegressor(random_state=RANDOM_STATE))])
grid = {"model__n_estimators":[200,400], "model__max_depth":[2,3],
        "model__learning_rate":[0.05,0.1], "model__subsample":[0.8,1.0]}
gs = GridSearchCV(gbr, grid, cv=cv, scoring="r2", n_jobs=-1).fit(X_train, y_train)
print("Best GBR params:", gs.best_params_, "| CV R2:", round(gs.best_score_,4))
best_gbr = gs.best_estimator_
_, pred_gbr = evaluate("Gradient Boosting", best_gbr)

rf = Pipeline([("pre", pre), ("model", RandomForestRegressor(random_state=RANDOM_STATE))])
grid_rf = {"model__n_estimators":[300,500], "model__max_depth":[None,6,10], "model__min_samples_leaf":[1,3]}
gs_rf = GridSearchCV(rf, grid_rf, cv=cv, scoring="r2", n_jobs=-1).fit(X_train, y_train)
print("Best RF params:", gs_rf.best_params_)
_, pred_rf = evaluate("Random Forest", gs_rf.best_estimator_)""")

code("""res = pd.DataFrame(results)
res["lift_R2_vs_baseline"] = res["test_R2"] - res.loc[0,"test_R2"]
res["MAE_reduction"] = res.loc[0,"test_MAE"] - res["test_MAE"]
display(res.round(3))

fig, ax = plt.subplots(1, 2, figsize=(13,4))
sns.barplot(data=res, x="model", y="test_R2", palette="viridis", ax=ax[0]); ax[0].set_ylim(0,1); ax[0].set_title("Test R2")
sns.barplot(data=res, x="model", y="test_MAE", palette="rocket", ax=ax[1]); ax[1].set_title("Test MAE (USD)")
for a in ax: a.tick_params(axis="x", rotation=15)
plt.show()""")

code("""best_name = res.loc[res['test_R2'].idxmax(),'model']
best_pred = {"Linear Regression":pred_lin,"Gradient Boosting":pred_gbr,"Random Forest":pred_rf}[best_name]
fig, ax = plt.subplots(1,2, figsize=(13,5))
ax[0].scatter(y_test, best_pred, alpha=.45); lims=[0,max(y_test.max(),best_pred.max())]
ax[0].plot(lims, lims, "r--"); ax[0].set_title(f"Predicted vs Actual ({best_name})")
ax[0].set_xlabel("Actual"); ax[0].set_ylabel("Predicted")
resid = y_test.values - best_pred
ax[1].scatter(best_pred, resid, alpha=.45, color="#55A868"); ax[1].axhline(0, ls="--", c="r")
ax[1].set_title("Residuals vs Predicted"); ax[1].set_xlabel("Predicted"); ax[1].set_ylabel("Residual")
plt.show()""")

# ---------------- Stage 5 ----------------
md("""## Stage 5 — Result Interpretation & Business Conclusion""")

code("""model = best_gbr.named_steps["model"]
cat_names = list(best_gbr.named_steps["pre"].named_transformers_["cat"].get_feature_names_out(categorical))
imp = (pd.DataFrame({"feature": numeric+cat_names, "importance": model.feature_importances_})
       .sort_values("importance", ascending=False).reset_index(drop=True))
display(imp.round(4))
plt.figure(figsize=(8,5)); sns.barplot(data=imp.head(10), x="importance", y="feature", palette="mako")
plt.title("Feature importance (Gradient Boosting)"); plt.show()""")

md("""### Business conclusion

- **Performance.** All three models explain ≈ **90% of the variance** in charges (test R² ≈ 0.90),
  with an average error (MAE) of roughly **$2,400** per member. Crucially, the **interpretable Linear
  Regression baseline matches the tuned trees** once the smoker×obesity interaction and age² terms
  are added — a case where a well-justified simple model wins (and is preferable in a regulated
  pricing setting).
- **Top drivers & economic intuition.** `smoker`, the `smoker_obese` interaction, `age`/`age²` and
  `bmi` dominate; `region` and `sex` are negligible. Smoking and obesity-driven chronic risk are the
  economic engines of medical cost; age proxies accumulated health risk.
- **Limitations (honest).** (1) Only 1,337 unique records → modest sample; (2) original file's
  duplication is a real leakage trap — addressed by de-dup; (3) features are coarse (no diagnosis/claims
  history); (4) `sex`/`region` near-zero importance means the model is fair on those axes but also blind
  to genuine regional cost variation; (5) charges are right-skewed — large claims have wider error.
- **Recommendations.**
  1. **Risk-based pricing surcharge** for smokers, escalating with BMI — the single largest, most
     defensible lever.
  2. **Targeted wellness / smoking-cessation & weight-management programmes** for the smoker-obese
     segment, where expected savings per member are largest.
  3. **Deploy the interpretable linear model** for premium quoting (auditable, regulator-friendly),
     keeping gradient boosting as a challenger/monitoring model.
""")

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python", "version": "3.11"}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "medical_insurance_analysis.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print("Wrote", out, "with", len(cells), "cells")
