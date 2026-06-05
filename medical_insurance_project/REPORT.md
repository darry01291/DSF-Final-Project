# Medical Insurance Charge Prediction — Final Project Report
**Module:** AI Applications in Finance (2026) · **Task:** Supervised Regression

> *Role: junior data scientist at a health insurer. The team lead needs a defensible,
> reproducible model that predicts a member's annual medical charges to support fair,
> risk-based premium pricing and early identification of high-cost members.*

---

## 1. Introduction & Task Framing

**Business question.** Given a member's demographics and health attributes
(`age, sex, bmi, children, smoker, region`), predict their **annual medical insurance
charges (USD)**. Accurate per-member cost prediction lets the insurer (i) price premiums
fairly against expected risk and (ii) flag high-cost members for care management.

**Task type.** The target `charges` is continuous → this is a **supervised regression**
problem (not classification or clustering).

**Why it matters.** Mis-pricing risk is the core failure mode of any insurer: under-pricing
high-risk members erodes the loss ratio, while over-pricing low-risk members loses them to
competitors (adverse selection). A transparent, accurate model directly protects margin.

---

## 2. Stage 1 — Exploratory Data Analysis

| Property | Value |
|---|---|
| Rows (raw) | 2,772 |
| Rows (unique) | **1,337** |
| Features | 6 (`age, sex, bmi, children, smoker, region`) |
| Target | `charges` (USD) |
| Missing values | **0** |
| Target mean / median | 13,261 / 9,333 |
| Target skew | **1.51** (right-skewed); log-skew −0.09 |

**Key findings (see `results/fig01`–`fig05`):**

1. **Heavy duplication.** The 2,772 rows collapse to **1,337 unique rows** — the file is the
   well-known 1,338-row Kaggle "Medical Cost" dataset duplicated ~2×. This is a **data-leakage
   trap** (identical rows could land in both train and test).
2. **Smoking dominates.** `smoker` correlates with `charges` at **0.79** — far above any other
   feature (`age` 0.30, `bmi` 0.20, `children` 0.07, `sex` 0.06).
3. **Three cost bands** (Fig 3–4): non-smokers (low), smokers with BMI < 30 (mid), and
   **smokers with BMI ≥ 30 (very high)**. Mean charges jump from ≈ $9,800 to ≈ $41,600 for the
   smoker-and-obese segment.
4. **Right-skewed target** with a heavy upper tail (large claims) → motivates models robust to
   outliers and non-linearity.

---

## 3. Stage 2 — Data Preprocessing & Feature Engineering

| Decision | Choice | Justification |
|---|---|---|
| Missing values | none to handle | dataset is complete |
| **De-duplication** | drop exact duplicates **before split** | prevents train/test leakage; 2,772 → 1,337 |
| Categorical encoding | One-Hot (`drop="first"`) | low-cardinality nominal vars; avoids dummy trap |
| Numeric scaling | StandardScaler | needed for linear-model coefficient comparability; harmless for trees |
| Train/test split | 80 / 20 random, seed 42 | no time ordering; continuous target needs no stratification |

**Engineered features (rationale required by rubric):**

- **`smoker_obese` = smoker × (BMI ≥ 30)** — an *interaction* term that explicitly isolates the
  highest-cost segment identified in EDA. This is the single most valuable engineered feature.
- **`age2` = age²** — medical cost accelerates with age; the quadratic term lets even a linear
  model capture the curvature.
- **`obese`** — clinical BMI ≥ 30 flag.

---

## 4. Stage 3 & 4 — Models, Tuning & Evaluation

**Models chosen.**
- **Baseline — Linear Regression:** transparent, regulator-friendly benchmark; charges are
  approximately linear in age and shift sharply by smoker, so a linear model + engineered
  interactions captures most structure.
- **Advanced — Gradient Boosting Regressor:** heavy tails, a BMI≥30 threshold effect and a
  smoker×BMI interaction make boosted trees a natural fit (non-linear, interaction-aware,
  outlier-robust, scale-free).
- **Reference — Random Forest:** second non-linear model for comparison.

All hyper-parameters tuned via **5-fold cross-validated grid search** (not by eye).

**Results (test set, 268 rows):**

| Model | Train R² | Test R² | Test MAE (USD) | Test RMSE (USD) |
|---|---|---|---|---|
| **Linear Regression (baseline)** | 0.852 | **0.906** | **2,394** | 4,158 |
| Gradient Boosting (tuned) | 0.869 | 0.902 | 2,451 | 4,237 |
| Random Forest (tuned) | 0.877 | 0.902 | 2,370 | 4,235 |

*Best GBR params:* `learning_rate=0.05, max_depth=2, n_estimators=200, subsample=1.0`.

**Lift analysis.** The advanced models do **not** beat the engineered linear baseline here
(ΔR² ≈ −0.004). This is the headline honest finding: **once the smoker×obesity interaction and
age² are added, a simple, interpretable model matches gradient boosting.** Per the project's own
guiding principle — *"a well-justified Random Forest beats a poorly-tuned deep model"* — the
well-justified linear model is the correct production choice in a regulated pricing context.

See `results/fig06` (comparison), `fig07` (predicted vs actual), `fig08` (residual diagnostics).

---

## 5. Stage 5 — Interpretation & Business Conclusion

**Top predictive features** (Gradient Boosting importance, `results/fig09`):

| Rank | Feature | Importance | Economic intuition |
|---|---|---|---|
| 1 | `smoker_obese` | 0.41 | smoking + obesity = compounded chronic-disease risk |
| 2 | `smoker_yes` | 0.39 | smoking alone is the dominant cost driver |
| 3 | `age2` / `age` | 0.12 | cost accumulates and accelerates with age |
| 4 | `bmi` | 0.07 | higher BMI → higher metabolic/cardiac risk |
| – | `region`, `sex` | ≈ 0.00 | negligible — model is effectively neutral on these |

**Business meaning.** The model explains ≈ **90%** of cost variation with an average error of
≈ **$2,400 per member**. Two health behaviours — smoking and obesity — drive the overwhelming
majority of predictable cost.

**Limitations (stated up front).**
- Only **1,337 unique records** — modest sample; large-claim tail has wider error.
- The source file's duplication is a genuine **leakage trap**; results are valid *only because*
  de-duplication precedes the split.
- Features are coarse: no diagnosis codes, claims history, or income — so the ceiling on accuracy
  is inherent to the data.
- Right-skewed target → heteroscedastic residuals (Fig 8); predictions for very high-cost members
  are less precise.
- `sex`/`region` near-zero importance means the model is fair on those axes but also blind to any
  real regional cost differences.

**Actionable recommendations.**
1. **Risk-based smoker surcharge that escalates with BMI** — the largest, most defensible pricing
   lever, directly grounded in the top feature.
2. **Targeted wellness programmes** (smoking-cessation, weight management) for the smoker-obese
   segment, where expected per-member savings are greatest.
3. **Deploy the interpretable linear model** for quoting (auditable, regulator-friendly) and run
   gradient boosting as a challenger/monitoring model to detect drift.

---

## 6. References (Harvard style)

- Choi, M. (2018) *Medical Cost Personal Datasets*. Kaggle. Available at:
  https://www.kaggle.com/datasets/mirichoi0218/insurance (Accessed: 5 June 2026).
- Lantz, B. (2019) *Machine Learning with R*. 3rd edn. Birmingham: Packt — original source of the
  insurance cost dataset.
- Pedregosa, F. et al. (2011) 'Scikit-learn: Machine Learning in Python', *Journal of Machine
  Learning Research*, 12, pp. 2825–2830.
- Friedman, J.H. (2001) 'Greedy function approximation: a gradient boosting machine', *Annals of
  Statistics*, 29(5), pp. 1189–1232.
- James, G., Witten, D., Hastie, T. and Tibshirani, R. (2021) *An Introduction to Statistical
  Learning*. 2nd edn. New York: Springer.
