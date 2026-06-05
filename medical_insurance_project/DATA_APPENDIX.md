# Data Appendix
**Project:** Medical Insurance Charge Prediction · AI Applications in Finance (2026)

## 1. Data source & citation
- **Dataset:** Medical Cost Personal Dataset (`medical_insurance.xlsx`), 2,772 rows × 7 columns.
- **Origin:** Choi, M. (2018) *Medical Cost Personal Datasets*, Kaggle
  (https://www.kaggle.com/datasets/mirichoi0218/insurance); originally published in
  Lantz, B. (2019) *Machine Learning with R*, 3rd edn, Packt.
- **Provenance note:** the 2,772-row file is the canonical **1,338-row** dataset duplicated; after
  removing exact duplicates **1,337 unique records** remain. No external data was added.

## 2. Variables
| Variable | Type | Description |
|---|---|---|
| `age` | numeric (int) | Age of primary beneficiary (18–64) |
| `sex` | categorical | `male` / `female` |
| `bmi` | numeric (float) | Body-mass index (kg/m²), 15.96–53.13 |
| `children` | numeric (int) | Number of dependents (0–5) |
| `smoker` | categorical | `yes` / `no` |
| `region` | categorical | `northeast / northwest / southeast / southwest` |
| `charges` | numeric (float) | **Target** — annual medical charges (USD) |

## 3. Preprocessing decisions
1. **Missing values:** none present → no imputation.
2. **De-duplication:** dropped exact duplicate rows **before** splitting (2,772 → 1,337) to
   prevent train/test leakage.
3. **Encoding:** One-Hot with `drop="first"` for `sex`, `smoker`, `region`.
4. **Scaling:** `StandardScaler` applied to all numeric features (required for the linear
   baseline; neutral for trees).
5. **Feature engineering:**
   - `smoker_obese = (smoker=="yes") × (bmi≥30)` — interaction isolating the top cost segment.
   - `age2 = age²` — non-linear age effect.
   - `obese = (bmi≥30)` — clinical flag.

## 4. Train / test split
- **Strategy:** random 80 / 20 split, `random_state=42`.
- **Sizes:** train = 1,069 rows, test = 268 rows.
- **Rationale:** no temporal ordering → random split valid; continuous target → no stratification.

## 5. Hyperparameters (tuned by 5-fold CV grid search)
| Model | Search grid | Selected |
|---|---|---|
| Linear Regression | — (no hyperparameters) | default |
| Gradient Boosting | n_estimators {200,400}, max_depth {2,3}, learning_rate {0.05,0.1}, subsample {0.8,1.0} | **lr=0.05, depth=2, n=200, subsample=1.0** |
| Random Forest | n_estimators {300,500}, max_depth {None,6,10}, min_samples_leaf {1,3} | depth=6, leaf=3, n=300 |

## 6. Reproducibility
- Global seed `RANDOM_STATE = 42` (NumPy + all sklearn estimators/splits).
- Run `python medical_insurance_pipeline.py` (regenerates `results/`) **or** execute
  `medical_insurance_analysis.ipynb` top-to-bottom — both reproduce identical numbers.
- Environment: Python 3.11, pandas, numpy, scikit-learn, matplotlib, seaborn, openpyxl.

## 7. Headline results
| Model | Test R² | Test MAE (USD) | Test RMSE (USD) |
|---|---|---|---|
| Linear Regression (baseline) | 0.906 | 2,394 | 4,158 |
| Gradient Boosting | 0.902 | 2,451 | 4,237 |
| Random Forest | 0.902 | 2,370 | 4,235 |
