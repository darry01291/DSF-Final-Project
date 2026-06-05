# Medical Insurance Charge Prediction
**AI Applications in Finance — Final Project (2026)** · Supervised Regression

Predict a member's **annual medical insurance charges** from demographics and health attributes,
to support risk-based premium pricing and high-cost member identification.

## Contents
| File | Purpose |
|---|---|
| `medical_insurance.xlsx` | Source data (2,772 rows → 1,337 unique) |
| `medical_insurance_pipeline.py` | End-to-end reproducible pipeline (Stages 1–5); writes `results/` |
| `medical_insurance_analysis.ipynb` | Reproducible notebook deliverable (with outputs) |
| `build_notebook.py` | Regenerates the notebook from source |
| `REPORT.md` | Full written report (5 stages, references) |
| `DATA_APPENDIX.md` | Required 1–2 page data appendix |
| `results/` | Figures (`fig01`–`fig09`), metrics CSV/JSON, full log |

## How to run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl nbformat
python medical_insurance_pipeline.py          # regenerates results/
# or open medical_insurance_analysis.ipynb and Run All
```

## Headline results (test set, 80/20 split, seed 42)
| Model | Test R² | Test MAE | Test RMSE |
|---|---|---|---|
| Linear Regression (baseline) | **0.906** | **$2,394** | $4,158 |
| Gradient Boosting (tuned) | 0.902 | $2,451 | $4,237 |
| Random Forest (tuned) | 0.902 | $2,370 | $4,235 |

**Key takeaways**
- Models explain ≈ **90%** of charge variance; avg error ≈ **$2,400/member**.
- Top drivers: **smoking**, **smoker×obesity interaction**, **age**, **BMI**. `region`/`sex` ≈ 0.
- The source file is the 1,338-row Kaggle set duplicated → **de-dup before splitting** to avoid
  data leakage.
- Honest finding: with engineered features, the **interpretable linear baseline matches the
  tuned trees** — the right production choice for a regulated pricing setting.
