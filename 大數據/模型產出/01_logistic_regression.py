"""
DFS504 - Domain B: Credit Risk & Default Intelligence
Model: Logistic Regression (Baseline)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    average_precision_score, precision_recall_curve
)

OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
DATA_PATH    = r'C:\Users\Acer\AppData\Local\Temp\UCI_Credit_Card.csv'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ── 1. 載入與清理資料 ─────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
df.drop(columns=['ID'], inplace=True)
df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
df['MARRIAGE']  = df['MARRIAGE'].replace({0: 3})

# ── 2. Feature Engineering ────────────────────────────────────
bill_cols  = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
pay_cols   = ['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']
delay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']

avg_bill = df[bill_cols].mean(axis=1)
avg_pay  = df[pay_cols].mean(axis=1)

df['utilization_rate']  = np.where(df['LIMIT_BAL'] > 0, avg_bill / df['LIMIT_BAL'], 0).clip(0, 5)
df['avg_pay_ratio']     = np.where(avg_bill > 0, avg_pay / avg_bill, 1.0).clip(0, 5)
df['consecutive_delay'] = (df[delay_cols] > 0).sum(axis=1)
df['bill_trend']        = df['BILL_AMT1'] - df['BILL_AMT6']
df['pay_trend']         = df['PAY_AMT1']  - df['PAY_AMT6']
df['credit_age_ratio']  = df['LIMIT_BAL'] / df['AGE']

# ── 3. 分割資料 ───────────────────────────────────────────────
TARGET = 'default.payment.next.month'
X = df.drop(columns=[TARGET])
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

categorical_features = ['SEX','EDUCATION','MARRIAGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
numeric_features = [c for c in X_train.columns if c not in categorical_features]

scaler = StandardScaler()
X_train_s = X_train.copy()
X_test_s  = X_test.copy()
X_train_s[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_s[numeric_features]  = scaler.transform(X_test[numeric_features])

# ── 4. 交叉驗證 ───────────────────────────────────────────────
model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, class_weight='balanced')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv  = cross_validate(model, X_train_s, y_train,
                     cv=skf, scoring=['accuracy','precision','recall','f1','roc_auc'], n_jobs=-1)

print('=== Logistic Regression - 5-Fold CV ===')
for m in ['accuracy','precision','recall','f1','roc_auc']:
    print(f'  {m:<12}: {cv[f"test_{m}"].mean():.4f} ± {cv[f"test_{m}"].std():.4f}')

# ── 5. 測試集評估 ─────────────────────────────────────────────
model.fit(X_train_s, y_train)
y_pred  = model.predict(X_test_s)
y_proba = model.predict_proba(X_test_s)[:, 1]

metrics = {
    'Accuracy':      accuracy_score(y_test, y_pred),
    'Precision':     precision_score(y_test, y_pred),
    'Recall':        recall_score(y_test, y_pred),
    'F1':            f1_score(y_test, y_pred),
    'ROC-AUC':       roc_auc_score(y_test, y_proba),
    'Avg Precision': average_precision_score(y_test, y_proba),
}

print('\n=== Test Set Results ===')
for k, v in metrics.items():
    print(f'  {k:<15}: {v:.4f}')

# ── 6. 圖表輸出 ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Logistic Regression - Evaluation', fontsize=14, fontweight='bold')

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Default','Default'], yticklabels=['No Default','Default'])
axes[0].set_title('Confusion Matrix')
axes[0].set_ylabel('Actual'); axes[0].set_xlabel('Predicted')

fpr, tpr, _ = roc_curve(y_test, y_proba)
axes[1].plot(fpr, tpr, color='steelblue', lw=2, label=f'AUC={metrics["ROC-AUC"]:.4f}')
axes[1].plot([0,1],[0,1],'k--', lw=1)
axes[1].set_title('ROC Curve'); axes[1].set_xlabel('FPR'); axes[1].set_ylabel('TPR')
axes[1].legend(); axes[1].grid(alpha=0.3)

prec_c, rec_c, _ = precision_recall_curve(y_test, y_proba)
axes[2].step(rec_c, prec_c, color='darkorange', lw=2, label=f'AP={metrics["Avg Precision"]:.4f}')
axes[2].axhline(y=y_test.mean(), color='gray', linestyle='--', label='Baseline')
axes[2].set_title('Precision-Recall Curve'); axes[2].set_xlabel('Recall'); axes[2].set_ylabel('Precision')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'logistic_regression_result.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.show()
print(f'\nSaved: {out_path}')
print(classification_report(y_test, y_pred, target_names=['No Default','Default']))

