import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    classification_report
)

DATA_PATH = r'C:\Users\Acer\AppData\Local\Temp\UCI_Credit_Card.csv'
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
RANDOM_STATE = 42
COLOR = '#e74c3c'


def load_and_clean(path):
    df = pd.read_csv(path)
    df.drop(columns=['ID'], errors='ignore', inplace=True)

    df['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})
    df['MARRIAGE'] = df['MARRIAGE'].replace({0: 3})

    return df


def engineer_features(df):
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    pay_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    pay_stat_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']

    df['utilization_rate'] = df[bill_cols].mean(axis=1) / (df['LIMIT_BAL'] + 1e-9)

    bill_mean = df[bill_cols].mean(axis=1).replace(0, np.nan)
    df['avg_pay_ratio'] = df[pay_cols].mean(axis=1) / bill_mean
    df['avg_pay_ratio'] = df['avg_pay_ratio'].fillna(0)

    df['consecutive_delay'] = (df[pay_stat_cols] > 0).sum(axis=1)

    bill_arr = df[bill_cols].values
    df['bill_trend'] = bill_arr[:, 0] - bill_arr[:, -1]

    pay_arr = df[pay_cols].values
    df['pay_trend'] = pay_arr[:, 0] - pay_arr[:, -1]

    df['credit_age_ratio'] = df['LIMIT_BAL'] / (df['AGE'] + 1e-9)

    return df


def build_features_target(df):
    engineered = [
        'utilization_rate', 'avg_pay_ratio', 'consecutive_delay',
        'bill_trend', 'pay_trend', 'credit_age_ratio'
    ]
    target = 'default.payment.next.month'
    X = df[engineered]
    y = df[target]
    return X, y


def run_cv(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['accuracy', 'roc_auc', 'f1', 'precision', 'recall']
    results = cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=False)

    print("=== 5-Fold Stratified CV Results ===")
    for metric in scoring:
        scores = results[f'test_{metric}']
        print(f"  {metric:12s}: {scores.mean():.4f} (+/- {scores.std():.4f})")


def plot_results(model, X_test, y_test, feature_names, output_path):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    avg_prec = average_precision_score(y_test, y_prob)
    importances = model.feature_importances_

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle('XGBoost Model Results', fontsize=16, fontweight='bold', color=COLOR)
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.4)

    ax0 = fig.add_subplot(gs[0])
    im = ax0.imshow(cm, interpolation='nearest', cmap='Reds')
    ax0.set_title('Confusion Matrix', fontsize=12, color=COLOR)
    ax0.set_xlabel('Predicted Label')
    ax0.set_ylabel('True Label')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax0.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=12)
    ax0.set_xticks([0, 1])
    ax0.set_yticks([0, 1])
    plt.colorbar(im, ax=ax0)

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(fpr, tpr, color=COLOR, lw=2, label=f'AUC = {roc_auc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', lw=1)
    ax1.set_title('ROC Curve', fontsize=12, color=COLOR)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right')

    ax2 = fig.add_subplot(gs[2])
    ax2.plot(recall, precision, color=COLOR, lw=2, label=f'AP = {avg_prec:.4f}')
    ax2.set_title('Precision-Recall Curve', fontsize=12, color=COLOR)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='upper right')

    ax3 = fig.add_subplot(gs[3])
    sorted_idx = np.argsort(importances)
    ax3.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx], color=COLOR)
    ax3.set_title('Feature Importance', fontsize=12, color=COLOR)
    ax3.set_xlabel('Importance')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


def main():
    df = load_and_clean(DATA_PATH)
    df = engineer_features(df)
    X, y = build_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=3.5,
        random_state=RANDOM_STATE, eval_metric='logloss', verbosity=0
    )

    run_cv(model, X_train, y_train)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("\n=== Test Set Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

    output_path = os.path.join(OUTPUT_DIR, 'xgboost_result.png')
    plot_results(model, X_test, y_test, list(X.columns), output_path)


if __name__ == '__main__':
    main()

