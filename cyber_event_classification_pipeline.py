"""
Cyber Event Classification Using Machine Learning Techniques
==============================================================
Capstone Project — Data110
Authors: Afifa Sabha, Satvik Kumar
Base Paper: Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.

Complete, self-contained pipeline: data cleaning, EDA, feature engineering,
model development (Logistic Regression, Decision Tree, Random Forest, ANN),
hyperparameter tuning, cross-validation, evaluation, ROC analysis, and
error analysis. Run this script top-to-bottom to reproduce every table,
figure, and metric reported in the notebook, thesis report, and presentation.

Usage:
    python cyber_event_classification_pipeline.py

Requires: cybersecurity_synthesized_data.csv in the working directory.
Outputs: PNG figures saved to ./images/, trained models saved to ./models/.
"""

import os
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import (train_test_split, GridSearchCV,
                                      StratifiedKFold, cross_val_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, label_binarize
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, roc_curve, auc, confusion_matrix,
                              classification_report)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

os.makedirs('images', exist_ok=True)
os.makedirs('models', exist_ok=True)

DATA_PATH = 'cybersecurity_synthesized_data.csv'


# =====================================================================
# 1. LOAD DATASET
# =====================================================================
def load_dataset(path=DATA_PATH):
    df = pd.read_csv(path)
    print(f"Dataset loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# =====================================================================
# 2. DATASET OVERVIEW
# =====================================================================
def dataset_overview(df):
    print("Shape:", df.shape)
    print("\nColumn names:", list(df.columns))
    print("\nData types:\n", df.dtypes)
    missing = df.isnull().sum()
    print("\nMissing values:", "None found" if missing.sum() == 0 else missing[missing > 0])
    print("Duplicate rows:", df.duplicated().sum())


# =====================================================================
# 3. DATA CLEANING
# =====================================================================
def clean_data(df):
    df_clean = df.copy().dropna().drop_duplicates()
    cat_cols_all = df_clean.select_dtypes(include='object').columns.tolist()
    for c in cat_cols_all:
        if c not in ['timestamp', 'attacker_ip', 'target_ip']:
            df_clean[c] = df_clean[c].astype(str).str.strip()
    df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])
    print(f"Shape after cleaning: {df_clean.shape}")

    numeric_cols = ['data_compromised_GB', 'attack_duration_min', 'attack_severity', 'response_time_min']
    outlier_summary = {}
    for col in numeric_cols:
        Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outlier_summary[col] = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    print("Outlier counts (IQR method):")
    print(pd.DataFrame.from_dict(outlier_summary, orient='index', columns=['Outlier Count']))

    return df_clean, numeric_cols


# =====================================================================
# 4. EXPLORATORY DATA ANALYSIS
# =====================================================================
def run_eda(df_clean, numeric_cols):
    plt.figure(figsize=(10, 6))
    order = df_clean['attack_type'].value_counts().index
    sns.countplot(data=df_clean, y='attack_type', order=order, palette='viridis')
    plt.title('Distribution of Attack Types (Target Variable)')
    plt.xlabel('Count'); plt.ylabel('Attack Type'); plt.tight_layout()
    plt.savefig('images/01_class_distribution.png', dpi=150); plt.close()

    plt.figure(figsize=(6, 6))
    df_clean['outcome'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], startangle=90)
    plt.title('Attack Outcome Distribution'); plt.ylabel(''); plt.tight_layout()
    plt.savefig('images/02_outcome_pie.png', dpi=150); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.histplot(df_clean[col], bins=30, kde=True, ax=ax, color='steelblue')
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout(); plt.savefig('images/03_histograms.png', dpi=150); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for ax, col in zip(axes.flatten(), numeric_cols):
        sns.boxplot(data=df_clean, x='attack_type', y=col, ax=ax, palette='Set2')
        ax.set_title(f'{col} by Attack Type'); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.savefig('images/04_boxplots.png', dpi=150); plt.close()

    plt.figure(figsize=(8, 6))
    corr = df_clean[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title('Correlation Heatmap — Numeric Features'); plt.tight_layout()
    plt.savefig('images/05_correlation_heatmap.png', dpi=150); plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    cat_features_to_plot = ['target_system', 'security_tools_used', 'user_role', 'mitigation_method']
    for ax, col in zip(axes.flatten(), cat_features_to_plot):
        order = df_clean[col].value_counts().index
        sns.countplot(data=df_clean, y=col, order=order, ax=ax, palette='mako')
        ax.set_title(f'Distribution of {col}')
    plt.tight_layout(); plt.savefig('images/06_categorical_counts.png', dpi=150); plt.close()

    ct = pd.crosstab(df_clean['industry'], df_clean['attack_type'])
    ct_pct = ct.div(ct.sum(axis=1), axis=0)
    ct_pct.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='tab20')
    plt.title('Attack Type Composition by Industry (Proportion)')
    plt.ylabel('Proportion'); plt.xlabel('Industry')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig('images/07_industry_vs_attacktype.png', dpi=150); plt.close()

    plt.figure(figsize=(9, 6))
    sns.violinplot(data=df_clean, x='outcome', y='attack_severity', palette='Set3')
    plt.title('Attack Severity by Outcome'); plt.tight_layout()
    plt.savefig('images/08_severity_by_outcome.png', dpi=150); plt.close()

    print("EDA complete. 8 figures saved to ./images/")


# =====================================================================
# 5. FEATURE ENGINEERING
# =====================================================================
def engineer_features(df_clean):
    fe = df_clean.copy()
    fe = fe.drop(columns=['attacker_ip', 'target_ip'])
    fe['hour'] = fe['timestamp'].dt.hour
    fe['day_of_week'] = fe['timestamp'].dt.dayofweek
    fe['month'] = fe['timestamp'].dt.month
    fe = fe.drop(columns=['timestamp'])
    fe['is_business_hours'] = fe['hour'].between(9, 17).astype(int)

    target_col = 'attack_type'
    y_raw = fe[target_col]
    X = fe.drop(columns=[target_col])

    le_target = LabelEncoder()
    y = le_target.fit_transform(y_raw)
    print("Target classes:", dict(zip(le_target.classes_, range(len(le_target.classes_)))))

    categorical_cols = ['target_system', 'outcome', 'security_tools_used', 'user_role',
                         'location', 'industry', 'mitigation_method']
    numeric_feature_cols = ['data_compromised_GB', 'attack_duration_min', 'attack_severity',
                             'response_time_min', 'hour', 'day_of_week', 'month', 'is_business_hours']

    preprocessor = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_feature_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

    return X, y, le_target, preprocessor, categorical_cols, numeric_feature_cols


# =====================================================================
# 6. MODEL DEVELOPMENT
# =====================================================================
def train_baseline_models(X_train_proc, y_train, X_test_proc, y_test):
    predictions, prediction_probs = {}, {}

    log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    log_reg.fit(X_train_proc, y_train)
    predictions['Logistic Regression'] = log_reg.predict(X_test_proc)
    prediction_probs['Logistic Regression'] = log_reg.predict_proba(X_test_proc)
    print("Logistic Regression trained.")

    dt_base = DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10)
    dt_base.fit(X_train_proc, y_train)
    print("Baseline Decision Tree trained.")

    rf_base = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_base.fit(X_train_proc, y_train)
    y_pred_rf_base = rf_base.predict(X_test_proc)
    print("Baseline Random Forest test accuracy:", round(accuracy_score(y_test, y_pred_rf_base), 4))

    return log_reg, predictions, prediction_probs


def tune_hyperparameters(X_train_proc, y_train, X_test_proc, y_test, predictions, prediction_probs):
    rf_param_grid = {'n_estimators': [100, 150], 'max_depth': [10, 20]}
    rf_grid = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
                            rf_param_grid, cv=3, scoring='accuracy', n_jobs=1, verbose=1)
    rf_grid.fit(X_train_proc, y_train)
    print("Best Random Forest parameters:", rf_grid.best_params_)
    print("Best CV accuracy:", round(rf_grid.best_score_, 4))
    rf_best = rf_grid.best_estimator_
    predictions['Random Forest'] = rf_best.predict(X_test_proc)
    prediction_probs['Random Forest'] = rf_best.predict_proba(X_test_proc)

    dt_param_grid = {'max_depth': [5, 10, 15, 20], 'min_samples_split': [2, 5, 10]}
    dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE),
                            dt_param_grid, cv=3, scoring='accuracy', n_jobs=1)
    dt_grid.fit(X_train_proc, y_train)
    print("Best Decision Tree parameters:", dt_grid.best_params_)
    print("Best CV accuracy:", round(dt_grid.best_score_, 4))
    dt_best = dt_grid.best_estimator_
    predictions['Decision Tree'] = dt_best.predict(X_test_proc)
    prediction_probs['Decision Tree'] = dt_best.predict_proba(X_test_proc)

    return rf_best, dt_best


def train_ann(X_train_proc, y_train, X_test_proc, y_test, n_classes, predictions, prediction_probs):
    y_train_cat = to_categorical(y_train, num_classes=n_classes)

    ann = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_proc.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    ann.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    history = ann.fit(X_train_proc, y_train_cat, validation_split=0.15, epochs=20,
                       batch_size=512, callbacks=[early_stop], verbose=2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('ANN Loss Curve'); axes[0].set_xlabel('Epoch'); axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('ANN Accuracy Curve'); axes[1].set_xlabel('Epoch'); axes[1].legend()
    plt.tight_layout(); plt.savefig('images/09_ann_training_curves.png', dpi=150); plt.close()

    y_pred_ann_proba = ann.predict(X_test_proc, verbose=0)
    predictions['ANN'] = np.argmax(y_pred_ann_proba, axis=1)
    prediction_probs['ANN'] = y_pred_ann_proba
    print("ANN trained and evaluated.")
    return ann


# =====================================================================
# 7. CROSS VALIDATION
# =====================================================================
def cross_validate_models(log_reg, dt_best, rf_best, X_train_proc, y_train):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {
        'Logistic Regression': cross_val_score(log_reg, X_train_proc, y_train, cv=cv, scoring='accuracy'),
        'Decision Tree': cross_val_score(dt_best, X_train_proc, y_train, cv=cv, scoring='accuracy'),
        'Random Forest': cross_val_score(rf_best, X_train_proc, y_train, cv=cv, scoring='accuracy'),
    }
    cv_summary = pd.DataFrame({m: [s.mean(), s.std()] for m, s in cv_results.items()},
                               index=['Mean CV Accuracy', 'Std Dev']).T
    print(cv_summary)
    return cv_summary


# =====================================================================
# 8. EVALUATION
# =====================================================================
def evaluate_all_models(predictions, prediction_probs, y_test, le_target):
    def evaluate_model(name, y_true, y_pred, y_proba):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        try:
            auc_score = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
        except Exception:
            auc_score = np.nan
        return {'Model': name, 'Accuracy': acc, 'Precision': prec, 'Recall': rec,
                'F1 Score': f1, 'ROC-AUC': auc_score}

    eval_rows = [evaluate_model(m, y_test, predictions[m], prediction_probs[m]) for m in predictions]
    results_table = pd.DataFrame(eval_rows).set_index('Model').round(4).sort_values('Accuracy', ascending=False)
    print(results_table)

    results_table[['Accuracy', 'Precision', 'Recall', 'F1 Score']].plot(kind='bar', figsize=(11, 6))
    plt.title('Model Performance Comparison'); plt.ylabel('Score')
    plt.xticks(rotation=0); plt.legend(loc='lower right'); plt.tight_layout()
    plt.savefig('images/10_model_comparison.png', dpi=150); plt.close()

    class_names = le_target.classes_
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for ax, model_name in zip(axes.flatten(), predictions.keys()):
        cm = confusion_matrix(y_test, predictions[model_name])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=class_names, yticklabels=class_names)
        ax.set_title(f'Confusion Matrix — {model_name}')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual'); ax.tick_params(axis='x', rotation=45)
    plt.tight_layout(); plt.savefig('images/11_confusion_matrices.png', dpi=150); plt.close()

    best_model_name = results_table.index[0]
    print(f"\nClassification Report — {best_model_name}\n")
    print(classification_report(y_test, predictions[best_model_name], target_names=class_names, zero_division=0))

    return results_table, best_model_name


def feature_importance_analysis(rf_best, preprocessor):
    feature_names = preprocessor.get_feature_names_out()
    importances = pd.Series(rf_best.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    plt.figure(figsize=(10, 7))
    importances.plot(kind='barh', color='teal')
    plt.title('Top 15 Feature Importances — Random Forest')
    plt.xlabel('Importance'); plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig('images/12_feature_importance.png', dpi=150); plt.close()
    return importances


# =====================================================================
# 9. ROC CURVE ANALYSIS
# =====================================================================
def roc_curve_analysis(predictions, prediction_probs, y_test, le_target):
    class_names = le_target.classes_
    n_classes = len(class_names)
    y_test_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    roc_auc_summary = {}
    for ax, model_name in zip(axes.flatten(), predictions.keys()):
        y_proba = prediction_probs[model_name]
        fpr, tpr, roc_auc_ = {}, {}, {}
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
            roc_auc_[i] = auc(fpr[i], tpr[i])
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        roc_auc_['macro'] = auc(all_fpr, mean_tpr)

        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], lw=1, alpha=0.5, label=f'{class_names[i]} (AUC={roc_auc_[i]:.2f})')
        ax.plot(all_fpr, mean_tpr, color='navy', lw=2.5, linestyle='--', label=f'Macro-avg (AUC={roc_auc_["macro"]:.3f})')
        ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle=':', label='Chance (AUC=0.50)')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves — {model_name}')
        ax.legend(fontsize=7, loc='lower right')
        roc_auc_summary[model_name] = roc_auc_['macro']

    plt.tight_layout(); plt.savefig('images/13_roc_curves.png', dpi=150); plt.close()
    for m, v in roc_auc_summary.items():
        print(f"{m}: Macro-average AUC = {v:.4f}")
    return roc_auc_summary


# =====================================================================
# 10. ERROR ANALYSIS
# =====================================================================
def error_analysis(predictions, y_test, le_target, best_model_name, X_test):
    class_names = le_target.classes_
    best_preds = predictions[best_model_name]
    y_test_arr = np.array(y_test)
    errors_mask = best_preds != y_test_arr

    print(f"Best model: {best_model_name}")
    print(f"Misclassified: {errors_mask.sum()} ({errors_mask.sum()/len(y_test_arr)*100:.2f}%)")
    print(f"Correct: {(~errors_mask).sum()} ({(~errors_mask).sum()/len(y_test_arr)*100:.2f}%)")

    per_class_error = {}
    for i, cname in enumerate(class_names):
        class_mask = y_test_arr == i
        per_class_error[cname] = round(errors_mask[class_mask].sum() / class_mask.sum() * 100, 2)
    per_class_error_df = pd.DataFrame.from_dict(per_class_error, orient='index',
                                                  columns=['Error Rate (%)']).sort_values('Error Rate (%)', ascending=False)
    print(per_class_error_df)

    plt.figure(figsize=(9, 6))
    per_class_error_df['Error Rate (%)'].plot(kind='barh', color='crimson')
    plt.title(f'Per-Class Error Rate — {best_model_name}')
    plt.xlabel('Error Rate (%)'); plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig('images/14_per_class_error.png', dpi=150); plt.close()

    X_test_reset = X_test.reset_index(drop=True)
    error_df = X_test_reset[errors_mask].copy()
    error_df['true_attack_type'] = [class_names[i] for i in y_test_arr[errors_mask]]
    error_df['predicted_attack_type'] = [class_names[i] for i in best_preds[errors_mask]]
    misclass_pairs = error_df.groupby(['true_attack_type', 'predicted_attack_type']).size().sort_values(ascending=False).head(10)
    print("\nTop 10 misclassification pairs (True -> Predicted):")
    print(misclass_pairs)

    return per_class_error_df, misclass_pairs


# =====================================================================
# MAIN PIPELINE
# =====================================================================
def main():
    df = load_dataset()
    dataset_overview(df)
    df_clean, numeric_cols = clean_data(df)
    run_eda(df_clean, numeric_cols)

    X, y, le_target, preprocessor, categorical_cols, numeric_feature_cols = engineer_features(df_clean)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
    print(f"Training set: {X_train.shape[0]:,} rows | Test set: {X_test.shape[0]:,} rows")

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    log_reg, predictions, prediction_probs = train_baseline_models(X_train_proc, y_train, X_test_proc, y_test)
    rf_best, dt_best = tune_hyperparameters(X_train_proc, y_train, X_test_proc, y_test, predictions, prediction_probs)
    n_classes = len(le_target.classes_)
    train_ann(X_train_proc, y_train, X_test_proc, y_test, n_classes, predictions, prediction_probs)

    cross_validate_models(log_reg, dt_best, rf_best, X_train_proc, y_train)
    results_table, best_model_name = evaluate_all_models(predictions, prediction_probs, y_test, le_target)
    feature_importance_analysis(rf_best, preprocessor)
    roc_curve_analysis(predictions, prediction_probs, y_test, le_target)
    error_analysis(predictions, y_test, le_target, best_model_name, X_test)

    # Persist trained models
    joblib.dump(rf_best, 'models/random_forest_best.pkl')
    joblib.dump(dt_best, 'models/decision_tree_best.pkl')
    joblib.dump(log_reg, 'models/logistic_regression.pkl')
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    joblib.dump(le_target, 'models/label_encoder.pkl')
    print("\nModels saved to ./models/. Pipeline complete.")

    return results_table, best_model_name


if __name__ == '__main__':
    main()
