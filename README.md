# Cyber Event Classification Using Machine Learning Techniques

A university capstone project (Data110) that builds a complete machine learning pipeline
to classify the type of a cyber-attack from a simulated cybersecurity incident log.

**Authors:** Afifa Sabha, Satvik Kumar
**Base Paper:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

---

## Project Overview

This project applies and compares four machine learning models — Logistic Regression,
Decision Tree, Random Forest, and an Artificial Neural Network (ANN) — to predict
`attack_type` (Phishing, DDoS, Zero-Day, Ransomware, Malware, SQL Injection, Brute Force,
XSS) from a synthesized cybersecurity events dataset.

## Objectives

- Perform thorough data cleaning and exploratory data analysis on a cyber-event dataset.
- Engineer meaningful features (temporal, categorical, derived) for classification.
- Train and rigorously compare four supervised learning models.
- Tune hyperparameters via `GridSearchCV` and validate stability via k-fold cross-validation.
- Evaluate models with multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC).
- Interpret and honestly report results, including limitations of the dataset.

## Dataset Description

`dataset/cybersecurity_synthesized_data.csv` — 100,000 simulated cybersecurity incident
records, 15 columns: `attack_type`, `target_system`, `outcome`, `timestamp`,
`attacker_ip`, `target_ip`, `data_compromised_GB`, `attack_duration_min`,
`security_tools_used`, `user_role`, `location`, `attack_severity`, `industry`,
`response_time_min`, `mitigation_method`. No missing values, no duplicate rows.

**Target variable:** `attack_type` (8-class, balanced, multiclass classification).

## Technologies Used

- Python 3.12
- pandas, NumPy — data manipulation
- Matplotlib, Seaborn — visualization
- scikit-learn — preprocessing, classical models, model selection
- TensorFlow / Keras — ANN
- Jupyter Notebook

## Installation Steps

```bash
git clone <repo-url>
cd Cyber-Event-Classification
pip install -r requirements.txt
jupyter notebook notebooks/Cyber_Event_Classification.ipynb
```

## Usage Instructions

- **Run the full pipeline interactively:** open `notebooks/Cyber_Event_Classification.ipynb` in Jupyter and run all cells top to bottom (Kernel → Restart & Run All). No external configuration is required beyond the dependencies in `requirements.txt`.
- **Run the pipeline as a script:** `src/cyber_event_classification_pipeline.py` is a single, self-contained script that reproduces the entire pipeline end-to-end. From the repository root: `python src/cyber_event_classification_pipeline.py` (the dataset CSV must be present in the working directory, or update `DATA_PATH` at the top of the script).
- **Reuse the trained models:** load the final model from `models/best_model.pkl` together with `models/preprocessor.joblib` and `models/label_encoder.joblib` to run inference on new, similarly-structured data.
- **View results without rerunning anything:** all figures are in `images/`, all metric tables/reports are in `results/`, the thesis is in `report/`, and the slide deck is in `presentation/`.

## Workflow Diagram

```
 Raw CSV (100,000 rows)
        │
        ▼
 Data Cleaning ──▶ Outlier Check (IQR)
        │
        ▼
 Exploratory Data Analysis (8 plots)
        │
        ▼
 Feature Engineering (temporal features, one-hot encoding, scaling)
        │
        ▼
 Stratified 80/20 Train-Test Split
        │
        ▼
 Model Training ──▶ Logistic Regression / Decision Tree / Random Forest / ANN
        │
        ▼
 GridSearchCV Hyperparameter Tuning (RF, DT)
        │
        ▼
 5-Fold Stratified Cross-Validation
        │
        ▼
 Evaluation ──▶ Accuracy / Precision / Recall / F1 / ROC-AUC / Confusion Matrix
        │
        ▼
 ROC Curve Analysis + Error Analysis
        │
        ▼
 Final Model Selection (Random Forest) + Feature Importance
```

## Project Workflow

1. **Data Loading & Overview** — shape, dtypes, missing values, duplicates
2. **Data Cleaning** — text standardization, timestamp parsing, IQR-based outlier check
3. **Exploratory Data Analysis** — 8 visualizations covering distributions, correlations,
   and categorical relationships
4. **Feature Engineering** — time features, one-hot encoding, scaling, derived features
5. **Train/Test Split** — 80/20 stratified split
6. **Model Development** — Logistic Regression, Decision Tree, Random Forest, ANN
7. **Hyperparameter Tuning** — `GridSearchCV` on Random Forest and Decision Tree
8. **Cross-Validation** — 5-fold stratified CV on classical models
9. **Evaluation** — Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrices
10. **ROC Curve Analysis** — one-vs-rest ROC curves and macro-AUC per model
11. **Error Analysis** — per-class error rates and most common misclassification pairs for the best model

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Random Forest | 0.1290 | 0.1294 | 0.1289 | 0.1285 | 0.5022 |
| Decision Tree | 0.1246 | 0.1253 | 0.1249 | 0.1144 | 0.4986 |
| Logistic Regression | 0.1241 | 0.1240 | 0.1241 | 0.1226 | 0.5007 |
| ANN | 0.1229 | 0.1158 | 0.1232 | 0.0888 | 0.5038 |

*(8-class random baseline ≈ 12.5% accuracy.)*

## Results

All models perform close to the random baseline. Exploratory analysis shows every
feature in this dataset is distributed almost uniformly with respect to `attack_type`,
indicating the synthesized data carries very little class-conditional signal — a genuine
and honestly reported finding rather than a modeling error. Independent confirmation comes
from ROC-AUC (all four models cluster around 0.50 — chance level) and error analysis
(misclassification is spread almost uniformly across all 8 classes and all class pairs,
rather than concentrated between visually/behaviorally similar attack types). **Random
Forest** is recommended as the most defensible model choice (per Breiman, 2001) on
robustness and interpretability grounds, and is the model saved to `models/`.

## Folder Structure

```
Cyber-Event-Classification/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── dataset/
│   └── cybersecurity_synthesized_data.csv
│
├── notebooks/
│   └── Cyber_Event_Classification.ipynb
│
├── src/
│   └── cyber_event_classification_pipeline.py   # single, self-contained pipeline script
│
├── models/
│   ├── best_model.pkl                # tuned Random Forest (final selected model)
│   ├── random_forest_best.joblib     # same model, joblib format
│   ├── decision_tree_best.joblib
│   ├── logistic_regression.joblib
│   ├── preprocessor.joblib           # fitted ColumnTransformer (scaler + one-hot encoder)
│   └── label_encoder.joblib          # fitted target LabelEncoder
│
├── images/
│   ├── 01_class_distribution.png
│   ├── 02_outcome_pie.png
│   ├── 03_histograms.png
│   ├── 04_boxplots.png
│   ├── 05_correlation_heatmap.png
│   ├── 06_categorical_counts.png
│   ├── 07_industry_vs_attacktype.png
│   ├── 08_severity_by_outcome.png
│   ├── 09_ann_training_curves.png
│   ├── 10_model_comparison.png
│   ├── 11_confusion_matrices.png
│   ├── 12_feature_importance.png
│   ├── 13_roc_curves.png
│   └── 14_per_class_error.png
│
├── results/
│   ├── results_table.csv                  # final model comparison (Accuracy/Precision/Recall/F1/ROC-AUC)
│   ├── cv_summary.csv                     # 5-fold cross-validation results
│   ├── hyperparameter_tuning_results.csv  # GridSearchCV best params per model
│   ├── classification_report.txt          # full classification report, best model
│   ├── confusion_matrix_random_forest.csv
│   ├── confusion_matrix_decision_tree.csv
│   ├── confusion_matrix_logistic_regression.csv
│   ├── confusion_matrix_ann.csv
│   ├── roc_auc_summary.csv                # macro ROC-AUC per model
│   ├── per_class_error.csv                # per-class error rate, best model
│   ├── feature_importances.csv            # top Random Forest feature importances
│   └── misclass_pairs.csv                 # most frequent misclassification pairs
│
├── report/
│   ├── Cyber_Event_Classification_Thesis_Report.docx
│   └── Cyber_Event_Classification_Thesis_Report.pdf
│
└── presentation/
    └── Cyber_Event_Classification_Presentation.pptx
```

## Future Improvements

- Apply the same pipeline to a real-world (non-synthetic) cybersecurity incident dataset.
- Add SHAP-based model interpretability analysis.
- Deploy the trained Random Forest model behind a lightweight inference API.
- Explore ensemble/stacking approaches combining tree-based and neural models.

## References

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32.
- Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
- Chollet, F. (2015). Keras. https://keras.io
