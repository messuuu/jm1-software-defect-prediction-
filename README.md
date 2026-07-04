# JM1 Software Defect Prediction using Machine Learning

Predicting software module defects from static code metrics using classical ML and gradient boosting, with a focus on handling severe class imbalance correctly (no data leakage between train/test splits).

## Overview

This project trains and compares four classifiers — Logistic Regression, Decision Tree, Random Forest, and XGBoost — to predict whether a software module is defective, using the NASA **JM1** dataset (21 McCabe/Halstead static code metrics per module, e.g. lines of code, cyclomatic complexity, Halstead volume/effort). The pipeline covers preprocessing, SMOTE-based class balancing applied only to the training fold, feature scaling, hyperparameter tuning, and evaluation using metrics appropriate for imbalanced data (ROC-AUC, PR-AUC, confusion matrix).

## Repository Structure

```
jm1-software-defect-prediction/
├── data/
│   ├── jm1.csv                  # NASA JM1 dataset (13,204 modules, 21 metrics + defect label)
│   ├── bug_dataset_large.csv    # Supplementary defect dataset (100k rows) used for extended experiments
│   └── data_README.md           # Column definitions and data source notes
├── notebooks/
│   ├── Bug_Prediction_KC1_Final.ipynb     # Baseline pipeline: scale → SMOTE → Random Forest
│   └── Agile_Testing_ML_Upgraded.ipynb    # Improved pipeline: correct train/test split before SMOTE, GridSearchCV tuning
├── figures/
│   ├── fig1_system_architecture.svg       # End-to-end pipeline diagram
│   ├── fig2_smote_flowchart.svg           # SMOTE resampling flow
│   ├── fig3_random_forest_diagram.svg     # Random Forest model diagram
│   ├── fig4_xgboost_diagram.svg           # XGBoost model diagram
│   ├── fig5_roc_curves.html               # Interactive ROC curves (all models)
│   ├── fig6_pr_curves.html                # Interactive Precision-Recall curves
│   ├── fig7_confusion_matrix.html         # XGBoost confusion matrix breakdown
│   └── fig8_feature_correlation.html      # Feature correlation with defect label
├── requirements.txt
├── LICENSE
└── README.md
```

## Dataset

**JM1** (NASA Metrics Data Program) — 13,204 software modules written in C, each described by 21 static code metrics (lines of code, cyclomatic complexity `v(g)`, Halstead metrics, operator/operand counts, branch count) and a binary `defects` label (`true`/`false`). This is one of the most widely used benchmark datasets in software defect prediction research.

Class distribution is imbalanced: roughly 16% of modules are labeled defective. See `data/data_README.md` for the full column reference.

## Methodology

1. **Preprocessing** — handled missing values, separated features/target.
2. **Train/test split first** — data is split into train and test sets *before* any resampling or scaling is fit, to prevent test-set leakage into the training process.
3. **Class balancing** — SMOTE (Synthetic Minority Over-sampling) applied only to the training fold.
4. **Feature scaling** — `StandardScaler` fit on training data only, then applied to the test set.
5. **Model training** — Logistic Regression, Decision Tree, Random Forest, and XGBoost, with `GridSearchCV` (5-fold CV, F1-scoring) used to tune the Random Forest's `n_estimators`, `max_depth`, and `min_samples_split`.
6. **Evaluation** — ROC-AUC and Precision-Recall AUC (more informative than ROC-AUC alone on imbalanced data), confusion matrix, accuracy, precision, recall, and F1.

The two notebooks represent an iteration of this pipeline: `Bug_Prediction_KC1_Final.ipynb` is the initial baseline, and `Agile_Testing_ML_Upgraded.ipynb` fixes a data-leakage issue (splitting before resampling/scaling) and adds hyperparameter tuning.

## Results

Evaluated on the JM1 held-out test set (n = 2,641):

| Model | ROC-AUC | PR-AUC |
|---|---|---|
| Logistic Regression | 0.673 | 0.312 |
| Decision Tree | 0.710 | 0.341 |
| Random Forest | 0.790 | 0.398 |
| **XGBoost** | **0.812** | **0.431** |

*(Baseline PR-AUC at the class prior is 0.159 — all models substantially outperform random guessing, and PR-AUC is prioritized here since the positive class is the minority.)*

**XGBoost confusion matrix** (best-performing model):

| | Predicted: Non-defective | Predicted: Defective |
|---|---|---|
| **Actual: Non-defective** | TN = 2,084 | FP = 138 |
| **Actual: Defective** | FN = 228 | TP = 191 |

Accuracy: **84.1%** · Precision: **58.0%** · Recall: **45.6%** · F1: **0.511**

**Top predictive features** (by absolute Pearson correlation with `defects`): `loc` (lines of code) and `branchCount` were the strongest individual predictors, followed by `total_Opnd` and `n` (program length).

See the `figures/` folder for the full ROC/PR curve comparisons and feature correlation chart (interactive HTML, open in a browser).

## Setup

```bash
pip install -r requirements.txt
```

Then open either notebook in Jupyter:

```bash
jupyter notebook notebooks/Agile_Testing_ML_Upgraded.ipynb
```

## Key Takeaways

- Naively applying SMOTE or scaling *before* the train/test split leaks information from the test set and inflates reported performance — this project explicitly avoids that.
- Tree-based ensembles (Random Forest, XGBoost) substantially outperform linear models on this dataset, consistent with published NASA MDP benchmark results.
- Recall on the defective class remains the harder problem (45.6% for the best model) — a reminder that static-metric-only defect prediction has real limits, and precision/recall tradeoffs should be tuned to the cost of missed defects vs. false alarms in a given deployment context.

## License

MIT — see `LICENSE`.
