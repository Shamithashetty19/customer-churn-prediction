# Predicting Customer Churn

An end-to-end machine learning project that predicts customer churn for a subscription-based
business using behavioural, transactional, and demographic data. Built as my MBA capstone
research project (Business Analytics, Jain Online — Deemed-to-be University).

## Problem

The dataset (11,260 customer accounts, 19 features) shows a 16.8% churn rate. The goal is to
identify, in advance, which customers are likely to churn so that retention teams can act
proactively instead of reactively.

## Approach

1. **Data cleaning** — handled missing values (15 of 19 columns) using median/mode imputation,
   corrected a data-entry placeholder (`&&&&` → NaN), and capped outliers at the 99th percentile
   for three heavily skewed columns (cashback, revenue per month, support contacts).
2. **Feature engineering** — encoded categorical variables (label + one-hot encoding), scaled
   numeric features with Min-Max normalization, and engineered two new features
   (`High_Value_Flag`, `Churn_Risk_Score`) based on EDA findings.
3. **Class imbalance** — applied SMOTE *inside* each cross-validation training fold (via an
   imbalanced-learn pipeline) to avoid leaking synthetic samples into validation/test data.
4. **Modeling** — trained and compared four classifiers: Logistic Regression, Decision Tree,
   Random Forest, and XGBoost, using 5-fold stratified cross-validation.
5. **Tuning** — ran GridSearchCV on XGBoost (27 hyperparameter combinations) to optimize F1-score.
6. **Validation** — evaluated on a held-out 20% test set using Accuracy, Precision, Recall,
   F1-score, ROC-AUC, and a confusion matrix (not accuracy alone, since the classes are imbalanced).

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 77.0% | 40.8% | 80.2% | 54.0% | 0.868 |
| Decision Tree | 89.2% | 65.4% | 75.7% | 70.2% | 0.893 |
| Random Forest | 97.5% | 94.7% | 90.0% | 92.3% | 0.993 |
| **XGBoost (Tuned)** | **97.7%** | **96.3%** | **90.0%** | **93.0%** | **0.993** |

**Top churn drivers** (by feature importance): complaint history in the last year, account
tenure, and days since last customer-care contact.

## Business Output

Model probabilities are translated into a three-tier risk segmentation (High / Medium / Low),
each paired with a differentiated retention action — e.g., immediate outreach and complaint
fast-tracking for high-risk accounts, automated campaigns for medium-risk accounts.

## Tech Stack

Python · Pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn (SMOTE) · Matplotlib · Seaborn

## Repository Structure

```
├── Customer_Churn_Prediction.ipynb   # Full analysis and modeling notebook
├── data/                             # Raw dataset and data dictionary
├── charts/                           # Generated visualizations (created on first run)
└── requirements.txt
```

## Running It Yourself

```bash
pip install -r requirements.txt
jupyter notebook Customer_Churn_Prediction.ipynb
```

## Author

**Shamitha R Shetty** — [LinkedIn](https://linkedin.com/in/shamithashetty19)
