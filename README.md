# BCS3101 Capstone — Heart Disease Prediction

**Name:** MUGISHA ALEX
**Reg Number:** 2024/A/KCS/5461/F
**Course:** BCS3101 — Basics of Machine Learning
**University:** Kabale University

---

## Overview

A supervised machine learning project predicting heart disease presence
from clinical and demographic features. Uses the UCI Heart Disease dataset
(Cleveland subset, 303 patients, 13 features).

## Problem

Given a patient's clinical measurements (age, blood pressure, cholesterol,
chest pain type, ECG results, etc.), predict whether they have heart disease.
The model is intended as a **screening aid** to prioritise patients for
confirmatory testing — not as a diagnostic tool.

## Approach

- **Preprocessing:** leakage-safe pipeline using scikit-learn
  `ColumnTransformer` — impute (median/mode), scale (StandardScaler),
  encode (OneHotEncoder). Fitted on training data only.
- **Models compared:** Logistic Regression, Decision Tree, k-Nearest
  Neighbours, Random Forest, Gradient Boosting.
- **Evaluation:** 5-fold stratified cross-validation, plus a held-out test set.
- **Tuning:** GridSearchCV on the two best candidates.

## Results

| Model | CV F1 (mean ± std) | Test F1 | Test Accuracy |
|-------|--------------------|---------|---------------|
| **Logistic Regression (tuned)** | **0.8363** | 0.8621 | 0.8689 |
| k-Nearest Neighbours (tuned) | 0.7955 | 0.9310 | 0.9344 |
| Random Forest | 0.7389 ± 0.043 | 0.8621 | 0.8689 |
| Gradient Boosting | 0.7238 ± 0.054 | 0.8667 | 0.8689 |
| Decision Tree | 0.6881 ± 0.069 | 0.6154 | 0.6721 |

**Final model:** Tuned Logistic Regression. Selected for its highest
cross-validated F1 score, stability across folds, and interpretability —
critical for a medical decision-support tool.

## Repository structure
