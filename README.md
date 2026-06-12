Diabetes Prediction Using Machine Learning

Overview

This project builds machine learning models to predict diabetes using demographic and clinical data.

Key steps included:

- Data preprocessing
- EDA
- Feature engineering
- Model training
- Hyperparameter tuning
- Threshold optimization
- Model evaluation

The final model uses XGBoost with threshold tuning for balanced precision and recall.

---

Dataset

Features

- Age
- Gender
- BMI
- Hypertension
- Heart Disease
- Smoking History
- HbA1c Level
- Blood Glucose Level

Target

- Diabetes (0 = No, 1 = Yes)

---

Data Cleaning

Preprocessing included:

- Removing duplicates
- Checking missing values and outliers
- Filtering unrealistic BMI and age values
- Removing the small "Other" gender category

Extreme HbA1c and glucose values were retained as they may be medically valid.

---

Exploratory Data Analysis

Tools used:

- Histograms
- Boxplots
- Countplots
- Correlation heatmaps

Key Findings

Diabetic patients generally had:

- Higher Blood Glucose
- Higher HbA1c
- Higher BMI
- Older Age

The dataset was also imbalanced, influencing model evaluation.

---

Feature Engineering

- One-Hot Encoding for Gender and Smoking History
- 80/20 stratified train-test split
- StandardScaler applied to numerical features

---

Models Evaluated

- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

---

Hyperparameter Optimization

GridSearchCV was used with F1 Score as the primary metric due to class imbalance.

---

Threshold Optimization

Probability thresholds were tested beyond the default 0.50.

Best XGBoost threshold:

0.27

---

Feature Importance

Most important features:

Feature| Importance
HbA1c Level| Highest
Blood Glucose Level| Very High
BMI| Moderate
Age| Moderate

Clinical measurements were the strongest predictors.

---

Model Comparison

Model| Precision| Recall| F1 Score
Logistic Regression| 0.851| 0.633| 0.726
Random Forest| 0.707| 0.775| 0.739
Gradient Boosting| 0.698| 0.814| 0.752
XGBoost| 0.805| 0.763| 0.784

XGBoost achieved the best overall F1 Score.

---

ROC-AUC Evaluation

Model| ROC-AUC
Logistic Regression| 0.9616
Random Forest| 0.9733
Gradient Boosting| 0.9783
XGBoost| 0.9780

All models showed excellent discrimination.

---

Final Model

Selected Model

XGBoost

Threshold

0.27

Performance

- Precision: 80%
- Recall: 76%
- F1 Score: 78%
- ROC-AUC: 97.8%

---

Deployment Preparation

Saved artifacts:

- Trained XGBoost model
- Feature names
- Classification threshold
- StandardScaler

---

Skills Demonstrated

- Data Cleaning & EDA
- Feature Engineering
- Classification Modeling
- Hyperparameter Tuning
- Model Evaluation
- Model Interpretation
- Deployment Preparation

---

Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Joblib
- Google Colab
