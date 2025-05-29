# Predicting Corporate Financial Distress

This project was developed as part of my Master’s dissertation in Data Science at the University of Essex. The goal is to build a machine learning model capable of identifying companies at risk of financial distress based on historical financial data.

## 📊 Objective

To develop an end-to-end pipeline that:
- Preprocesses and cleans a high-dimensional financial dataset
- Handles missing data and class imbalance
- Trains and evaluates various classification models to predict financial distress
- Interprets feature importance and model performance for real-world impact

## 🗂️ Dataset

- Size: 250,000+ rows, ~180 features
- Source: Public financial datasets (confidential sources anonymized)
- Merged from 5 CSV files with numeric, categorical, and binary columns

## 🔧 Technologies & Libraries

- Python, Jupyter Notebook
- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `imblearn`

## 🧹 Data Preprocessing

- Downcasting for memory optimization
- Imputation using:
  - Mean/Median
  - KNN
  - Regression
  - Random Forest
- Feature engineering (ratios, log transforms)
- Handling class imbalance with **SMOTE**
- Encoding categorical variables
- Scaling using `StandardScaler`

## 📈 Models Trained

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

## 🔍 Evaluation Metrics

- ROC-AUC
- Confusion Matrix
- Precision-Recall Curve
- F1 Score

## 🧪 Model Tuning

- Used `Pipeline` + `GridSearchCV` to streamline hyperparameter tuning and preprocessing

## 📁 Repository Structure

