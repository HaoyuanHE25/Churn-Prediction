# Churn Prediction

A comprehensive machine learning project for predicting customer churn in telecommunications using advanced data preprocessing, model training, and explainability analysis.

## 📋 Project Overview

This project implements an end-to-end machine learning pipeline to predict customer churn. It combines data preprocessing, feature engineering, multiple model training approaches, and business-driven threshold optimization to maximize business value.

**Key Achievements:**
- **85.3% Accuracy** with Random Forest (10-Fold CV)
- **97% Recall** for churn detection with business-optimized threshold
- **Explainable AI**: SHAP-based model interpretability
- **Production-Ready**: Serialized model (`churn_model_v1.pkl`) ready for deployment

## 📁 Repository Structure

├── WA_Fn-UseC_-Telco-Customer-Churn.csv # Original raw dataset ├── CleanedDataset.csv # Preprocessed & balanced dataset ├── DataPreprocessing.ipynb # Data cleaning & SMOTE oversampling ├── Modeling&Analytics.ipynb # Model training, tuning & analysis ├── churn_model_v1.pkl # Trained Random Forest model ├── shap_values.pkl # Pre-computed SHAP values └── README.md # This file


## 🔄 Pipeline Workflow

### 1. **Data Preprocessing** (`DataPreprocessing.ipynb`)

- **Data Loading**: Telco customer dataset with 7,043 records and 31 features
- **Cleaning**: 
  - Handled hidden null values in `TotalCharges`
  - Filled missing values with monthly charges
- **Encoding**:
  - Converted categorical features to numeric (One-Hot Encoding)
  - Target variable `Churn`: Yes → 1, No → 0
- **Balancing**: 
  - Applied SMOTE (Synthetic Minority Over-sampling Technique)
  - Balanced class distribution: 5,174 samples per class
  - Final dataset: 10,348 records × 31 features
- **Normalization**: MinMax scaling (0-1 range)
- **Shuffling**: Randomized data order post-SMOTE

**Output**: `CleanedDataset.csv`

### 2. **Model Development** (`Modeling&Analytics.ipynb`)

#### Model Comparison
10-Fold Cross-Validation results:

| Model | Mean Accuracy | Std Dev |
|-------|---------------|---------| 
| **Random Forest** | **85.30%** | 0.0147 |
| Gradient Boosting | 83.36% | 0.0111 |
| Logistic Regression | 82.35% | 0.0168 |

#### Feature Engineering
- **Charges_per_Tenure**: Normalized spending intensity (Total Charges / Tenure)
- **Service_Count**: Aggregated extra services (OnlineSecurity, TechSupport, etc.)

#### Threshold Optimization
Business-driven approach with cost-sensitive classification:
- **Missing a churner**: $100 (FN cost)
- **False marketing alarm**: $20 (FP cost)
- **Optimal threshold**: 0.18 (vs default 0.50)

**Results with optimized threshold:**

Precision: 0.69 (Churn class) Recall: 0.97 (Catch 97% of churners!) F1-Score: 0.81 Overall Accuracy: 0.77


#### Model Explainability
- SHAP (SHapley Additive exPlanations) analysis
- Top features driving churn:
  1. Payment Method (Electronic check): 9.6%
  2. Internet Service (Fiber optic): 8.5%
  3. Tenure: 6.7%
  4. Total Charges: 4.9%
  5. Paperless Billing: 4.1%

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn shap joblib

# Execute DataPreprocessing.ipynb
# Generates: CleanedDataset.csv

Usage
1. Run Data Preprocessing

# Execute DataPreprocessing.ipynb
# Generates: CleanedDataset.csv

2. Train Models

# Execute Modeling&Analytics.ipynb
# Generates: churn_model_v1.pkl, shap_values.pkl

3. Load Pre-trained Model

import joblib
import pandas as pd

# Load the trained model
model = joblib.load('churn_model_v1.pkl')

# Load your data
df = pd.read_csv('CleanedDataset.csv')
X = df.drop('Churn', axis=1)

# Make predictions
predictions = model.predict(X)
probabilities = model.predict_proba(X)[:, 1]

# Apply business-optimized threshold
churn_predictions = (probabilities >= 0.18).astype(int)

4. Model Interpretability

import shap
import matplotlib.pyplot as plt

# SHAP Summary Plot
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values[1], X, show=True)

📊 Key Features
✅ Balanced Dataset: SMOTE oversampling resolves class imbalance
✅ Multiple Models: Ensemble methods vs. linear models comparison
✅ Business Optimization: Cost-aware threshold tuning
✅ Model Explainability: SHAP values for feature importance
✅ Production Ready: Serialized model for deployment
✅ Cross-Validation: Robust 10-Fold CV evaluation

📈 Performance Metrics
Accuracy: 77% (with business-optimized threshold)
Recall: 97% (catches almost all churners)
Precision: 69% (reasonable false-positive rate)
F1-Score: 0.81
🔧 Model Details
Algorithm: Random Forest Classifier

Estimators: 200
Max Depth: 20
Min Samples Split: 2
Min Samples Leaf: 2
Bootstrap: False
File: churn_model_v1.pkl (43.8 MB)

💡 Business Impact
Proactive Retention: Identify 97% of likely churners before they leave
Cost Optimization: Threshold tuned to minimize intervention costs
Actionable Insights: SHAP values reveal which factors drive churn for each customer
Scalable: Ready-to-deploy model for real-time predictions
📚 Dataset Info
Source: Telco Customer Churn (Kaggle-like dataset)
Records: 7,043 customers (original), 10,348 (after SMOTE)
Features: 31 (after one-hot encoding)
Target: Churn (binary: Yes/No)
Classes: Balanced 50-50 after preprocessing
🛠️ Technologies
Python 3.13.7
scikit-learn: Machine learning models
pandas: Data manipulation
SHAP: Model explainability
matplotlib/seaborn: Visualization
imbalanced-learn: SMOTE oversampling
joblib: Model serialization
📝 License
This project is open source. Feel free to use and modify for your own purposes.

👤 Author
Created by HaoyuanHE25

🤝 Contributing
Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.