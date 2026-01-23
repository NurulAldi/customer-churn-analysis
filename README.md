# Customer Churn Prediction

A machine learning project to predict customer churn in a telecommunications company using Logistic Regression with an adjusted prediction threshold.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results & Performance](#results--performance)
- [Key Insights](#key-insights)
- [Predicting New Data](#predicting-new-data)

## Overview

This project aims to predict the likelihood of customer churn (unsubscribing) based on historical customer data. The model used is **Logistic Regression** with a comprehensive preprocessing pipeline that includes:
- Handling missing values
- Standardization for numerical features
- One-Hot Encoding for categorical features
- Class balancing with `class_weight="balanced"`
- Custom threshold (0.3) to improve recall

## Dataset

**Source**: Telco Customer Churn Dataset

**Total Records**: 7,043 customers

**Features**:
- **Demographic**: gender, senior_citizen, partner, dependents
- **Services**: phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies
- **Account**: tenure, contract, paperless_billing, payment_method, monthly_charges, total_charges
- **Target**: churn (Yes/No)


## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup Environment

1. Clone repository atau download project folder

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**Dependencies**:
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- scikit-learn
- imbalanced-learn
- joblib (included in scikit-learn)

## Usage

### 1. Data Cleaning
Jalankan notebook [data_cleaning.ipynb](notebooks/data_cleaning.ipynb) untuk:
- Membersihkan data mentah
- Handle missing values
- Normalize column names
- Convert data types

### 2. Exploratory Data Analysis (EDA)
Jalankan notebook [eda.ipynb](notebooks/eda.ipynb) untuk:
- Analisis distribusi data
- Visualisasi korelasi features
- Identifikasi pattern churn

### 3. Model Training
Jalankan notebook [model.ipynb](notebooks/model.ipynb) untuk:
- Preprocessing data
- Train model
- Evaluate performa
- Analyze feature importance
- Save trained model

## Model

### Model Architecture

**Pipeline**:
```
Input Data
    ↓
Preprocessing Pipeline
    ├── Numeric Features → SimpleImputer (median) → StandardScaler
    └── Categorical Features → SimpleImputer (most_frequent) → OneHotEncoder
    ↓
Logistic Regression (max_iter=2000, class_weight='balanced')
    ↓
Prediction (threshold=0.3)
```

### Hyperparameters

- **Algorithm**: Logistic Regression
- **Max Iterations**: 2000
- **Class Weight**: balanced (handles imbalanced data)
- **Solver**: lbfgs (default)
- **Prediction Threshold**: 0.3 (reduced from default 0.5 to improve recall)

### Preprocessing Details

**Numerical Features**:
- tenure
- monthly_charges
- total_charges

**Categorical Features**:
- gender
- senior_citizen
- partner
- dependents
- phone_service
- multiple_lines
- internet_service
- online_security
- online_backup
- device_protection
- tech_support
- streaming_tv
- streaming_movies
- contract
- paperless_billing
- payment_method

## Results & Performance

### Model Performance (Threshold = 0.3)

**ROC-AUC Score**: ~0.84

**Classification Report**:
```
              precision    recall  f1-score   support

           0       0.96      0.56      0.70      1035
           1       0.43      0.93      0.59       374

    accuracy                           0.65      1409
   macro avg       0.69      0.74      0.65      1409
weighted avg       0.82      0.65      0.67      1409
```

**Confusion Matrix**:
```
[[575   460]
 [ 27  347]]
```

### Why Threshold 0.3?

The threshold was reduced from 0.5 (default) to 0.3 to:
- **Improve Recall**: Capture more customers with potential churn risk
- **Business Impact**: False positives (incorrectly predicting churn) are preferable to false negatives (missing customers who will churn)
- **Cost-Benefit**: Retention campaign costs are cheaper than losing customers

## Key Insights

### Top Features (based on Coefficient & Odds Ratio)

Features with the greatest impact on churn:

1. **Contract_Two year** - Negative coefficient (reduces churn)
   - Customers with 2-year contracts are significantly more loyal
   
2. **Contract_Month-to-month** - Positive coefficient (increases churn)
   - Customers with monthly contracts are at higher risk of churn
   
3. **InternetService_Fiber optic** - Positive coefficient
   - Fiber optic customers are at higher risk of churn (possibly due to pricing or competition)
   
4. **tenure** - Negative coefficient
   - The longer the tenure, the lower the churn risk
   

### Business Action Recommendations

Based on model insights and feature importance analysis:

1. **Offer discounts for one-year and two-year contracts** for new customers (< 12 months tenure) currently on month-to-month contracts
   - Expected Impact: Reduce churn by locking in customers with contract commitments

2. **Provide Free Security Bundle (Tech Support + Online Security) for 4 Months** for new customers (< 12 months tenure) on month-to-month contracts using fiber optic service
   - Expected Impact: Increase perceived value and reduce churn among high-risk fiber optic customers


## Predicting New Data

### Using Python Script

1. Load model and predict:

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('model/churn_model.pkl')

# Prepare new customer data (must have same columns as training data, except 'churn')
new_customer = pd.DataFrame({
    'gender': ['Female'],
    'senior_citizen': [0],
    'partner': ['Yes'],
    'dependents': ['No'],
    'tenure': [12],
    'phone_service': ['Yes'],
    'multiple_lines': ['No'],
    'internet_service': ['Fiber optic'],
    'online_security': ['No'],
    'online_backup': ['No'],
    'device_protection': ['No'],
    'tech_support': ['No'],
    'streaming_tv': ['Yes'],
    'streaming_movies': ['Yes'],
    'contract': ['Month-to-month'],
    'paperless_billing': ['Yes'],
    'payment_method': ['Electronic check'],
    'monthly_charges': [85.0],
    'total_charges': [1020.0]
})

# Drop customer_id if exists
if 'customer_id' in new_customer.columns:
    new_customer = new_customer.drop('customer_id', axis=1)

# Predict
churn_probability = model.predict_proba(new_customer)[:, 1]
churn_prediction = (churn_probability >= 0.3).astype(int)

print(f"Churn Probability: {churn_probability[0]:.2%}")
print(f"Churn Prediction: {'Yes' if churn_prediction[0] == 1 else 'No'}")
```

### Using Predict Script

The `predict.py` script provides a convenient function for making predictions:

```python
import sys
sys.path.append('scripts')
from predict import predict_churn
import pandas as pd

# Prepare new customer data
new_data = pd.DataFrame({
    'gender': ['Female'],
    'senior_citizen': [0],
    'partner': ['Yes'],
    'dependents': ['No'],
    'tenure': [12],
    'phone_service': ['Yes'],
    'multiple_lines': ['No'],
    'internet_service': ['Fiber optic'],
    'online_security': ['No'],
    'online_backup': ['No'],
    'device_protection': ['No'],
    'tech_support': ['No'],
    'streaming_tv': ['Yes'],
    'streaming_movies': ['Yes'],
    'contract': ['Month-to-month'],
    'paperless_billing': ['Yes'],
    'payment_method': ['Electronic check'],
    'monthly_charges': [85.0],
    'total_charges': [1020.0]
})

# Predict with default threshold (0.3)
predictions, probabilities = predict_churn(new_data)

print(f"Churn Prediction: {['No', 'Yes'][predictions[0]]}")
print(f"Churn Probability: {probabilities[0]:.2%}")

# Or use custom threshold
predictions_05, probabilities_05 = predict_churn(new_data, threshold=0.5)
print(f"\nWith 0.5 threshold:")
print(f"Churn Prediction: {['No', 'Yes'][predictions_05[0]]}")
```
```

### Input Format Requirements

Input data must have:
- Same columns as training data (except 'churn' and 'customer_id')
- Column names in snake_case format (lowercase with underscores)
- Categorical values must exactly match training data values
- No unexpected missing values


