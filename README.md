# 📡 Telco Customer Churn Prediction & ML Comparison Dashboard

> **Live App:** 🚀 [Launch Dashboard](https://telco-dashboard-3bfjmvq9x9wd7nfqz4zuhz.streamlit.app/) LIVE App You Can See


## 🧭 Project Overview

This is a **complete end-to-end Data Science and Machine Learning project** built around the real-world problem of customer churn in the telecommunications industry. The project covers the full ML pipeline — from raw data exploration to a deployed, interactive prediction dashboard.

The solution compares **6 machine learning classification algorithms**, applies structured hyperparameter tuning, and presents results through a polished **Streamlit web application** with live prediction capability.

---

## 💼 Business Problem

> **"Retaining a customer costs far less than acquiring a new one."**

In the telecom industry, customer churn (i.e., subscribers canceling their service) directly impacts monthly recurring revenue. The goals of this project are:

- ✅ **Predict** which customers are at high risk of churning
- ✅ **Identify** the key behavioral and demographic factors driving churn
- ✅ **Enable** proactive retention strategies such as loyalty incentives, targeted offers, and contract renegotiations
- ✅ **Minimize** false negatives (missed churners) to protect revenue

---

## 📊 Dataset Details

| Property | Details |
|---|---|
| **Source** | [Telco Customer Churn — Kaggle](https://github.com/Abu-Bakar-Hameed/telco-dashboard) |
| **Total Records** | 7,043 rows |
| **Total Features** | 21 columns |
| **Target Variable** | `Churn` (Yes / No) |
| **Class Distribution** | ~73.5% No Churn · ~26.5% Churn (imbalanced) |

### Feature Summary

| Category | Features |
|---|---|
| **Demographics** | `gender`, `SeniorCitizen`, `Partner`, `Dependents` |
| **Account Info** | `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod` |
| **Services** | `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `TechSupport`, `StreamingTV`, `StreamingMovies` |
| **Charges** | `MonthlyCharges`, `TotalCharges` |
| **Target** | `Churn` |

> ⚠️ **Note:** `TotalCharges` was stored as an `object` string type and required explicit conversion to `float64`. Rows with blank `TotalCharges` values were removed.





## 🛠 Technologies Used

| Category | Tools & Libraries |
|---|---|
| **Language** | Python 3.10+ |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | scikit-learn, XGBoost |
| **Class Imbalance** | imbalanced-learn (SMOTE) |
| **Web App** | Streamlit |
| **Model Serialization** | Joblib / Pickle |
| **Deployment** | Streamlit Community Cloud |
| **Version Control** | Git, GitHub |

---

## 🔍 Key EDA Discoveries

### 1. Contract Type is the Strongest Churn Predictor
Customers on **Month-to-Month contracts churn at ~42%**, compared to just ~11% on one-year and ~3% on two-year contracts. Locking customers into longer contracts is a clear retention lever.

### 2. Fiber Optic Users Churn More
Despite being a premium service, **Fiber Optic internet customers show significantly higher churn rates** than DSL or No-internet customers — suggesting service quality or pricing dissatisfaction.

### 3. Electronic Check Payment = Higher Churn
Customers paying via **electronic check churn at nearly double the rate** of those using automatic bank transfer or credit card — a potential indicator of lower engagement or trust.

### 4. Short Tenure = High Churn Risk
The majority of churned customers have a **tenure under 12 months**. Early onboarding experience is critical.

### 5. Senior Citizens Are More Vulnerable
**Senior citizen customers (~25% churn rate)** are disproportionately represented among churned users compared to non-seniors (~19%).

### 6. Higher Monthly Charges Correlate with Churn
Churned customers have a noticeably higher **median monthly charge (~$74)** vs. retained customers (~$61), indicating price sensitivity.

---

## 🤖 Machine Learning Models

Six classification algorithms were implemented, trained, and evaluated:

| # | Model | Type | Key Strength |
|---|---|---|---|
| 1 | **Logistic Regression** | Linear | Interpretable baseline |
| 2 | **Decision Tree** | Non-linear | Intuitive split rules |
| 3 | **Random Forest** | Ensemble (Bagging) | Reduces variance & overfitting |
| 4 | **K-Nearest Neighbors (KNN)** | Instance-based | Local spatial density classification |
| 5 | **Support Vector Machine (SVM)** | Margin-based | Maximum-margin hyperplane optimization |

> 🔬 **Optional Extensions:** AdaBoost, CatBoost, LightGBM were also explored for additional benchmarking.

---

### Metric Interpretation Guide

| Metric | Why It Matters |
|---|---|
| **Accuracy** | Overall correctness — use carefully with imbalanced data |
| **Precision** | Avoids wasting marketing budget on false churn alerts |
| **Recall** | Ensures high-risk churners are NOT missed |
| **F1 Score** | Balanced trade-off between precision and recall |
| **ROC-AUC** | Model's ability to distinguish churners vs. non-churners across thresholds |

---



## 🖥 Streamlit Dashboard Features

The interactive web application is divided into **6 sections**:

| Section | Description |
|---|---|
| 🏠 **Home** | Project overview, business context, and navigation index |
| 📂 **Dataset Explorer** | Searchable data tables, `df.describe()`, null value logs, column profiles |
| 📈 **EDA Dashboard** | Interactive Plotly charts with dropdown selectors for trends, correlations, demographics |
| 🧪 **Model Training** | Select any model, trigger training, and instantly view performance metrics |
| 📊 **Model Comparison** | Leaderboard table, ROC curves, confusion matrix grid, feature importance charts |
| 🔮 **Prediction Form** | User input form (sliders, dropdowns, toggles) → returns **Churn Risk / Retained** with a **Confidence Score %** |

---

## ☁️ Deployment

The application is deployed publicly and accessible via the link at the top of this README.

**Supported Platforms:**
- https://telco-dashboard-3bfjmvq9x9wd7nfqz4zuhz.streamlit.app/ ← Recommended (free, GitHub sync)

---

## 💻 Local Installation & Execution

### Prerequisites
- Python 3.10 or higher
- Git

### Step 1 — Clone the Repository
```bash
git clone https://github.com/Abu-Bakar-Hameed/telco-dashboard.git
cd telco-churn-prediction
```

### Step 2 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Launch the Streamlit App
```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

### Step 4 — Run the Notebooks (Optional)
```bash
jupyter notebook notebooks/
```
Execute notebooks in order: `01_EDA` → `02_Preprocessing` → `03_Modeling_and_Tuning`

---

## 🔮 Future Improvements

- [ ] Add real-time customer data ingestion via API
- [ ] Implement SHAP explainability plots for per-prediction reasoning
- [ ] Integrate LightGBM and CatBoost into the comparison pipeline
- [ ] Add a time-series churn forecasting module
- [ ] Build an automated retraining pipeline with MLflow tracking

---

## 👤 Author

**Abu Bakar Hameed**
📧 abubakarhameedpirzado@gmail.com
🔗 www.linkedin.com/in/abu-bakar-hameed · https://github.com/Abu-Bakar-Hameed

---

> *Built as part of a Data Science & Machine Learning course project. Dataset sourced from [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).*
