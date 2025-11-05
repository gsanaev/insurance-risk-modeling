# 🚗 Insurance Risk Modeling – Frequency, Severity, and Pure Premium Analysis

## 📋 Overview
This project develops an analytical and machine learning framework for **risk scoring and premium modeling** in the **auto insurance domain**.  
It simulates a realistic insurance portfolio to estimate expected claim costs (pure premiums) by combining **frequency** and **severity** models.

The project demonstrates a full actuarial–data science workflow that integrates:
- **Statistical modeling (GLMs)** for interpretability, and  
- **Machine learning (XGBoost, LightGBM)** for predictive performance.

All results are reproducible using synthetic data generated via a controlled simulation process that mimics real auto insurance portfolios.

---

## 🎯 Objectives
- Simulate a realistic **auto insurance dataset** with driver, vehicle, and regional risk factors.
- Model **claim frequency** using Poisson regression and tree-based ML.
- Model **claim severity** using Gamma regression and tree-based ML.
- Combine both models to estimate **expected pure premium** per policy.
- Evaluate model performance using both **statistical** and **business metrics**.
- Provide **explainable results** aligned with real-world insurance practice.

---

## 🏗️ Methodological Framework

### 1. Data Simulation
Synthetic dataset generation based on:
- Policy-level variables (driver age, vehicle type, region, etc.)
- Realistic risk relationships
- Poisson process for claim frequency
- Gamma distribution for claim severity

### 2. Modeling Components
| Component | Method | Description |
|------------|---------|-------------|
| Frequency | Poisson GLM / XGBoost | Predicts number of claims per policy |
| Severity | Gamma GLM / XGBoost | Predicts average claim cost given a claim |
| Pure Premium | Frequency × Severity | Expected total claim cost |

### 3. Evaluation Metrics
| Type | Metric | Purpose |
|------|---------|----------|
| Statistical | RMSE, MAE, Deviance | Model accuracy |
| Business | Gini, Loss Ratio, Calibration | Pricing performance |
| Explainability | SHAP, Partial Dependence | Model interpretation |

---

## 🧩 Project Structure

```
insurance-risk-modeling/
├── src/
│ └── ins/
│ ├── simulate_data.py
│ ├── preprocess.py
│ ├── model_frequency.py
│ ├── model_severity.py
│ ├── model_purepremium.py
│ └── utils.py
│
├── notebooks/
│ ├── 01_data_simulation.ipynb
│ ├── 02_exploration.ipynb
│ ├── 03_model_frequency.ipynb
│ ├── 04_model_severity.ipynb
│ ├── 05_combined_purepremium.ipynb
│ └── 06_business_evaluation.ipynb
│
├── data/
│ ├── raw/
│ └── processed/
│
├── outputs/
│ ├── figures/
│ ├── models/
│ └── reports/
│
├── docs/
│ ├── index.html
│ ├── notebooks_html/
│ └── assets/
│
├── requirements.txt
├── LICENSE
├── README.md
└── .gitignore
```


---

## ⚙️ Tools & Libraries
- **Core:** Python, pandas, numpy, scikit-learn  
- **Statistical Models:** statsmodels (GLMs)  
- **Machine Learning:** XGBoost, LightGBM  
- **Visualization:** matplotlib, seaborn, plotly  
- **Interpretability:** SHAP  
- **Dashboard (optional):** Streamlit  

---

## 🧮 Workflow Summary

| Step | Notebook | Description |
|------|-----------|-------------|
| 1 | 01_data_simulation.ipynb | Generate and validate synthetic data |
| 2 | 02_exploration.ipynb | Exploratory data analysis |
| 3 | 03_model_frequency.ipynb | Build frequency model (Poisson, XGBoost) |
| 4 | 04_model_severity.ipynb | Build severity model (Gamma, XGBoost) |
| 5 | 05_combined_purepremium.ipynb | Combine models for expected premium |
| 6 | 06_business_evaluation.ipynb | Analyze results and business metrics |

---

## 📈 Expected Outcomes
- A **reproducible simulation dataset** for auto insurance analytics.
- Comparative analysis of GLM and ML methods.
- Interpretability and business insights.
- Final risk segmentation and pricing evaluation.

---

## 📜 License
This project is licensed under the **MIT License** – you’re free to use, modify, and distribute it with attribution.

---

## 👤 Author
Developed by **[Your Name]**, Data Scientist  
Focused on applied data science and risk analytics in the insurance domain.
