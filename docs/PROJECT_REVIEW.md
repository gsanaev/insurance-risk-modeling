# 🧮 Project Review — Synthetic Insurance Risk Modeling

## 🎯 Objective
This project develops a **synthetic, fully reproducible motor insurance portfolio** to support end-to-end risk modeling workflows.  
It bridges actuarial modeling and data science by integrating **simulation**, **exploratory analysis**, and **predictive modeling** using modern Python tools.

---

## 🏗️ Methodological Overview

### 1. Data Simulation
A fully synthetic dataset of **100,000 motor insurance policies** was generated to mimic realistic portfolio structures, claim behaviors, and policy attributes.  
The simulation incorporates:
- demographic and vehicle-level features (e.g., driver age, vehicle type, mileage),
- region and usage factors (urbanization, commercial use, telematics),
- claim frequency and severity components drawn from Poisson–Gamma distributions,
- reference-based multipliers for calibration to realistic market profiles.

#### 🔒 Data Reserve
To ensure reproducibility, a fixed backup dataset is provided:
> `data/data_reserve/synthetic_insurance_portfolio_2025-11-07.csv`  
> This version-controlled file guarantees project functionality even when `simulate_data.py` cannot be executed.

---

### 2. Exploratory Analysis & Feature Assessment
Notebook **`02_exploratory_analysis.ipynb`** provides:
- descriptive statistics, feature correlations, and visual diagnostics,
- claim frequency and severity distributions by region and vehicle type,
- pairwise relationships and variance analysis across key risk factors.

---

### 3. Modeling Framework
Two-step loss modeling structure:

| Component | Model | Objective |
|------------|--------|-----------|
| **Frequency** | Poisson GLM + Random Forest | Predict number of claims |
| **Severity** | Gamma GLM + Gradient Boosting | Predict average claim size |
| **Pure Premium** | Combined (Freq × Sev) | Expected loss cost per policy |

Each model is benchmarked on MAE, RMSE, and R² metrics.  
Outputs are stored in `outputs/reports/` for later business evaluation.

---

### 4. Business Evaluation
Notebook **`06_business_evaluation.ipynb`** summarizes:
- portfolio KPIs: frequency, severity, and pure premium,
- risk segmentation (vehicle, urban density, usage),
- loss concentration (Lorenz, Gini),
- pricing uplift scenarios (technical → loaded premiums),
- portfolio loss ratios and profit margins.

---

## 📊 Key Deliverables

| Output | Description |
|:--------|:-------------|
| **Synthetic Portfolio (`.csv`)** | Reproducible dataset of 100k policies and claim outcomes |
| **EDA Reports** | Visual exploration of key risk drivers |
| **Model Performance Tables** | Comparative results of GLM and ML models |
| **Business KPI Dashboards** | Portfolio metrics and pricing outcomes |
| **Reproducibility Artifacts** | Reserve dataset and stable random seeds |

---

## 🧩 Tools & Libraries
- **Core:** pandas, numpy, matplotlib, seaborn  
- **Modeling:** scikit-learn, statsmodels  
- **Environment:** Python 3.11+, `uv` for environment management  
- **Outputs:** CSV summaries and high-resolution figures under `outputs/`

---

## 🔁 Reproducibility & Execution Order
For full workflow reproducibility:

1. Run `01a_reference_data_exploration.ipynb`
2. Execute `simulate_data.py` (or use reserve dataset if unavailable)
3. Sequentially run Notebooks `02` → `06`
4. All results and visualizations are exported under `outputs/`

---

## ✅ Summary
This project provides a transparent, modular, and reproducible framework for **insurance portfolio risk modeling** —  
from synthetic data generation to business interpretation.  
It demonstrates actuarial principles, machine learning integration, and practical implementation standards for reproducible research.

---

**Author:** Golib Sanaev  
**Last updated:** 07 November 2025
