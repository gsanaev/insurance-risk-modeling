# 🚗 Insurance Risk Modeling – Frequency, Severity & Pure Premium Analysis  
> **Version 1.2 – November 2025**

## 📋 Overview
This project develops a **data-driven insurance risk modeling framework** for **auto insurance**, integrating **actuarial and machine learning methods** to estimate expected claim costs (*pure premiums*).  

The workflow simulates a **synthetic German insurance portfolio** and models:  
- **Claim frequency** (Poisson / tree-based ML)  
- **Claim severity** (Gamma / tree-based ML)  
- **Expected pure premium** as their product  

It provides an **end-to-end reproducible pipeline**, grounded in empirical reference data from **KBA**, **MiD**, and **GDV**.

---

## 🎯 Objectives
- Simulate a realistic **auto insurance dataset** (driver, vehicle, region, behavior)  
- Model **claim frequency** via Poisson GLM and ML approaches  
- Model **claim severity** via Gamma GLM and ML approaches  
- Combine both components to compute **expected pure premium**  
- Evaluate model accuracy and **business KPIs** (loss ratio, Gini, calibration)  
- Ensure results are **transparent, interpretable, and reproducible**  

---

## 🏗️ Methodological Framework

### 1. Data Simulation
Synthetic portfolio generated using:
- **Region-first stratified sampling** (KBA vehicle stock)
- Policy-level features: age, mileage, region, garage, usage, etc.  
- **Poisson process** for claim counts  
- **Gamma process** for claim severities  

### 1.1. Data Reserve
> The file `data/data_reserve/synthetic_insurance_portfolio_2025-11-07.csv` is **version-controlled** to guarantee project reproducibility even if data generation cannot be re-run.  
> Use this reserve dataset as a fallback when `simulate_data.py` cannot be executed —  
> all analytical notebooks will remain fully functional and consistent with the reference setup.

### 2. Modeling Components

| Component | Method | Description |
|------------|---------|-------------|
| **Frequency** | Poisson GLM / Random Forest | Predicts claim count per policy |
| **Severity** | Gamma GLM / Gradient Boosting | Predicts average claim cost per claim |
| **Pure Premium** | Frequency × Severity | Expected total claim cost per policy |

### 3. Evaluation Metrics

| Type | Metric | Purpose |
|------|---------|----------|
| Statistical | RMSE, MAE, Deviance | Model accuracy |
| Business | Gini, Loss Ratio, Calibration | Portfolio risk differentiation |
| Explainability | SHAP, Partial Dependence | Model interpretation |

---

## 🧩 Project Structure

```
insurance-risk-modeling/
├── data/
│   ├── data_reserve/       # synthetic_insurance_portfolio_2025-11-07.csv          
│   ├── raw/                # synthetic_insurance_portfolio.csv
│   └── reference/          # KBA, MiD, GDV reference datasets
│
├── notebooks/              # analysis and modeling steps
│   ├── 01a_reference_data_exploration.ipynb
│   ├── 01b_data_simulation_validation.ipynb
│   ├── 02_exploration.ipynb
│   ├── 03_model_frequency.ipynb
│   ├── 04_model_severity.ipynb
│   ├── 05_combined_purepremium.ipynb
│   └── 06_business_evaluation.ipynb
│
├── src/ins/                # reproducible app and simulation scripts
│   ├── simulate_data.py
│   └── app_dashboard.py
│
├── outputs/                # figures, model summaries, reports
│   ├── figures/
│   └── reports/
│
├── docs/                   # documentation and review artifacts
│   ├── notebooks_html/
│   └── PROJECT_REVIEW.md
│
├── requirements.txt
├── LICENSE
├── README.md
└── PROJECT_SETUP.md
```

---

## ⚙️ Tools & Libraries

- **Core:** Python, pandas, numpy, scikit-learn  
- **Statistical Modeling:** statsmodels (GLM)  
- **Machine Learning:** RandomForest, GradientBoosting, XGBoost  
- **Visualization:** matplotlib, seaborn, plotly  
- **Interpretability:** SHAP, PDP  
- **Deployment (optional):** Streamlit dashboard  

---

## 🧮 Workflow Summary

| Step | Notebook | Description |
|------|-----------|-------------|
| 1 | 01a_reference_data_exploration.ipynb | Explore MiD, KBA, GDV reference data |
| 2 | 01b_data_simulation_validation.ipynb | Generate & validate synthetic portfolio |
| 3 | 03_model_frequency.ipynb | Model claim frequency (Poisson GLM / RF) |
| 4 | 04_model_severity.ipynb | Model claim severity (Gamma GLM / GBM) |
| 5 | 05_combined_purepremium.ipynb | Compute expected pure premium |
| 6 | 06_business_evaluation.ipynb | Evaluate KPIs, Gini, Lorenz, and pricing |

---

## 📊 Validation & Calibration Targets

| Metric | Target Range | Typical Result |
|---------|---------------|----------------|
| Claim frequency | 0.07–0.09 | ✅ 0.08 |
| Mean severity | €2,200–€3,200 | ✅ €2,700 |
| Pure premium | €170–€260 | ✅ €214 |
| Gini (loss concentration) | 0.25–0.40 | ✅ 0.31 |

These indicators confirm that the synthetic dataset and models behave consistently with realistic insurance portfolios.

---

## 📈 Expected Outcomes

- **Synthetic, auditable dataset** representative of the German auto market  
- Comparison of **GLM** (interpretability) vs **ML** (predictive power)  
- Visualization of key risk factors and portfolio performance  
- **Business-ready KPIs** (Loss Ratio, Gini, Lorenz, pricing balance)  

---

## 🧭 Interactive Dashboard — Streamlit

A lightweight **Streamlit dashboard** complements the analytical notebooks, providing an interactive interface for:
- KPIs (claims, frequency, severity, premium)
- Segment analysis (region, vehicle type, density)
- Dynamic plots (Plotly)
- Filtered data export (CSV)

### ▶ Run locally

```bash
uv run streamlit run src/ins/app_dashboard.py
```

📁 **Path:** `src/ins/app_dashboard.py`  
🧰 **Stack:** Streamlit, Plotly, pandas, numpy  

---

## 📜 License
Licensed under the **MIT License** — free for use, modification, and distribution with attribution.

---

## 👤 Author
Developed by **Golib Sanaev**  
*Data Scientist | Applied Risk Analytics & Insurance Modeling*  

📧 **Email:** gsanaev@gmail.com  
🔗 **LinkedIn:** [golib-sanaev](https://linkedin.com/in/golib-sanaev)  
💻 **GitHub:** [@gsanaev](https://github.com/gsanaev)

---

## 📚 Citation
> Sanaev, G. (2025). *Insurance Risk Modeling – Frequency, Severity & Pure Premium Simulation (German Auto Market, 2023–2025).*  
> GitHub: [github.com/gsanaev/insurance-risk-modeling](https://github.com/gsanaev/insurance-risk-modeling)

---

## 🙏 Acknowledgements

- [StackFuel](https://stackfuel.com/) — applied data science education  
- [GDV](https://www.gdv.de/), [KBA](https://www.kba.de/), [MiD](https://www.mobilitaet-in-deutschland.de/) — empirical reference data  
- [Allianz SE](https://www.allianz.com/) — for actuarial practice alignment  
- [scikit-learn](https://scikit-learn.org/), [statsmodels](https://www.statsmodels.org/), [SHAP](https://github.com/shap/shap) — core modeling tools  
- [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [matplotlib](https://matplotlib.org/) — data & visualization foundations  
- **OpenAI GPT-5 Assistant** — documentation, automation & code review support  

⭐ *If you find this project useful, please give it a star!*  
