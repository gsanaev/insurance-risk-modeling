# 🧱 Project Setup & Environment Configuration  
> **Version 1.2 — November 2025**

## 📋 Purpose  
This document provides a **complete setup guide** for reproducing and running the **Insurance Risk Modeling – Frequency, Severity & Pure Premium Analysis** project.  
It describes all dependencies, virtual environment configuration, and recommended practices for local or cloud execution.

---

## ⚙️ 1. Environment Overview  

### 🧩 Core Components
| Category | Tools / Packages | Description |
|-----------|------------------|--------------|
| **Language** | Python ≥ 3.10 | Core scripting and analysis |
| **Data Handling** | pandas, numpy | Structured and numerical computation |
| **Modeling** | scikit-learn, statsmodels | GLM and ML modeling framework |
| **Visualization** | matplotlib, seaborn, plotly | Static & interactive data visualization |
| **Dashboard (optional)** | Streamlit | Interactive portfolio and KPI dashboard |
| **Environment** | uv / venv / conda | Virtual environment and dependency management |

---

## 🧮 2. Recommended Setup (via `uv`)  

### ✅ Step-by-Step

1. **Clone the repository**
   ```bash
   git clone https://github.com/gsanaev/insurance-risk-modeling.git
   cd insurance-risk-modeling
   ```

2. **Synchronize the environment**  
   ```bash
   uv sync
   ```  
   This command automatically creates a virtual environment and installs all dependencies listed in `pyproject.toml` or `requirements.txt`.
   It ensures **complete reproducibility** of the environment used in this project.

3. **(Optional) Activate the environment manually**
   ```bash
   source .venv/bin/activate  # (Linux / macOS)
   .venv\Scripts\activate     # (Windows)
   ```

4. **Verify installation**
   ```bash
   python --version
   python -m pip show pandas scikit-learn statsmodels
   ```

5. **Execution order (recommended for reproducibility)**  
   Run the following scripts and notebooks in sequence:
   ```bash
   # 1️⃣ Explore reference data
   jupyter notebook notebooks/01a_reference_data_exploration.ipynb

   # 2️⃣ Simulate synthetic insurance portfolio
   uv run -m src.ins.simulate_data

   # 3️⃣ Proceed with analysis & modeling
   jupyter notebook notebooks/02_exploration.ipynb
   jupyter notebook notebooks/03_model_frequency.ipynb
   jupyter notebook notebooks/04_model_severity.ipynb
   jupyter notebook notebooks/05_combined_purepremium.ipynb
   jupyter notebook notebooks/06_business_evaluation.ipynb
   ```

> 💾 **Note:**  
> A pre-generated backup of the simulated portfolio is stored in:
> ```
> data/data_reserve/synthetic_insurance_portfolio_2025-11-07.csv
> ```
> You can use this file if data simulation (`simulate_data.py`) cannot be executed,  
> ensuring all notebooks remain reproducible and functional.

---

## 🧠 3. Folder Structure Summary  

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

## 🧰 4. Key Dependencies

| Package | Version | Purpose |
|----------|----------|----------|
| **pandas** | ≥ 2.0 | Data manipulation & tabular processing |
| **numpy** | ≥ 1.25 | Vectorized computation |
| **matplotlib** | ≥ 3.7 | Plotting and visualization |
| **seaborn** | ≥ 0.13 | Statistical data visualization |
| **scikit-learn** | ≥ 1.3 | Machine learning (Random Forest, Gradient Boosting) |
| **statsmodels** | ≥ 0.14 | GLM (Poisson, Gamma) modeling |
| **plotly** | ≥ 5.17 | Interactive plotting |
| **streamlit** | ≥ 1.38 | Dashboard interface |
| **jupyterlab** | ≥ 4.0 | Notebook interface |
| **shap** | ≥ 0.45 | Explainability tools |

---

## 💾 5. Reproducibility Guidelines  

- Use **fixed random seeds** in modeling notebooks for deterministic results.  
- Keep a consistent **folder hierarchy** when exporting reports and figures.  
- Version your results via Git commits and store model outputs under `outputs/reports`.  
- The **synthetic dataset** is reproducible using `simulate_data.py`.

---

## 🚀 6. Optional: Streamlit Dashboard Setup  

The Streamlit app provides an interactive visualization of portfolio KPIs and model outputs.

### Run locally
```bash
uv run streamlit run src/ins/app_dashboard.py
```

### Features
- Visualize claim frequency, severity, and pure premium  
- Filter by region, vehicle type, or usage  
- Compare model outputs interactively  
- Export segment summaries  

---

## 🧭 7. Validation Checks  

| Test | Command | Expected Result |
|------|----------|----------------|
| Verify data load | `python src/ins/simulate_data.py` | Creates `synthetic_insurance_portfolio.csv` |
| Check GLM import | `python -c "import statsmodels.api as sm"` | ✅ No error |
| Notebook execution | Run any notebook cell | ✅ Outputs appear without warnings |
| Dashboard run | `uv run streamlit run src/ins/app_dashboard.py` | ✅ Local app opens |

---

## 🔍 8. Troubleshooting  

| Issue | Likely Cause | Resolution |
|-------|---------------|------------|
| `ModuleNotFoundError` | Missing dependency | Reinstall: `uv pip install -r requirements.txt` |
| `FileNotFoundError: synthetic_insurance_portfolio.csv` | Data not simulated | Run: `python src/ins/simulate_data.py` |
| Streamlit not launching | Port conflict | Run: `streamlit run ... --server.port 8502` |
| Statsmodels convergence warnings | Model complexity | Adjust model formula or sample size |

---

## 📦 9. System Requirements  

| Resource | Minimum | Recommended |
|-----------|----------|-------------|
| CPU | Dual-core | Quad-core+ |
| RAM | 8 GB | 16 GB+ |
| Disk | 1 GB | 2 GB (with figures/reports) |
| OS | macOS / Linux / Windows | Any (Python ≥ 3.10 supported) |

---

## 🧾 10. References  

- [Python 3.12 Docs](https://docs.python.org/3.12/)  
- [scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [statsmodels GLM Reference](https://www.statsmodels.org/stable/glm.html)  
- [Streamlit Docs](https://docs.streamlit.io/)  
- [Plotly Express Reference](https://plotly.com/python/plotly-express/)  

---

## ✅ Summary  

- Environment and dependencies fully specified  
- Reproducibility ensured via virtual environments  
- Compatible with Linux, macOS, and Windows  
- Streamlit app optional for visualization  
