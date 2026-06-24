# 🏥 InsureIQ — Insurance Charges Intelligence Platform

A Streamlit web app that predicts annual health insurance charges using a Linear Regression model, with interactive Plotly visualizations for model performance, data exploration, and prediction explainability.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F89939?logo=scikitlearn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Overview

InsureIQ trains a Linear Regression model on the classic *Medical Cost Personal Datasets* benchmark and wraps it in a dark-mode, glassmorphism-styled dashboard. Users can predict charges for a custom policyholder profile, explore the underlying dataset interactively, inspect model diagnostics, and run sensitivity analysis on individual features.

**Dataset features:** `age`, `sex`, `bmi`, `children`, `smoker`, `region`
**Target:** `charges` (USD/year)
**Model:** scikit-learn `LinearRegression` with `LabelEncoder` + `StandardScaler` preprocessing

---

## ✨ Features

| Page | What it does |
|---|---|
| 🏠 **Predict** | Input form for a policyholder profile → gauge chart prediction, feature-contribution waterfall, cost percentile donut, risk-threshold slider, and session prediction history with CSV export |
| 📊 **Data Explorer** | Filterable data table, 5 chart types (scatter/violin/box/histogram/strip), sunburst & treemap breakdowns, 3D scatter, parallel coordinates, correlation heatmap, and scatter matrix |
| 📈 **Model Performance** | Actual vs. predicted scatter, cumulative error distribution, residual analysis (with Q-Q plot), feature importance (bar + radar), error breakdown by segment, and interactive sensitivity analysis |
| ℹ️ **About** | Problem statement, model card, feature reference, and tech stack |

---

## 🛠️ Tech Stack

- **Frontend/App framework:** Streamlit 1.35+
- **ML:** scikit-learn 1.4+ (LinearRegression, LabelEncoder, StandardScaler)
- **Visualization:** Plotly 5.20+ (Express + Graph Objects), statsmodels (for OLS trendlines)
- **Data:** pandas 2.x, NumPy 1.26+

---

## 📁 Project Structure

```
insureiq/
├── app.py                              # Main application (single-file)
├── insurance-checkpoint ml-01.csv      # Training dataset (must match this exact filename)
├── requirements.txt
└── README.md
```

> ⚠️ **Important:** `app.py` looks for a CSV named exactly `insurance-checkpoint ml-01.csv` in the same folder. If that file is missing, the app does **not** error — it silently falls back to synthetic random data so the demo doesn't crash. If your metrics ever look unusually different from the model card below, check that the real CSV is present and correctly named.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repo
git clone https://github.com/<your-username>/insureiq.git
cd insureiq

# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` by default. The model trains automatically on first load (cached via `@st.cache_resource`, so it only retrains if the code changes or the cache is cleared).

---

## 📊 Model Card

| Property | Value |
|---|---|
| Algorithm | Linear Regression |
| Library | scikit-learn |
| Dataset rows | 1,338 |
| Features | age, sex, bmi, children, smoker, region |
| Target | charges (USD/year) |
| Train/Test split | 80 / 20 (`random_state=42`) |
| Preprocessing | `LabelEncoder` (sex, smoker, region) + `StandardScaler` |
| Test R² | ~0.78 |
| Test MAE | ~$4,187 |
| Test RMSE | ~$5,800 |

*Exact metrics are computed live on app load and shown in the sidebar and Model Performance page — the values above are reference figures from the original training run.*

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only. Predictions are estimates derived from a single linear model on a public benchmark dataset and **should not be used for actual insurance underwriting, pricing, or financial decisions**.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 👤 Author

**Om** — ML Engineer / Data Scientist
[LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
