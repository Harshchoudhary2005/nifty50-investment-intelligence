# 📈 NIFTY-50 Investment Intelligence Platform

> **IIT Roorkee Finance Club — Open Projects 2026**
> Data-Driven Investment Intelligence Using NIFTY-50 Market Data

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit)](https://nifty50-intelligence.streamlit.app/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data)

---

## 🌐 Live Demo

### 👉 [nifty50-intelligence.streamlit.app](https://nifty50-intelligence.streamlit.app/)

---

## 📌 Overview

An end-to-end AI-powered investment intelligence platform built on **21 years of NIFTY-50 historical market data** (Jan 2000 – Apr 2021). The platform transforms raw OHLCV data into actionable investment insights — covering stock prediction, portfolio optimisation, risk assessment, anomaly detection, and explainable AI.

Built for the **IIT Roorkee Finance Club Open Projects 2026** competition.

---

## 🧩 Platform Modules

| Module | What It Does |
|---|---|
| 📊 **EDA & Market Overview** | NIFTY-50 index history (1990–2021), India VIX, return distributions, sector performance, correlation heatmap |
| 🔍 **Stock Analysis** | Candlestick charts, RSI, MACD, Bollinger Bands, ATR, Stochastic — per stock |
| 🤖 **Stock Predictor Engine** | XGBoost + Random Forest trained on 48 technical features, directional accuracy, SHAP explainability |
| 💼 **Portfolio Builder** | Monte Carlo efficient frontier, Conservative / Balanced / Aggressive / Risk Parity profiles |
| ⚠️ **Risk Assessment** | Sharpe, Sortino, Max Drawdown, Calmar, VaR, CVaR, Beta — per stock and per portfolio |
| 🚨 **Anomaly Detection** | Rolling Z-score based market crash and volume spike detection |
| 💡 **Investment Signals** | Composite buy/sell signal scores with plain-English explanations for all 49 stocks |

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Stocks Analysed | 49 |
| Date Range | 2000-01-03 → 2021-04-30 |
| Total Records | 235,192 |
| Features Engineered | 48 per stock (incl. India VIX) |
| Best Model RMSE (XGBoost) | 0.027196 |
| Avg Directional Accuracy | 50.34% |
| Conservative Portfolio Sharpe | 0.15 |
| Balanced Portfolio Sharpe | 0.93 |
| Aggressive Portfolio Sharpe | 0.93 |
| Best Stock Sharpe (SHREECEM) | 0.72 |
| Systemic Anomaly Days Detected | 18 |
| Market Crashes Identified | Dot-com (2000), GFC (2008), COVID-19 (2020) |

---

## 📁 Repository Structure

```
nifty50-investment-intelligence/
│
├── app.py                                      # Streamlit web application
├── requirements.txt                            # Python dependencies
├── README.md                                   # This file
│
├── NIFTY50_Investment_Intelligence.ipynb       # Full analysis notebook (Colab-ready)
│
└── data/
    ├── NIFTY50_all.csv                         # Combined dataset (all 50 stocks)
    ├── stock_metadata.csv                      # Company name, sector, symbol
    └── processed/                              # Generated after running notebook
        ├── model_performance_all_stocks.csv
        ├── risk_metrics_all_stocks.csv
        ├── investment_signals.csv
        ├── portfolio_conservative.csv
        ├── portfolio_balanced.csv
        ├── portfolio_aggressive.csv
        └── portfolio_risk_parity.csv
```

---

## 🗂️ Datasets used 

| Dataset | Source | Usage |
|---|---|---|
| NIFTY-50 Stock Market Data | [Kaggle](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data) | 50 individual stock CSVs, 2000–2021 |
| India Stock Data NSE 1990–2020 | [Kaggle](https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020) | NIFTY-50 index (1990–2021), India VIX, sector indices, G-Sec yield |

> **Note:** Raw zip files are not included due to size. Download from Kaggle and follow setup instructions below.

---

## ⚙️ Setup & Reproduce Results

### Option A — Google Colab (Recommended)

1. Open `NIFTY50_Investment_Intelligence.ipynb` in [Google Colab](https://colab.research.google.com)
2. Upload both Kaggle zip files to `/content/` when prompted
3. Run all cells top to bottom — takes ~10 minutes
4. All outputs save to `/content/outputs/`

### Option B — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Harshchoudhary2005/nifty50-investment-intelligence.git
cd nifty50-investment-intelligence

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw data zips in project root:
#    archive (3).zip  →  NIFTY-50 stock CSVs
#    archive (4).zip  →  NSE index + SCRIP data

# 5. Run the Streamlit app
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web App | Streamlit |
| ML Models | XGBoost, Random Forest, LightGBM |
| Explainability | SHAP |
| Technical Indicators | `ta` library + custom implementation |
| Portfolio Optimisation | SciPy (SLSQP) + Monte Carlo Simulation |
| Visualisations | Plotly, Matplotlib, Seaborn |
| Data Processing | Pandas, NumPy |
| Statistical Analysis | SciPy, Statsmodels |

---

## 🔬 Methodology

**Feature Engineering** — 48 features per stock including SMA/EMA (5/10/21/50/200), RSI, MACD, Bollinger Bands, ATR, Stochastic Oscillator, OBV, rolling volatility, momentum returns, and India VIX as a macro signal.

**Prediction** — XGBoost and Random Forest trained on a strict time-series split (train: pre-2020, test: 2020–2021). No data leakage. Evaluated on RMSE, MAE, R², and Directional Accuracy.

**Portfolio Optimisation** — Mean-Variance Optimisation via SciPy SLSQP with three objectives: minimum volatility (Conservative), maximum Sharpe (Balanced), and maximum return (Aggressive). Risk-free rate derived from NIFTY GS 10YR G-Sec yield.

**Risk Metrics** — Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Calmar Ratio, VaR (95%), CVaR (95%), Beta vs NIFTY-50 benchmark.

**Anomaly Detection** — Rolling Z-score (window=21) on daily returns. Days exceeding 3σ flagged as anomalies. Cross-stock correlation used to identify systemic vs idiosyncratic events.

---

## 📄 License

MIT License — free to use and modify with attribution.

---

## 👤 Author

**Harsh Choudhary**
IIT Roorkee | Finance Club Open Projects 2026
GitHub: [@Harshchoudhary2005](https://github.com/Harshchoudhary2005)
