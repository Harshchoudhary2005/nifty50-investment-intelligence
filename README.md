# 📈 NIFTY-50 Investment Intelligence Platform

> **IIT Roorkee Finance Club — Open Projects 2026**  
> Data-Driven Investment Intelligence Using NIFTY-50 Market Data

---

## 🌐 Live Demo
[👉 Click here to open the app](https://your-app-name.streamlit.app)  
*(Replace with your Streamlit Cloud URL after deployment)*

---

## 📌 Project Overview

An AI-powered investment intelligence platform built on 21 years of NIFTY-50 historical market data (Jan 2000 – Apr 2021). The platform transforms raw OHLCV data into actionable investment insights across six modules.

### Features
| Module | Description |
|---|---|
| 🏠 Market Overview | NIFTY-50 index history, VIX, returns distribution, top performers |
| 🔍 Stock Analysis | Candlestick charts, RSI, MACD, Bollinger Bands, buy/sell signals |
| 💼 Portfolio Builder | Monte Carlo optimization for Conservative / Balanced / Aggressive profiles |
| ⚠️ Risk Assessment | Sharpe, Sortino, Max Drawdown per stock and portfolio |
| 🚨 Anomaly Detection | Z-score based market crash and volume spike detection |
| 📊 Sector Analysis | Sector cumulative returns, correlation, seasonality |

---

## 📁 Repository Structure

```
nifty50-investment-intelligence/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── notebooks/
│   ├── 01_eda_pipeline.ipynb       # Data loading, cleaning, EDA
│   ├── 02_feature_engineering.ipynb
│   ├── 03_predictor_model.ipynb
│   ├── 04_portfolio.ipynb
│   └── 05_risk_assessment.ipynb
│
└── data/
    └── processed/                  # Saved after running notebooks
        ├── master_clean.csv
        ├── returns_wide.csv
        ├── sector_stats.csv
        └── metadata.csv
```

---

## 🗂️ Datasets Used

1. **NIFTY-50 Stock Market Data**  
   https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data

2. **India Stock Data NSE 1990–2020**  
   https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020  
   *(Used only for NIFTY-50 index benchmark and India VIX)*

---

## ⚙️ Environment Setup

### Option A — Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/nifty50-investment-intelligence.git
cd nifty50-investment-intelligence

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add data files to root directory
# Place these files in the project root:
#   - NIFTY50_all.csv
#   - stock_metadata.csv
#   - Datasets/INDEX/NIFTY 50.csv
#   - Datasets/INDEX/INDIA VIX.csv

# 5. Run the app
streamlit run app.py
```

### Option B — Run on Google Colab

Open any notebook in `notebooks/` directly in Google Colab.  
Upload both dataset zips when prompted (first run only — then save to Google Drive).

---

## 🚀 Deploying on Streamlit Cloud

1. Push this repo to GitHub (must be public)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repo → set `app.py` as the main file
5. Add your data files via Streamlit Cloud file uploader or GitHub LFS
6. Click **Deploy**

---

## 📊 Key Results

| Metric | Value |
|---|---|
| Stocks Analysed | 49 |
| Date Range | 2000-01-03 → 2021-04-30 |
| Total Records | 235,192 |
| Conservative Portfolio Sharpe | 0.15 |
| Balanced Portfolio Sharpe | 0.93 |
| Aggressive Portfolio Sharpe | 0.93 |
| Anomaly Days Detected (2σ) | ~180 |
| Market Crashes Identified | Dot-com, 2008 GFC, COVID-19 |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Web App | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualizations | Plotly, Matplotlib, Seaborn |
| Portfolio Optimization | Monte Carlo Simulation |
| Technical Indicators | Custom implementation |
| Anomaly Detection | Rolling Z-Score |
| Statistical Analysis | SciPy |

---

## 📄 License
MIT License — free to use and modify.

---

## 👤 Author
**Harsh Choudhary**  
IIT Roorkee | Finance Club Open Projects 2026  
GitHub: [@Harshchoudhary2005](https://github.com/Harshchoudhary2005)
