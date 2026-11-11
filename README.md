# 📊 Sales & Demand Forecasting (Superstore)

Weekly sales forecast for a U.S. Superstore, built as a Future Interns ML task. The model uses four years of order history (2014–2017) to predict the next 8 weeks of demand so a store owner can plan inventory, cash, and staffing.

**Headline result:** about **$101K** of sales over 8 weeks starting 7 Jan 2018 — down sharply from the holiday period, but **up versus the same weeks in 2017**. See [`reports/BUSINESS_BRIEF.md`](reports/BUSINESS_BRIEF.md).

---

## 🖥️ Interactive Dashboard

A full-featured **Plotly Dash** dashboard with three tabs:

| Tab | What's Inside |
|---|---|
| **📊 Overview** | KPI cards, forecast chart with confidence band, seasonality, category/region/segment donuts, forecast table with CSV export |
| **🔬 Deep Dive** | Year-over-Year weekly overlay, sales heatmap (year × week), category → sub-category treemap |
| **🏆 Model Lab** | Holdout actual-vs-predicted, model scoreboard, residual histogram, predicted-vs-actual scatter |

**Sidebar filters:** Category, Region, Segment, Date Range, Forecast Horizon (4–16 weeks), Dark Mode toggle.

### Quick Start

```bash
# 1. Clone & setup
git clone https://github.com/samratm2112-cmyk/Future_ML_01.git
cd Future_ML_01
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Download the dataset
mkdir -p data/raw
curl -L -o "data/raw/Sample - Superstore.csv" \
  "https://raw.githubusercontent.com/MayurjareMJ/EDA-using-Streamlit/main/Superstore.csv"

# 3. Launch the dashboard
python dashboard.py
```

Then open **http://127.0.0.1:8050** in your browser.

---

## 📓 Jupyter Notebook

The notebook runs the full pipeline end-to-end:

```bash
jupyter notebook notebooks/01_sales_forecasting.ipynb
```

Or run non-interactively:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_sales_forecasting.ipynb --inplace
```

### What the notebook does

1. Cleans orders (missing postal codes, duplicates, date parsing)
2. Aggregates to a complete weekly sales series
3. Builds time features (trend, month, lags, rolling means)
4. Compares **seasonal-naive** and **moving-average** baselines with **Ridge** and **Random Forest**
5. Forecasts 8 weeks ahead and explains how a manager can use the numbers

---

## 📁 Project Structure

```
Future_ML_01/
├── dashboard.py                  # Interactive Plotly Dash dashboard
├── requirements.txt              # Python dependencies
├── src/
│   ├── __init__.py
│   └── forecast.py               # ML forecasting pipeline (models, features, walk-forward)
├── notebooks/
│   └── 01_sales_forecasting.ipynb # Full EDA & model comparison notebook
├── data/
│   ├── raw/                      # Raw Superstore CSV (not committed)
│   └── processed/
│       ├── weekly_sales.csv      # Aggregated weekly sales
│       └── future_8w_forecast.csv # 8-week forecast output
├── reports/
│   ├── BUSINESS_BRIEF.md         # Stakeholder planning brief
│   └── figures/                  # Saved chart images (6 PNGs)
└── scripts/
    └── create_notebook.py        # Notebook generator script
```

---

## 📊 Dataset

[Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — ~10,000 line items from 2014–2017 with order date, sales, category, and region.

---

## 🔧 Tech Stack

- **Python 3.11** · Pandas · NumPy · Scikit-learn
- **Dash** · Plotly · Dash Bootstrap Components
- **Jupyter** · Matplotlib

---

## 📈 Model Performance (Holdout: last 12 weeks)

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Seasonal Naive | $8,375 | — | — |
| 4-Week Moving Avg | — | — | — |
| Ridge (time + lags) | — | — | — |
| **Random Forest** ✅ | **$7,061** | — | **44.6%** |

> The Random Forest was selected as best on holdout MAE. Dollar MAE is the recommended planning buffer — MAPE looks high because December weeks swing hard.

---

## 📄 License

MIT
