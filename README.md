# Sales & Demand Forecasting (Superstore)

Weekly sales forecast for a U.S. Superstore, built as a Future Interns ML task. The model uses four years of order history (2014–2017) to predict the next 8 weeks of demand so a store owner can plan inventory, cash, and staffing.

**Headline result:** about **$101k** of sales over 8 weeks starting 7 Jan 2018 — down sharply from the holiday period, but **up versus the same weeks in 2017**. See [`reports/BUSINESS_BRIEF.md`](reports/BUSINESS_BRIEF.md).

## Dataset

[Superstore Sales](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — ~10,000 line items from 2014–2017 with order date, sales, category, and region.

Place the file at `data/raw/Sample - Superstore.csv` (not committed; download once):

```bash
curl -L -o "data/raw/Sample - Superstore.csv" \
  "https://raw.githubusercontent.com/MayurjareMJ/EDA-using-Streamlit/main/Superstore.csv"
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_sales_forecasting.ipynb
```

Or run the notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_sales_forecasting.ipynb --inplace
```

## What the notebook does

1. Cleans orders (missing postal codes, duplicates, date parsing).
2. Aggregates to a complete weekly sales series.
3. Builds time features (trend, month, lags, rolling means).
4. Compares seasonal-naive and moving-average baselines with Ridge and Random Forest.
5. Forecasts 8 weeks ahead and explains how a manager can use the numbers.

Processed weekly sales are written to `data/processed/weekly_sales.csv`. The stakeholder summary is in [`reports/BUSINESS_BRIEF.md`](reports/BUSINESS_BRIEF.md).
