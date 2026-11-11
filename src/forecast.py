"""Weekly Superstore demand forecast used by the dashboard (and the notebook)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_FALLBACK = ROOT / "data" / "processed" / "orders.csv.gz"
HOLDOUT_WEEKS = 12
LAGS = (1, 4, 52)
ROLLS = (4, 12)
CALENDAR_COLS = ["t", "year", "weekofyear", "month", "quarter"]


def project_root() -> Path:
    return ROOT


def find_raw_file() -> Path:
    preferred = [
        RAW_DIR / "Sample - Superstore.csv",
        RAW_DIR / "superstore_final_dataset.csv",
        RAW_DIR / "Superstore.csv",
    ]
    extra = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls"))
    for path in preferred + extra:
        if path.exists() and path.name != ".gitkeep":
            return path
    if PROCESSED_FALLBACK.exists():
        return PROCESSED_FALLBACK
    raise FileNotFoundError(
        "Put the Superstore file in data/raw/ as 'Sample - Superstore.csv'.\n"
        "Download: https://www.kaggle.com/datasets/vivek468/superstore-dataset-final"
    )


def load_orders() -> pd.DataFrame:
    raw_path = find_raw_file()
    if raw_path.suffix.lower() in {".xls", ".xlsx"}:
        orders = pd.read_excel(raw_path)
    elif raw_path.suffix.lower() == ".gz":
        orders = pd.read_csv(raw_path, encoding="utf-8")
    else:
        orders = pd.read_csv(raw_path, encoding="latin-1")
    orders.columns = [c.strip() for c in orders.columns]
    keep = ["Order Date", "Sales", "Quantity", "Category", "Region", "Segment", "Postal Code"]
    missing = [c for c in keep if c not in orders.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
    if "Sub-Category" in orders.columns:
        keep.append("Sub-Category")
    # PROCESSED_FALLBACK is already cleaned & deduplicated against full raw columns.
    if raw_path != PROCESSED_FALLBACK:
        orders = orders.drop_duplicates()
    orders["Order Date"] = pd.to_datetime(orders["Order Date"], errors="coerce")
    orders = orders.dropna(subset=["Order Date", "Sales"])
    if orders["Postal Code"].isna().any():
        orders["Postal Code"] = orders["Postal Code"].fillna("05401")
    orders = orders[orders["Sales"] > 0]
    orders = orders[keep].copy()
    orders["Order Date"] = orders["Order Date"].dt.normalize()
    return orders


def weekly_series(orders: pd.DataFrame) -> pd.DataFrame:
    if orders.empty:
        raise ValueError("No orders in the current filter.")
    weekly = (
        orders.set_index("Order Date")
        .resample("W-SUN")
        .agg(sales=("Sales", "sum"), quantity=("Quantity", "sum"), orders=("Sales", "size"))
    )
    full_idx = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-SUN")
    weekly = weekly.reindex(full_idx, fill_value=0)
    weekly.index.name = "week_end"
    return weekly.reset_index()


def add_time_features(frame: pd.DataFrame, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    out = frame.copy()
    out["week_end"] = pd.to_datetime(out["week_end"])
    out["t"] = np.arange(len(out))
    out["year"] = out["week_end"].dt.year
    out["weekofyear"] = out["week_end"].dt.isocalendar().week.astype(int)
    out["month"] = out["week_end"].dt.month
    out["quarter"] = out["week_end"].dt.quarter
    for lag in lags:
        out[f"lag_{lag}"] = out["sales"].shift(lag)
    for win in ROLLS:
        out[f"roll_{win}"] = out["sales"].shift(1).rolling(win).mean()
    return out


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def _metrics(name, y_true, y_pred) -> dict:
    return {
        "model": name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE_%": mape(y_true, y_pred),
    }


def _feature_cols(lags: tuple[int, ...]) -> list[str]:
    return [*CALENDAR_COLS, *[f"lag_{lag}" for lag in lags], *[f"roll_{w}" for w in ROLLS]]


def _ridge():
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=2.0))])


def _forest():
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=1)


def _walk_forward(weekly: pd.DataFrame, model, horizon: int, lags: tuple[int, ...], cols: list[str]) -> pd.DataFrame:
    history = weekly.copy()
    rows = []
    for _ in range(horizon):
        next_end = history["week_end"].iloc[-1] + pd.Timedelta(weeks=1)
        trial = pd.concat(
            [history, pd.DataFrame({"week_end": [next_end], "sales": [np.nan], "quantity": [np.nan], "orders": [np.nan]})],
            ignore_index=True,
        )
        trial = add_time_features(trial, lags=lags)
        row = trial.iloc[-1]
        x = row[cols].to_frame().T
        yhat = max(float(model.predict(x)[0]), 0.0)
        rows.append({"week_end": next_end, "forecast": yhat})
        history = pd.concat(
            [history, pd.DataFrame({"week_end": [next_end], "sales": [yhat], "quantity": [np.nan], "orders": [np.nan]})],
            ignore_index=True,
        )
    return pd.DataFrame(rows)


@dataclass
class ForecastResult:
    weekly: pd.DataFrame
    train: pd.DataFrame
    test: pd.DataFrame
    scoreboard: pd.DataFrame
    best_name: str
    chosen_name: str
    holdout: pd.DataFrame
    future: pd.DataFrame
    month_avg: pd.DataFrame
    mix_category: pd.DataFrame
    mix_region: pd.DataFrame
    mix_segment: pd.DataFrame
    forecast_sum: float
    last_8_sum: float
    yoy_sum: float
    typical_miss: float
    n_orders: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp


def run_forecast(
    orders: pd.DataFrame,
    horizon: int = 8,
    model_choice: str = "Best on holdout",
) -> ForecastResult:
    weekly = weekly_series(orders)
    n_weeks = len(weekly)
    lags = tuple(lag for lag in LAGS if n_weeks > lag + HOLDOUT_WEEKS + 8)
    if 1 not in lags:
        lags = (1,) + tuple(lag for lag in lags if lag != 1)
    cols = _feature_cols(lags)
    featured = add_time_features(weekly, lags=lags)
    model_ready = featured.dropna().reset_index(drop=True)
    if len(model_ready) <= HOLDOUT_WEEKS + 8:
        raise ValueError("Not enough weekly history in this filter to train a forecast. Widen the filters.")

    split = len(model_ready) - HOLDOUT_WEEKS
    train = model_ready.iloc[:split].copy()
    test = model_ready.iloc[split:].copy()
    X_train, y_train = train[cols], train["sales"]
    y_true = test["sales"]

    naive_ma = np.full(len(test), train["sales"].iloc[-4:].mean())
    seasonal = test["lag_52"].to_numpy() if "lag_52" in test.columns else naive_ma

    ridge = _ridge()
    ridge.fit(X_train, y_train)
    ridge_hat = ridge.predict(test[cols])

    rf = _forest()
    rf.fit(X_train, y_train)
    rf_hat = rf.predict(test[cols])

    rows = [
        _metrics("Seasonal naive (same week last year)", y_true, seasonal),
        _metrics("4-week moving average", y_true, naive_ma),
        _metrics("Ridge (time features + lags)", y_true, ridge_hat),
        _metrics("Random Forest", y_true, rf_hat),
    ]
    scoreboard = pd.DataFrame(rows).set_index("model")
    best_name = scoreboard["MAE"].idxmin()

    pred_map = {
        "Seasonal naive (same week last year)": seasonal,
        "4-week moving average": naive_ma,
        "Ridge (time features + lags)": ridge_hat,
        "Random Forest": rf_hat,
        "Best on holdout": None,
    }
    chosen_name = best_name if model_choice == "Best on holdout" else model_choice
    best_hat = pred_map[chosen_name]
    holdout = test[["week_end", "sales"]].copy()
    holdout["predicted"] = best_hat
    holdout["error"] = holdout["sales"] - holdout["predicted"]

    use_rf = chosen_name == "Random Forest"
    final_model = _forest() if use_rf else _ridge()
    if chosen_name not in {"Random Forest", "Ridge (time features + lags)"}:
        final_model = _forest() if best_name == "Random Forest" else _ridge()
        chosen_name = best_name
    final_model.fit(model_ready[cols], model_ready["sales"])

    future = _walk_forward(weekly, final_model, horizon, lags, cols)
    typical_miss = float(scoreboard.loc[chosen_name, "MAE"])
    future["lo"] = (future["forecast"] - typical_miss).clip(lower=0)
    future["hi"] = future["forecast"] + typical_miss

    month_avg = (
        weekly.assign(month=lambda d: d["week_end"].dt.month)
        .groupby("month", as_index=False)["sales"]
        .mean()
    )
    mix_category = orders.groupby("Category", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    mix_region = orders.groupby("Region", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)
    mix_segment = orders.groupby("Segment", as_index=False)["Sales"].sum().sort_values("Sales", ascending=False)

    last_8 = float(weekly.tail(8)["sales"].sum())
    forecast_sum = float(future["forecast"].sum())
    yoy_start = future["week_end"].min() - pd.DateOffset(years=1)
    yoy_end = future["week_end"].max() - pd.DateOffset(years=1)
    yoy = float(weekly[(weekly["week_end"] >= yoy_start) & (weekly["week_end"] <= yoy_end)]["sales"].sum())

    return ForecastResult(
        weekly=weekly,
        train=train,
        test=test,
        scoreboard=scoreboard,
        best_name=best_name,
        chosen_name=chosen_name,
        holdout=holdout,
        future=future,
        month_avg=month_avg,
        mix_category=mix_category,
        mix_region=mix_region,
        mix_segment=mix_segment,
        forecast_sum=forecast_sum,
        last_8_sum=last_8,
        yoy_sum=yoy,
        typical_miss=typical_miss,
        n_orders=int(len(orders)),
        date_min=orders["Order Date"].min(),
        date_max=orders["Order Date"].max(),
    )


def planning_copy(result: ForecastResult) -> str:
    start = result.future["week_end"].min().strftime("%b %d, %Y")
    end = result.future["week_end"].max().strftime("%b %d, %Y")
    vs_hol = (result.forecast_sum - result.last_8_sum) / result.last_8_sum * 100 if result.last_8_sum else 0
    vs_yoy = (result.forecast_sum - result.yoy_sum) / result.yoy_sum * 100 if result.yoy_sum else 0
    mape_val = float(result.scoreboard.loc[result.chosen_name, "MAPE_%"])
    return (
        f"Plan for **${result.forecast_sum:,.0f}** between {start} and {end}. "
        f"That is **{vs_hol:+.1f}%** vs the last 8 actual weeks and **{vs_yoy:+.1f}%** vs the same weeks last year. "
        f"Do not restock at holiday-peak quantities if this window is a softer season; keep a cash and inventory buffer of about "
        f"**${result.typical_miss:,.0f} per week** (the model’s typical miss, MAPE {mape_val:.1f}%). "
        f"Staff below November–December headcount unless the forecast weeks sit in the peak months."
    )
