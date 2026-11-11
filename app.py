"""
Interactive Sales Forecasting Dashboard — Streamlit Edition
============================================================
Run locally:  streamlit run app.py
Streamlit Cloud entrypoint: app.py
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.charts import (
    make_forecast_chart,
    make_heatmap,
    make_holdout_chart,
    make_pie,
    make_residuals_chart,
    make_scatter_chart,
    make_seasonality_chart,
    make_treemap,
    make_yoy_chart,
)
from src.forecast import load_orders, planning_copy, run_forecast

st.set_page_config(
    page_title="Superstore Sales Forecast Dashboard",
    page_icon="📊",
    layout="wide",
)

MODEL_OPTIONS = [
    "Best on holdout",
    "Random Forest",
    "Ridge (time features + lags)",
    "Seasonal naive (same week last year)",
    "4-week moving average",
]


@st.cache_data(show_spinner=False)
def get_orders() -> pd.DataFrame:
    """Load and cache the cleaned Superstore order history."""
    return load_orders()


@st.cache_data(show_spinner=False)
def compute_forecast(
    categories: tuple[str, ...],
    regions: tuple[str, ...],
    segments: tuple[str, ...],
    start_date: str,
    end_date: str,
    horizon: int,
    model_choice: str,
):
    """Filter orders and run the weekly forecasting pipeline."""
    df = get_orders()
    if categories:
        df = df[df["Category"].isin(categories)]
    if regions:
        df = df[df["Region"].isin(regions)]
    if segments:
        df = df[df["Segment"].isin(segments)]
    df = df[(df["Order Date"] >= pd.Timestamp(start_date)) & (df["Order Date"] <= pd.Timestamp(end_date))]
    if len(df) < 100:
        return df, None, "Not enough orders in this filter (need at least 100). Widen your selection."
    try:
        result = run_forecast(df, horizon=horizon, model_choice=model_choice)
    except Exception as exc:
        return df, None, str(exc)
    return df, result, None


all_orders = get_orders()
categories_all = sorted(all_orders["Category"].unique())
regions_all = sorted(all_orders["Region"].unique())
segments_all = sorted(all_orders["Segment"].unique())
min_date = all_orders["Order Date"].min().date()
max_date = all_orders["Order Date"].max().date()

# ── Sidebar Filters ──────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")
sel_categories = st.sidebar.multiselect("Category", categories_all, default=categories_all)
sel_regions = st.sidebar.multiselect("Region", regions_all, default=regions_all)
sel_segments = st.sidebar.multiselect("Segment", segments_all, default=segments_all)
date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
horizon = st.sidebar.slider("Forecast Horizon (weeks)", min_value=4, max_value=16, value=8, step=2)
model_choice = st.sidebar.selectbox("Forecast Model", MODEL_OPTIONS, index=0)

if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
    start_d, end_d = date_range
else:
    start_d, end_d = min_date, max_date

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Superstore Sales Forecast Dashboard")
st.caption("Interactive ML demand forecasting • Filter by category, region, segment, dates & horizon")

filtered_df, result, err = compute_forecast(
    tuple(sel_categories),
    tuple(sel_regions),
    tuple(sel_segments),
    str(start_d),
    str(end_d),
    horizon,
    model_choice,
)

if err or result is None:
    st.warning(f"⚠️ {err}")
    st.stop()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_overview, tab_deep, tab_model = st.tabs(["📊 Overview", "🔬 Deep Dive", "🏆 Model Lab"])

with tab_overview:
    vs_last8 = (result.forecast_sum - result.last_8_sum) / result.last_8_sum * 100 if result.last_8_sum else 0
    vs_yoy = (result.forecast_sum - result.yoy_sum) / result.yoy_sum * 100 if result.yoy_sum else 0
    avg_weekly = result.forecast_sum / horizon

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔮 Forecast Total", f"${result.forecast_sum:,.0f}", f"{vs_last8:+.1f}% vs last 8 wks")
    c2.metric("📊 Avg Weekly", f"${avg_weekly:,.0f}", f"± ${result.typical_miss:,.0f} MAE", delta_color="off")
    c3.metric("📈 Year-over-Year", f"${result.yoy_sum:,.0f}", f"{vs_yoy:+.1f}% change")
    c4.metric("📦 Total Orders", f"{result.n_orders:,}", f"{result.date_min:%b %Y} – {result.date_max:%b %Y}", delta_color="off")

    st.subheader("📈 Weekly Sales & Forecast")
    st.plotly_chart(make_forecast_chart(result))

    st.info(f"💡 **Planning Insight:** {planning_copy(result)}")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.subheader("📅 Monthly Seasonality")
        st.plotly_chart(make_seasonality_chart(result))
    with r2:
        st.subheader("🏷️ By Category")
        st.plotly_chart(make_pie(result.mix_category, "Category", "Sales", "Sales by Category"))
    with r3:
        st.subheader("🌎 By Region")
        st.plotly_chart(make_pie(result.mix_region, "Region", "Sales", "Sales by Region"))

    b1, b2 = st.columns([1, 2])
    with b1:
        st.subheader("👥 By Segment")
        st.plotly_chart(make_pie(result.mix_segment, "Segment", "Sales", "Sales by Segment"))
    with b2:
        st.subheader("📋 Forecast Table")
        export_df = result.future.copy()
        export_df["week_end"] = export_df["week_end"].dt.strftime("%Y-%m-%d")

        display_df = result.future.copy()
        display_df["Week"] = display_df["week_end"].dt.strftime("%b %d, %Y")
        display_df["Forecast"] = display_df["forecast"].map(lambda x: f"${x:,.0f}")
        display_df["Low"] = display_df["lo"].map(lambda x: f"${x:,.0f}")
        display_df["High"] = display_df["hi"].map(lambda x: f"${x:,.0f}")
        st.dataframe(display_df[["Week", "Forecast", "Low", "High"]], hide_index=True)

        st.download_button(
            "⬇ Download Forecast CSV",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name="superstore_forecast.csv",
            mime="text/csv",
        )

with tab_deep:
    st.subheader("📆 Year-over-Year Weekly Comparison")
    st.plotly_chart(make_yoy_chart(result))

    st.subheader("🌡️ Sales Heatmap (Year × Week)")
    st.plotly_chart(make_heatmap(result))

    st.subheader("🌳 Sales Treemap (Category → Sub-Category)")
    st.plotly_chart(make_treemap(filtered_df))

with tab_model:
    m1, m2 = st.columns([7, 5])
    with m1:
        st.subheader("🎯 Holdout: Actual vs Predicted")
        st.plotly_chart(make_holdout_chart(result))
    with m2:
        st.subheader("🏆 Model Scoreboard")
        sb = result.scoreboard.reset_index().copy()
        sb["Selected"] = sb["model"].map(lambda m: "✅" if m == result.chosen_name else "")
        sb["MAE"] = sb["MAE"].map(lambda x: f"${x:,.0f}")
        sb["RMSE"] = sb["RMSE"].map(lambda x: f"${x:,.0f}")
        sb["MAPE_%"] = sb["MAPE_%"].map(lambda x: f"{x:.1f}%")
        st.dataframe(
            sb[["Selected", "model", "MAE", "RMSE", "MAPE_%"]].rename(columns={"model": "Model", "MAPE_%": "MAPE"}),
            hide_index=True,
        )

    m3, m4 = st.columns(2)
    with m3:
        st.subheader("📊 Residual Distribution")
        st.plotly_chart(make_residuals_chart(result))
    with m4:
        st.subheader("📐 Predicted vs Actual Scatter")
        st.plotly_chart(make_scatter_chart(result))
