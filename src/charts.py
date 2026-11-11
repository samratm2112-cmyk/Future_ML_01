"""Shared Plotly figures used by both the Streamlit and Dash dashboards."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACCENT = "#2c3e50"
BLUE = "#3498db"
GREEN = "#27ae60"
RED = "#e74c3c"
ORANGE = "#f39c12"
PURPLE = "#8e44ad"
TEAL = "#1abc9c"

COLORS = {
    "Technology": "#3498db",
    "Furniture": "#e74c3c",
    "Office Supplies": "#2ecc71",
    "West": "#3498db",
    "East": "#e74c3c",
    "Central": "#f39c12",
    "South": "#2ecc71",
    "Consumer": "#3498db",
    "Corporate": "#e74c3c",
    "Home Office": "#f39c12",
}


def chart_layout(fig: go.Figure, height: int = 340, **kw) -> go.Figure:
    """Apply consistent Plotly white layout defaults."""
    defaults = dict(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=30, b=40),
    )
    defaults.update(kw)
    fig.update_layout(**defaults)
    return fig


def empty_fig(msg: str = "Not enough data — widen filters") -> go.Figure:
    """Return a blank placeholder figure with a centered annotation."""
    return go.Figure().update_layout(
        template="plotly_white",
        height=300,
        annotations=[dict(text=msg, showarrow=False, font=dict(size=16))],
    )


def make_forecast_chart(result) -> go.Figure:
    """Weekly historical sales with future forecast and +-MAE confidence band."""
    w, f = result.weekly.copy(), result.future.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=w["week_end"], y=w["sales"], mode="lines", name="Actual Sales",
        line=dict(color=BLUE, width=2),
        hovertemplate="Week: %{x|%b %d, %Y}<br>Sales: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=f["week_end"], y=f["forecast"], mode="lines+markers", name="Forecast",
        line=dict(color=ORANGE, width=3, dash="dot"), marker=dict(size=8, symbol="diamond"),
        hovertemplate="Week: %{x|%b %d, %Y}<br>Forecast: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([f["week_end"], f["week_end"][::-1]]),
        y=pd.concat([f["hi"], f["lo"][::-1]]),
        fill="toself", fillcolor="rgba(243,156,18,0.15)",
        line=dict(width=0), showlegend=True, name="± MAE band", hoverinfo="skip",
    ))
    return chart_layout(
        fig, 400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_tickprefix="$", yaxis_tickformat=",",
        hovermode="x unified", xaxis_title="", yaxis_title="Weekly Sales ($)",
    )


def make_holdout_chart(result) -> go.Figure:
    """Holdout actual vs predicted weekly sales."""
    h = result.holdout
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=h["week_end"], y=h["sales"], name="Actual",
        mode="lines+markers", line=dict(color=BLUE, width=2),
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=h["week_end"], y=h["predicted"], name="Predicted",
        mode="lines+markers", line=dict(color=ORANGE, width=2, dash="dash"),
        hovertemplate="$%{y:,.0f}<extra></extra>",
    ))
    return chart_layout(
        fig, 320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_tickprefix="$", yaxis_tickformat=",", xaxis_title="", yaxis_title="Sales ($)",
    )


def make_seasonality_chart(result) -> go.Figure:
    """Average weekly sales by calendar month with overall average marker."""
    m = result.month_avg.copy()
    names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    m["month_name"] = m["month"].map(lambda x: names[int(x) - 1])
    avg = m["sales"].mean()
    fig = go.Figure(go.Bar(
        x=m["month_name"], y=m["sales"],
        marker_color=[GREEN if s >= avg else BLUE for s in m["sales"]],
        hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_hline(
        y=avg, line_dash="dash", line_color=RED,
        annotation_text=f"Avg ${avg:,.0f}", annotation_position="top right",
    )
    return chart_layout(
        fig, 300,
        yaxis_tickprefix="$", yaxis_tickformat=",", xaxis_title="", yaxis_title="Avg Weekly Sales",
    )


def make_pie(df: pd.DataFrame, name_col: str, val_col: str, title: str) -> go.Figure:
    """Donut chart showing sales share across a categorical dimension."""
    colors = [COLORS.get(n, BLUE) for n in df[name_col]]
    fig = go.Figure(go.Pie(
        labels=df[name_col], values=df[val_col], hole=0.55, marker_colors=colors,
        textinfo="label+percent", textposition="outside",
        hovertemplate="%{label}: $%{value:,.0f} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white", height=290,
        margin=dict(l=10, r=10, t=35, b=10),
        title=dict(text=title, x=0.5, font=dict(size=14)), showlegend=False,
    )
    return fig


def make_yoy_chart(result) -> go.Figure:
    """Year-over-Year weekly sales overlay."""
    w = result.weekly.copy()
    w["year"] = w["week_end"].dt.year
    w["week_num"] = w["week_end"].dt.isocalendar().week.astype(int)
    years = sorted(w["year"].unique())
    palette = [BLUE, GREEN, ORANGE, RED, PURPLE, TEAL]
    fig = go.Figure()
    for i, yr in enumerate(years):
        sub = w[w["year"] == yr].sort_values("week_num")
        fig.add_trace(go.Scatter(
            x=sub["week_num"], y=sub["sales"],
            mode="lines", name=str(yr),
            line=dict(color=palette[i % len(palette)], width=2),
            hovertemplate=f"{yr} W%{{x}}: $%{{y:,.0f}}<extra></extra>",
        ))
    return chart_layout(
        fig, 340,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Week of Year", yaxis_title="Weekly Sales ($)",
        yaxis_tickprefix="$", yaxis_tickformat=",", hovermode="x unified",
    )


def make_heatmap(result) -> go.Figure:
    """Year x week-of-year weekly sales heatmap."""
    w = result.weekly.copy()
    w["year"] = w["week_end"].dt.year
    w["week_num"] = w["week_end"].dt.isocalendar().week.astype(int)
    pivot = w.pivot_table(index="year", columns="week_num", values="sales", aggfunc="sum")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"W{c}" for c in pivot.columns], y=[str(y) for y in pivot.index],
        colorscale="YlOrRd", hovertemplate="Year %{y}, %{x}<br>Sales: $%{z:,.0f}<extra></extra>",
        colorbar=dict(title="Sales", tickprefix="$", tickformat=","),
    ))
    return chart_layout(
        fig, 260,
        xaxis_title="Week of Year", yaxis_title="",
        yaxis=dict(autorange="reversed"),
    )


def make_treemap(orders_df: pd.DataFrame) -> go.Figure:
    """Category -> Sub-Category sales treemap."""
    if "Sub-Category" not in orders_df.columns:
        return empty_fig("Sub-Category column not available")
    agg = orders_df.groupby(["Category", "Sub-Category"], as_index=False)["Sales"].sum()
    labels, parents, values, colors = [], [], [], []
    cat_totals = agg.groupby("Category")["Sales"].sum().to_dict()
    for cat in sorted(cat_totals):
        labels.append(cat)
        parents.append("")
        values.append(cat_totals[cat])
        colors.append(COLORS.get(cat, BLUE))
    for _, row in agg.iterrows():
        labels.append(row["Sub-Category"])
        parents.append(row["Category"])
        values.append(row["Sales"])
        colors.append(COLORS.get(row["Category"], BLUE))
    fig = go.Figure(go.Treemap(
        labels=labels, parents=parents, values=values,
        marker_colors=colors,
        textinfo="label+value", texttemplate="%{label}<br>$%{value:,.0f}",
        hovertemplate="%{label}: $%{value:,.0f}<extra></extra>",
        branchvalues="total",
    ))
    return chart_layout(fig, 380, margin=dict(l=5, r=5, t=5, b=5))


def make_residuals_chart(result) -> go.Figure:
    """Histogram of holdout prediction residuals."""
    fig = go.Figure(go.Histogram(
        x=result.holdout["error"], nbinsx=15, marker_color=BLUE, opacity=0.8,
        hovertemplate="Error: $%{x:,.0f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color=RED)
    return chart_layout(
        fig, 300, xaxis_title="Prediction Error ($)", yaxis_title="Count",
        xaxis_tickprefix="$", xaxis_tickformat=",",
    )


def make_scatter_chart(result) -> go.Figure:
    """Predicted vs actual scatter plot for the holdout window."""
    h = result.holdout
    max_val = max(h["sales"].max(), h["predicted"].max()) * 1.1
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, max_val], y=[0, max_val], mode="lines", name="Perfect",
        line=dict(color="grey", dash="dash"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=h["sales"], y=h["predicted"], mode="markers", name="Weeks",
        marker=dict(color=BLUE, size=10, line=dict(width=1, color="white")),
        hovertemplate="Actual: $%{x:,.0f}<br>Predicted: $%{y:,.0f}<extra></extra>",
    ))
    return chart_layout(
        fig, 300, xaxis_title="Actual ($)", yaxis_title="Predicted ($)",
        xaxis_tickprefix="$", yaxis_tickprefix="$",
        xaxis_tickformat=",", yaxis_tickformat=",",
    )
