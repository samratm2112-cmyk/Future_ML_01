"""
Interactive Sales Forecasting Dashboard — Superstore
=====================================================
Run:  python dashboard.py          (then open http://127.0.0.1:8050)

Features:
  • Multi-tab layout: Overview · Deep-Dive · Model Lab
  • Sidebar filters: Category, Region, Segment, Date range, Horizon slider
  • KPI cards with deltas
  • Forecast time-series with confidence band
  • Planning insight auto-copy
  • Holdout actual-vs-predicted chart
  • Model scoreboard table
  • Monthly seasonality bar chart
  • Category / Region / Segment donut charts
  • Year-over-Year weekly comparison chart
  • Weekly sales heatmap (year × week)
  • Category treemap
  • Exportable forecast table (CSV download)
  • Dark-mode toggle
"""

from __future__ import annotations

import dash
import dash_bootstrap_components as dbc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html

from src.charts import (
    ACCENT,
    BLUE,
    GREEN,
    ORANGE,
    RED,
    empty_fig,
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

# ── Theme & colours ──────────────────────────────────────────────────────────
LIGHT_THEME = dbc.themes.FLATLY
CARD_STYLE = {"boxShadow": "0 2px 8px rgba(0,0,0,0.08)", "borderRadius": "12px"}

# ── Data ─────────────────────────────────────────────────────────────────────
all_orders = load_orders()
CATEGORIES = sorted(all_orders["Category"].unique())
REGIONS = sorted(all_orders["Region"].unique())
SEGMENTS = sorted(all_orders["Segment"].unique())
MIN_DATE = all_orders["Order Date"].min()
MAX_DATE = all_orders["Order Date"].max()

# ── Helpers ──────────────────────────────────────────────────────────────────

def kpi_card(title, value, delta="", icon="💰"):
    delta_color = GREEN if delta.startswith("+") else RED if delta.startswith("-") else ACCENT
    return dbc.Card(
        dbc.CardBody([
            html.Div(icon, style={"fontSize": "28px", "marginBottom": "4px"}),
            html.P(title, className="text-muted mb-1", style={"fontSize": "0.82rem", "fontWeight": 500}),
            html.H4(value, className="mb-0", style={"fontWeight": 700, "color": ACCENT}),
            html.Small(delta, style={"color": delta_color, "fontWeight": 600}) if delta else html.Span(),
        ], className="text-center py-3"),
        style=CARD_STYLE, className="h-100",
    )


def make_scoreboard_table(result):
    sb = result.scoreboard.reset_index()
    sb["MAE"] = sb["MAE"].map(lambda x: f"${x:,.0f}")
    sb["RMSE"] = sb["RMSE"].map(lambda x: f"${x:,.0f}")
    sb["MAPE_%"] = sb["MAPE_%"].map(lambda x: f"{x:.1f}%")
    best = result.chosen_name
    rows = []
    for _, r in sb.iterrows():
        is_best = r["model"] == best
        style = {"fontWeight": 700, "backgroundColor": "#eaf6ff"} if is_best else {}
        rows.append(html.Tr([
            html.Td(("✅ " if is_best else "") + r["model"], style=style),
            html.Td(r["MAE"], style=style),
            html.Td(r["RMSE"], style=style),
            html.Td(r["MAPE_%"], style=style),
        ]))
    return dbc.Table(
        [html.Thead(html.Tr([html.Th("Model"), html.Th("MAE"), html.Th("RMSE"), html.Th("MAPE")])),
         html.Tbody(rows)],
        bordered=True, hover=True, responsive=True, size="sm", className="mb-0")


# ── Layout ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[LIGHT_THEME],
    title="Superstore Forecast Dashboard",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)
server = app.server

# --- Sidebar ---
sidebar = dbc.Card([dbc.CardBody([
    html.H5("🔍 Filters", className="mb-3", style={"fontWeight": 700}),

    dbc.Label("Category", className="fw-bold small"),
    dcc.Dropdown(id="filter-category", options=[{"label": c, "value": c} for c in CATEGORIES],
                 value=CATEGORIES, multi=True, placeholder="All categories"),
    html.Hr(className="my-2"),

    dbc.Label("Region", className="fw-bold small"),
    dcc.Dropdown(id="filter-region", options=[{"label": r, "value": r} for r in REGIONS],
                 value=REGIONS, multi=True, placeholder="All regions"),
    html.Hr(className="my-2"),

    dbc.Label("Segment", className="fw-bold small"),
    dcc.Dropdown(id="filter-segment", options=[{"label": s, "value": s} for s in SEGMENTS],
                 value=SEGMENTS, multi=True, placeholder="All segments"),
    html.Hr(className="my-2"),

    dbc.Label("Date Range", className="fw-bold small"),
    dcc.DatePickerRange(id="filter-dates", min_date_allowed=MIN_DATE, max_date_allowed=MAX_DATE,
                        start_date=MIN_DATE, end_date=MAX_DATE,
                        display_format="MMM D, YYYY", className="w-100"),
    html.Hr(className="my-2"),

    dbc.Label("Forecast Horizon (weeks)", className="fw-bold small"),
    dcc.Slider(id="filter-horizon", min=4, max=16, step=2, value=8,
               marks={i: str(i) for i in range(4, 18, 2)}),
    html.Hr(className="my-2"),

    dbc.Button("Update Dashboard", id="btn-update", color="primary", className="w-100 mt-2", n_clicks=0),

    html.Hr(className="my-3"),
    html.Div([
        dbc.Label("Appearance", className="fw-bold small"),
        dbc.Switch(id="dark-mode-switch", label="Dark Mode", value=False, className="mt-1"),
    ]),
])], style={**CARD_STYLE, "position": "sticky", "top": "20px"})

# --- Tab: Overview ---
tab_overview = dbc.Tab(label="📊 Overview", tab_id="tab-overview", children=[
    html.Div(id="kpi-row", className="mt-3"),

    dbc.Card([
        dbc.CardHeader(html.H6("📈 Weekly Sales & Forecast", className="mb-0 fw-bold")),
        dbc.CardBody(dcc.Graph(id="chart-forecast", config={"displayModeBar": False})),
    ], style=CARD_STYLE, className="mb-3"),

    dbc.Card([
        dbc.CardBody(html.Div(id="planning-copy", className="mb-0")),
    ], style={**CARD_STYLE, "borderLeft": f"4px solid {ORANGE}"}, className="mb-3"),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("📅 Monthly Seasonality", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-season", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=4, className="mb-3"),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("🏷️ By Category", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-cat", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=4, className="mb-3"),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("🌎 By Region", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-region", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=4, className="mb-3"),
    ]),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("👥 By Segment", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-segment", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=4, className="mb-3"),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Div([
                html.H6("📋 Forecast Table", className="mb-0 fw-bold d-inline"),
                dbc.Button("⬇ Download CSV", id="btn-download", color="outline-primary",
                           size="sm", className="float-end"),
            ])),
            dbc.CardBody(html.Div(id="forecast-table")),
            dcc.Download(id="download-csv"),
        ], style=CARD_STYLE, className="h-100"), lg=8, className="mb-3"),
    ]),
])

# --- Tab: Deep Dive ---
tab_deep = dbc.Tab(label="🔬 Deep Dive", tab_id="tab-deep", children=[
    html.Div(className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("📆 Year-over-Year Weekly Comparison", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-yoy", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=12, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("🌡️ Sales Heatmap (Year × Week)", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-heatmap", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=12, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("🌳 Sales Treemap (Category → Sub-Category)", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-treemap", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=12, className="mb-3"),
    ]),
])

# --- Tab: Model Lab ---
tab_model = dbc.Tab(label="🏆 Model Lab", tab_id="tab-model", children=[
    html.Div(className="mt-3"),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("🎯 Holdout: Actual vs Predicted", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-holdout", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=7, className="mb-3"),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("🏆 Model Scoreboard", className="mb-0 fw-bold")),
            dbc.CardBody(html.Div(id="scoreboard")),
        ], style=CARD_STYLE, className="h-100"), lg=5, className="mb-3"),
    ]),
    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("📊 Residual Distribution", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-residuals", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=6, className="mb-3"),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.H6("📐 Predicted vs Actual Scatter", className="mb-0 fw-bold")),
            dbc.CardBody(dcc.Graph(id="chart-scatter", config={"displayModeBar": False})),
        ], style=CARD_STYLE), lg=6, className="mb-3"),
    ]),
])

# --- Full layout ---
app.layout = dbc.Container([
    dcc.Store(id="store-result", data=None),

    # Header row
    dbc.Row(dbc.Col(html.Div([
        html.Div([
            html.H2("📊 Superstore Sales Forecast Dashboard",
                     style={"fontWeight": 800, "color": ACCENT, "marginBottom": "4px"}),
            html.P("Interactive demand forecasting with ML • Filter, explore, and export",
                   className="text-muted mb-0", style={"fontSize": "0.9rem"}),
        ]),
    ], className="py-3 d-flex justify-content-between align-items-center"))),

    dbc.Row([
        # Sidebar
        dbc.Col(sidebar, lg=3, md=4, className="mb-4"),
        # Main
        dbc.Col([
            dbc.Tabs([tab_overview, tab_deep, tab_model],
                     id="main-tabs", active_tab="tab-overview", className="mb-3"),
        ], lg=9, md=8),
    ]),

    html.Hr(),
    html.P("Built with Plotly Dash • Data: Superstore 2014–2017 • Models: Random Forest · Ridge · Baselines",
           className="text-center text-muted small pb-3"),
    dcc.Loading(id="loading", type="circle", children=html.Div(id="loading-output"), fullscreen=True),
], fluid=True, className="px-4", id="main-container")


# ── Callbacks ────────────────────────────────────────────────────────────────

@callback(
    Output("kpi-row", "children"),
    Output("chart-forecast", "figure"),
    Output("chart-holdout", "figure"),
    Output("chart-season", "figure"),
    Output("chart-cat", "figure"),
    Output("chart-region", "figure"),
    Output("chart-segment", "figure"),
    Output("scoreboard", "children"),
    Output("planning-copy", "children"),
    Output("forecast-table", "children"),
    Output("chart-yoy", "figure"),
    Output("chart-heatmap", "figure"),
    Output("chart-treemap", "figure"),
    Output("chart-residuals", "figure"),
    Output("chart-scatter", "figure"),
    Output("store-result", "data"),
    Output("loading-output", "children"),
    Input("btn-update", "n_clicks"),
    State("filter-category", "value"),
    State("filter-region", "value"),
    State("filter-segment", "value"),
    State("filter-dates", "start_date"),
    State("filter-dates", "end_date"),
    State("filter-horizon", "value"),
)
def update_all(n_clicks, categories, regions, segments, start_date, end_date, horizon):
    # Apply filters
    df = all_orders.copy()
    if categories:
        df = df[df["Category"].isin(categories)]
    if regions:
        df = df[df["Region"].isin(regions)]
    if segments:
        df = df[df["Segment"].isin(segments)]
    if start_date:
        df = df[df["Order Date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["Order Date"] <= pd.Timestamp(end_date)]

    ef = empty_fig()
    fail = (dbc.Alert("⚠️ Not enough data. Widen filters.", color="warning"),
            ef, ef, ef, ef, ef, ef, html.P("—"), html.P("—"), html.P("—"),
            ef, ef, ef, ef, ef, None, "")
    if len(df) < 100:
        return fail
    try:
        result = run_forecast(df, horizon=horizon)
    except Exception as exc:
        return (dbc.Alert(f"⚠️ {exc}", color="danger"),
                ef, ef, ef, ef, ef, ef, html.P("—"), html.P("—"), html.P("—"),
                ef, ef, ef, ef, ef, None, "")

    # --- KPIs ---
    vs_last8 = (result.forecast_sum - result.last_8_sum) / result.last_8_sum * 100 if result.last_8_sum else 0
    vs_yoy = (result.forecast_sum - result.yoy_sum) / result.yoy_sum * 100 if result.yoy_sum else 0
    avg_weekly = result.forecast_sum / horizon
    mape_val = result.scoreboard.loc[result.chosen_name, "MAPE_%"]
    kpis = dbc.Row([
        dbc.Col(kpi_card("Forecast Total", f"${result.forecast_sum:,.0f}",
                         f"{vs_last8:+.1f}% vs last 8 wks", "🔮"), lg=3, md=6, className="mb-3"),
        dbc.Col(kpi_card("Avg Weekly", f"${avg_weekly:,.0f}",
                         f"± ${result.typical_miss:,.0f} MAE", "📊"), lg=3, md=6, className="mb-3"),
        dbc.Col(kpi_card("Year-over-Year", f"${result.yoy_sum:,.0f}",
                         f"{vs_yoy:+.1f}% growth", "📈"), lg=3, md=6, className="mb-3"),
        dbc.Col(kpi_card("Total Orders", f"{result.n_orders:,}",
                         f"{result.date_min:%b %Y} – {result.date_max:%b %Y}", "📦"), lg=3, md=6, className="mb-3"),
    ], className="mb-2")

    # --- Overview charts ---
    fig_fc = make_forecast_chart(result)
    fig_ho = make_holdout_chart(result)
    fig_se = make_seasonality_chart(result)
    fig_ca = make_pie(result.mix_category, "Category", "Sales", "Sales by Category")
    fig_re = make_pie(result.mix_region, "Region", "Sales", "Sales by Region")
    fig_sg = make_pie(result.mix_segment, "Segment", "Sales", "Sales by Segment")
    sb = make_scoreboard_table(result)

    plan = planning_copy(result)
    plan_div = dcc.Markdown(f"💡 **Planning Insight:** {plan}", style={"fontSize": "0.9rem"})

    # Forecast table
    ft = result.future.copy()
    ft_store = ft.copy()  # for CSV export
    ft_store["week_end"] = ft_store["week_end"].dt.strftime("%Y-%m-%d")
    ft["week_end"] = ft["week_end"].dt.strftime("%b %d, %Y")
    ft["forecast"] = ft["forecast"].map(lambda x: f"${x:,.0f}")
    ft["lo"] = ft["lo"].map(lambda x: f"${x:,.0f}")
    ft["hi"] = ft["hi"].map(lambda x: f"${x:,.0f}")
    ft = ft.rename(columns={"week_end": "Week", "forecast": "Forecast", "lo": "Low", "hi": "High"})
    fc_table = dbc.Table.from_dataframe(ft, striped=True, bordered=True, hover=True, size="sm")

    # --- Deep Dive charts ---
    fig_yoy = make_yoy_chart(result)
    fig_hm = make_heatmap(result)
    fig_tm = make_treemap(df)

    # --- Model Lab charts ---
    fig_res = make_residuals_chart(result)
    fig_scat = make_scatter_chart(result)

    return (kpis, fig_fc, fig_ho, fig_se, fig_ca, fig_re, fig_sg,
            sb, plan_div, fc_table,
            fig_yoy, fig_hm, fig_tm,
            fig_res, fig_scat,
            ft_store.to_dict("records"), "")


# CSV download
@callback(
    Output("download-csv", "data"),
    Input("btn-download", "n_clicks"),
    State("store-result", "data"),
    prevent_initial_call=True,
)
def download_forecast(n, data):
    if not data:
        return dash.no_update
    df = pd.DataFrame(data)
    return dcc.send_data_frame(df.to_csv, "superstore_forecast.csv", index=False)


# Dark mode
@callback(
    Output("main-container", "style"),
    Input("dark-mode-switch", "value"),
)
def toggle_dark(dark):
    if dark:
        return {"backgroundColor": "#1a1a2e", "color": "#e0e0e0", "minHeight": "100vh"}
    return {"backgroundColor": "#f8f9fa", "color": "#2c3e50", "minHeight": "100vh"}


# ── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀  Dashboard starting at  http://127.0.0.1:8050\n")
    app.run(debug=True, host="127.0.0.1", port=8050)
