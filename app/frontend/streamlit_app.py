import math
import os
import sys
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add parent so we can import from backend if needed (or use API only)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------- Config ----------
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000")
# LSTM is secondary: when False, always use ensemble (no model choice). When True, show Ensemble vs LSTM selector.
# Override via env: LSTM_ENABLED=true to enable.
LSTM_ENABLED = os.environ.get("LSTM_ENABLED", "false").lower() in ("true", "1", "yes")
# When LSTM_ENABLED is False, this model is used (ensemble).
DEFAULT_MODEL = "ensemble"
# Last week of actual data (week ending this date). No predictions for weeks on or before this — show actuals only.
LAST_ACTUAL_WEEK_END = pd.Timestamp("2025-09-05").date()
# Weather forecast limit: predictions only for current week and up to this many days ahead.
PREDICTION_FORECAST_DAYS_AHEAD = 14
STREAMLIT_THEME = {
    "primaryColor": "#0ea5e9",
    "backgroundColor": "#0f172a",
    "secondaryBackgroundColor": "#1e293b",
    "textColor": "#f1f5f9",
    "font": "sans serif",
}

# Display range: "median + half the gap" so planning isn't wasteful (not full 2× or raw upper).
def _capped_range(lower: float, median: float, upper: float) -> tuple[float, float]:
    """Return (display_lower, display_upper) using half the gap from median to lower/upper."""
    gap_lo = median - lower
    gap_hi = upper - median
    lo = median - 0.5 * gap_lo if gap_lo > 0 else lower
    hi = median + 0.5 * gap_hi if gap_hi > 0 else upper
    return (lo, hi)

ENSEMBLE_PLANNING_CAPTION = "Use **median** for planning; range shows model uncertainty, not a capacity target."

# ---------- Page config ----------
st.set_page_config(
    page_title="Disease Case Forecast | Sri Lanka",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Custom CSS for colorful cards and sections ----------
st.markdown("""
<style>
    /* Gradient header */
    .main-header {
        background: linear-gradient(135deg, #0ea5e9 0%, #8b5cf6 50%, #ec4899 100%);
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        font-size: 1.75rem;
        font-weight: 700;
        text-align: center;
        box-shadow: 0 4px 14px rgba(14, 165, 233, 0.4);
    }
    /* Metric cards - fixed same size for all three */
    .metric-card {
        background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
        padding: 1.25rem;
        border-radius: 12px;
        border-left: 4px solid #0ea5e9;
        margin: 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        height: 140px !important;
        min-height: 140px !important;
        display: flex !important;
        flex-direction: column;
        justify-content: center;
        box-sizing: border-box;
        min-width: 0;
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        line-height: 1.3;
        word-break: break-word;
        overflow-wrap: break-word;
    }
    .metric-card .metric-label {
        font-size: 0.95rem;
        font-weight: 600;
        display: block;
    }
    /* Summary row: equal-width columns and proper spacing */
    .summary-cards-row {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr);
        gap: 1.25rem;
        width: 100%;
        max-width: 100%;
    }
    .summary-cards-row-4 {
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr) minmax(0, 1fr);
    }
    .summary-cards-row-2 {
        grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
    }
    .metric-value-sm { font-size: 1.35rem !important; }
    .summary-cards-wrapper {
        width: 100%;
        margin-bottom: 1.75rem;
    }
    .summary-cards-row .metric-card {
        min-width: 0;
        overflow: hidden;
    }
    .metric-card.green { border-left-color: #22c55e; }
    .metric-card.orange { border-left-color: #f97316; }
    .metric-card.purple { border-left-color: #8b5cf6; }
    .metric-card.pink { border-left-color: #ec4899; }
    /* Section titles */
    .section-title {
        color: #0ea5e9;
        font-size: 1.25rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.25rem;
        border-bottom: 2px solid #334155;
    }
    .section-title-quickview {
        margin-bottom: 0.35rem !important;
    }
    /* Dataframe styling */
    .dataframe { font-size: 0.9rem; }
    div[data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


def api_get(path: str, params: dict = None, timeout: int = 10) -> dict:
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API error: {e}")
        return None


def main():
    st.markdown('<p class="main-header">Disease Case Forecast — Sri Lanka</p>', unsafe_allow_html=True)

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("### Filters")
        diseases_resp = api_get("/diseases")
        if not diseases_resp:
            st.warning("Start the FastAPI backend (see README). Using fallback list.")
            disease_options = [
                {"id": "leptospirosis", "name": "Leptospirosis"},
                {"id": "typhus", "name": "Typhus"},
                {"id": "hepatitis_a", "name": "Hepatitis A"},
                {"id": "chickenpox", "name": "Chickenpox"},
            ]
        else:
            disease_options = diseases_resp.get("diseases", [])

        disease_labels = [d["name"] for d in disease_options]
        disease_ids = [d["id"] for d in disease_options]
        sel_label = st.selectbox("Disease", disease_labels, index=0)
        disease_id = disease_ids[disease_labels.index(sel_label)]

        districts_resp = api_get("/districts", {"disease_id": disease_id})
        if districts_resp and "districts" in districts_resp:
            all_districts = districts_resp["districts"]
        else:
            all_districts = []

        geo = st.radio("Geography", ["All districts (country)", "Select districts"], index=0)
        selected_districts = None
        if geo == "Select districts" and all_districts:
            selected_districts = st.multiselect("Districts", all_districts, default=all_districts[:3])

        # Model: Ensemble is primary. LSTM optional (enable in config).
        if LSTM_ENABLED:
            st.markdown("### Model")
            model_choice = st.radio("Select model", ["XGB + LGB (Ensemble)", "LSTM"], index=0, help="Ensemble (primary): lower–median–upper range. LSTM (secondary): single point prediction.")
            model_param = "ensemble" if "Ensemble" in model_choice else "lstm"
        else:
            model_param = DEFAULT_MODEL

        st.markdown("### Prediction week")
        today = pd.Timestamp("today").date()
        min_pred_date = today - pd.Timedelta(days=365 * 10)  # allow past dates to view actuals
        max_pred_date = today + pd.Timedelta(days=365)  # allow future dates; prediction only for ~2 weeks ahead (weather limit)
        prediction_week_date = st.date_input(
            "Select date (past, today, or future)",
            value=today,
            min_value=min_pred_date,
            max_value=max_pred_date,
            help="Any date. Actuals only for weeks up to 2025-09-05; predictions only for weeks after that and within 2 weeks ahead (weather forecast limit).",
            key="prediction_week_date",
        )
        st.caption("Actuals only up to 2025-09-05. Predictions for weeks after that, up to 2 weeks ahead (weather forecast).")

        st.markdown("---")
        st.markdown("### Date range (past cases)")
        use_date_range = st.checkbox("Limit date range", value=False)
        start_date = end_date = None
        if use_date_range:
            start_date = st.date_input("From", value=pd.Timestamp("2024-01-01"))
            end_date = st.date_input("To", value=pd.Timestamp.today())

        st.markdown("---")
        st.markdown("### Explainability")
        explainability_on = st.checkbox("Enable Explainability tab", value=False, help="When on, show the Explainability tab (live SHAP). Uses one district at a time.")

    # ---------- Tabs ----------
    if explainability_on:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Past cases", "Predictions", "Charts(Overall Country)", "Explainability"])
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["Dashboard", "Past cases", "Predictions", "Charts(Overall Country)"])

    today = pd.Timestamp("today").date()
    d_sel = pd.Timestamp(prediction_week_date)
    week_start_sel = (d_sel - pd.Timedelta(days=d_sel.dayofweek)).date()
    week_end_sel = week_start_sel + pd.Timedelta(days=6)
    # Predictions only when: (1) week is after last actual data, (2) week is within weather forecast (current + 2 weeks)
    within_forecast_window = week_start_sel <= (today + pd.Timedelta(days=PREDICTION_FORECAST_DAYS_AHEAD))
    show_prediction_anywhere = (week_end_sel > LAST_ACTUAL_WEEK_END) and within_forecast_window

    # ---------- Dashboard tab ----------
    with tab1:
        show_prediction_on_dashboard = show_prediction_anywhere

        st.markdown('<p class="section-title">Summary</p>', unsafe_allow_html=True)
        # Selected view: same card style (Disease, Date range, Districts, Case count for this week)
        week_range_str = f"{week_start_sel} to {week_end_sel}"
        districts_count_str = "All (country)" if geo == "All districts (country)" else str(len(selected_districts)) if (geo == "Select districts" and selected_districts) else "0"
        districts_label_view = "Districts" if geo == "All districts (country)" else "Selected districts"
        # Case count for selected week (actuals when available)
        past_week_params = {"disease_id": disease_id, "start_date": str(week_start_sel), "end_date": str(week_end_sel)}
        if selected_districts and geo == "Select districts":
            past_week_params["districts"] = ",".join(selected_districts)
        past_week_summary = api_get("/past_cases", past_week_params)
        case_count_str = "—"
        if past_week_summary and past_week_summary.get("data"):
            total_cases = sum(r.get("cases", 0) for r in past_week_summary["data"])
            case_count_str = f"{total_cases} cases"
        selected_view_cards = f'''
        <div class="summary-cards-wrapper">
            <div class="summary-cards-row">
                <div class="metric-card orange"><span class="metric-label">Date range</span><br><span class="metric-value metric-value-sm">{week_range_str}</span></div>
                <div class="metric-card purple"><span class="metric-label">{districts_label_view}</span><br><span class="metric-value">{districts_count_str}</span></div>
                <div class="metric-card pink"><span class="metric-label">Cases (this week)</span><br><span class="metric-value">{case_count_str}</span></div>
            </div>
        </div>
        '''
        st.markdown(selected_view_cards, unsafe_allow_html=True)
        if geo == "Select districts" and selected_districts:
            st.caption(f"Selected: {', '.join(selected_districts)}")
        pred = None
        if show_prediction_on_dashboard:
            pred_params = {"disease_id": disease_id, "model": model_param, "prediction_date": prediction_week_date.strftime("%Y-%m-%d")}
            if selected_districts and geo == "Select districts":
                pred_params["districts"] = ",".join(selected_districts)
            pred = api_get("/predict", pred_params, timeout=90)
            if pred and "error" in pred:
                st.warning(pred["error"])
            if pred and "error" not in pred:
                if pred.get("weather_fallback"):
                    st.info("Using last available data from model (weather API unavailable). Set **OPENWEATHER_API_KEY** in the backend environment to use live weather.")
                week_start = pred.get("prediction_week_start", "—")
                week_end = pred.get("prediction_week_end", "—")
                date_range_str = f"{week_start} to {week_end}" if (week_start != "—" and week_end != "—") else week_start
                is_ensemble = pred.get("model") == "ensemble"
                if is_ensemble:
                    me = pred.get("country_total_median", 0) or 0
                    lo, hi = pred.get("country_total_lower", 0), pred.get("country_total_upper", 0)
                    lo_cap, _ = _capped_range(lo, me, hi)
                    if me != round(me):
                        total_str = f"{me:.1f} ({math.floor(lo_cap):.0f}–{math.ceil(me):.0f})"
                    else:
                        total_str = f"{me:.0f} ({lo_cap:.0f}–{me:.0f})"
                else:
                    total_str = f"{pred.get('country_total', 0):.0f}"
                total_label = "District(s) total (pred.)" if (geo == "Select districts" and selected_districts) else "Country total (pred.)"
                cards_html = f'''
                <div class="summary-cards-wrapper">
                    <div class="summary-cards-row summary-cards-row-2">
                        <div class="metric-card green"><span class="metric-label">Prediction week:</span><br><span class="metric-value">{date_range_str}</span></div>
                        <div class="metric-card orange"><span class="metric-label">{total_label}:</span><br><span class="metric-value">{total_str} cases</span></div>
                    </div>
                </div>
                '''
                st.markdown(cards_html, unsafe_allow_html=True)
                if is_ensemble:
                    st.caption(ENSEMBLE_PLANNING_CAPTION)
            else:
                st.info("Load backend to see prediction summary.")
        elif week_end_sel > LAST_ACTUAL_WEEK_END and not within_forecast_window:
            st.info("Predictions are only available for the **current week** and **up to 2 weeks ahead** (weather forecast limit). Select a date within that range for predictions.")
        else:
            st.info("Selected week is within the actual data period (up to 2025-09-05). Only actual data is shown; no predictions.")

        st.markdown('<p class="section-title section-title-quickview">Quick view: Past 4 weeks + Next week prediction</p>', unsafe_allow_html=True)
        # Past 4 weeks range: when showing prediction, end day before selected week; when actuals only, include selected week
        d_for_range = pd.Timestamp(prediction_week_date)
        pred_start = (d_for_range - pd.Timedelta(days=d_for_range.dayofweek))
        if show_prediction_on_dashboard:
            past_end = pred_start - pd.Timedelta(days=1)
            past_start = pred_start - pd.Timedelta(days=28)
        else:
            past_end = week_end_sel
            past_start = (pd.Timestamp(week_start_sel) - pd.Timedelta(days=21)).date()
        past_params = {"disease_id": disease_id, "start_date": past_start.strftime("%Y-%m-%d"), "end_date": past_end.strftime("%Y-%m-%d")}
        if selected_districts and geo == "Select districts":
            past_params["districts"] = ",".join(selected_districts)
        past_4w = api_get("/past_cases", past_params)

        if past_4w and past_4w.get("data"):
            df_past = pd.DataFrame(past_4w["data"])
            df_past["week_start"] = pd.to_datetime(df_past["week_start"])
            district_filter_on = geo == "Select districts" and selected_districts
            pred_list = (pred.get("districts") or []) if pred and "error" not in pred else []

            if show_prediction_on_dashboard and pred and "error" not in pred and "prediction_week_start" in pred and pred_list:
                pred_start = pd.Timestamp(pred["prediction_week_start"])
                if district_filter_on and len(selected_districts) >= 1:
                    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
                    fig = go.Figure()
                    for idx, dist in enumerate(selected_districts):
                        sub = df_past[df_past["district"] == dist].sort_values("week_start")
                        pred_row = next((p for p in pred_list if p["district"] == dist), None)
                        xs = sub["week_start"].tolist()
                        ys = sub["cases"].tolist()
                        symbols = ["circle"] * len(xs)
                        if pred_row is not None:
                            xs.append(pred_start)
                            y_pred = pred_row.get("median", pred_row.get("predicted_cases", 0))
                            ys.append(y_pred)
                            symbols.append("diamond")
                        if len(xs) > 0:
                            fig.add_trace(go.Scatter(
                                x=xs, y=ys, mode="lines+markers", name=dist,
                                line=dict(color=colors[idx % len(colors)], width=2),
                                marker=dict(size=10, symbol=symbols, line=dict(width=1, color="white")),
                            ))
                    fig.update_layout(
                        title="Past 4 weeks (actual) + Next week (predicted) — by district (◆ = predicted)",
                        xaxis_title="Week", yaxis_title="Cases",
                        template="plotly_dark",
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, xref="paper", itemwidth=40, tracegroupgap=12, font=dict(size=12)),
                        margin=dict(t=45, r=180),
                    )
                else:
                    by_week = df_past.groupby("week_start")["cases"].sum().reset_index().sort_values("week_start")
                    total_pred = pred.get("country_total_median", pred.get("country_total", 0))
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=by_week["week_start"], y=by_week["cases"],
                        mode="lines+markers", name="Past 4 weeks (actual)",
                        line=dict(color="#0ea5e9", width=2), marker=dict(size=10),
                    ))
                    fig.add_trace(go.Scatter(
                        x=[pred_start], y=[total_pred],
                        mode="markers", name="Predicted",
                        marker=dict(size=14, color="#f97316", symbol="diamond", line=dict(width=2, color="white")),
                    ))
                    fig.update_layout(
                        title="Past 4 weeks (actual) + Next week (predicted) — country total",
                        xaxis_title="Week", yaxis_title="Cases",
                        template="plotly_dark",
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, xref="paper", itemwidth=40, font=dict(size=12)),
                        margin=dict(t=45, r=140),
                    )
                st.plotly_chart(fig, use_container_width=True, key="plot_dashboard_quickview")
            else:
                # No prediction: show only past 4 weeks (actual), no diamond
                fig = go.Figure()
                if district_filter_on and len(selected_districts) >= 1:
                    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
                    for idx, dist in enumerate(selected_districts):
                        sub = df_past[df_past["district"] == dist].sort_values("week_start")
                        if len(sub) > 0:
                            fig.add_trace(go.Scatter(
                                x=sub["week_start"], y=sub["cases"],
                                mode="lines+markers", name=dist,
                                line=dict(color=colors[idx % len(colors)], width=2),
                                marker=dict(size=10, line=dict(width=1, color="white")),
                            ))
                    fig.update_layout(
                        title="Past 4 weeks (actual) — by district",
                        xaxis_title="Week", yaxis_title="Cases",
                        template="plotly_dark",
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, xref="paper", itemwidth=40, tracegroupgap=12, font=dict(size=12)),
                        margin=dict(t=45, r=180),
                    )
                else:
                    by_week = df_past.groupby("week_start")["cases"].sum().reset_index().sort_values("week_start")
                    fig.add_trace(go.Scatter(
                        x=by_week["week_start"], y=by_week["cases"],
                        mode="lines+markers", name="Past 4 weeks (actual)",
                        line=dict(color="#0ea5e9", width=2), marker=dict(size=10),
                    ))
                    fig.update_layout(
                        title="Past 4 weeks (actual)",
                        xaxis_title="Week", yaxis_title="Cases",
                        template="plotly_dark",
                        legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02, xref="paper", itemwidth=40, font=dict(size=12)),
                        margin=dict(t=45, r=140),
                    )
                st.plotly_chart(fig, use_container_width=True, key="plot_dashboard_quickview")
                if not show_prediction_on_dashboard:
                    st.caption("Predictions (and the diamond) are shown only for weeks after 2025-09-05.")
        elif show_prediction_on_dashboard and pred and "districts" in pred and "error" not in pred:
            pred_list = pred.get("districts") or []
            if pred_list:
                district_filter_on = geo == "Select districts" and selected_districts
                y_val = lambda p: p.get("median", p.get("predicted_cases", 0))
                if district_filter_on and selected_districts:
                    fig = go.Figure()
                    for i, p in enumerate(pred_list):
                        fig.add_trace(go.Bar(x=[p["district"]], y=[y_val(p)], name=p["district"], marker_color=px.colors.qualitative.Set2[i % len(px.colors.qualitative.Set2)]))
                    fig.update_layout(title="Predicted — no past 4 weeks data", xaxis_title="District", yaxis_title="Cases", template="plotly_dark", barmode="group", margin=dict(t=40))
                else:
                    tot = pred.get("country_total_median", pred.get("country_total", 0))
                    fig = go.Figure(go.Indicator(mode="number", value=tot, title="Predicted — country total"))
                    fig.update_layout(template="plotly_dark", margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True, key="plot_dashboard_pred_only")
                st.caption("Past 4 weeks case data not available for this range; showing prediction only.")
            else:
                st.caption("No prediction or past data available for the chart.")
        else:
            st.caption("Run the backend and load past data to see the chart. Predictions shown only for weeks after 2025-09-05.")

    # ---------- Past cases tab ----------
    with tab2:
        st.markdown('<p class="section-title">Past case counts</p>', unsafe_allow_html=True)
        params = {"disease_id": disease_id}
        if selected_districts and geo == "Select districts":
            params["districts"] = ",".join(selected_districts)
        if use_date_range and start_date and end_date:
            params["start_date"] = str(start_date)
            params["end_date"] = str(end_date)
        past = api_get("/past_cases", params)
        if past and "data" in past and past["data"]:
            df = pd.DataFrame(past["data"])
            st.dataframe(df, use_container_width=True, hide_index=True)
            # Aggregate by week for time series (if multiple districts, sum for country view)
            df["week_start"] = pd.to_datetime(df["week_start"])
            if geo == "All districts (country)" or (selected_districts and len(selected_districts) > 1):
                by_week = df.groupby("week_start")["cases"].sum().reset_index()
                by_week = by_week.sort_values("week_start")
                fig = px.area(by_week, x="week_start", y="cases", color_discrete_sequence=["#0ea5e9"])
                fig.update_layout(title="Country total cases by week", xaxis_title="Week", yaxis_title="Cases", template="plotly_dark", margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True, key="plot_past_country")
            else:
                fig = px.line(df.sort_values("week_start"), x="week_start", y="cases", color="district", color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(title="Cases by district", xaxis_title="Week", yaxis_title="Cases", template="plotly_dark", margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True, key="plot_past_district")
        else:
            st.info("No past case data for the selected filters.")

    # ---------- Predictions tab ----------
    with tab3:
        today = pd.Timestamp("today").date()
        d = pd.Timestamp(prediction_week_date)
        week_start = (d - pd.Timedelta(days=d.dayofweek)).date()
        week_end = week_start + pd.Timedelta(days=6)
        week_in_past = week_end < today

        past_has_actual = False
        if week_in_past:
            st.markdown('<p class="section-title">Actual case counts (selected week is in the past)</p>', unsafe_allow_html=True)
            past_params = {"disease_id": disease_id, "start_date": str(week_start), "end_date": str(week_end)}
            if selected_districts and geo == "Select districts":
                past_params["districts"] = ",".join(selected_districts)
            past_week = api_get("/past_cases", past_params)
            if past_week and past_week.get("data"):
                past_has_actual = True
                df_actual = pd.DataFrame(past_week["data"])
                st.caption(f"Week: {week_start} to {week_end}")
                # Table: district and Cases only (integers, no decimals, no ranges)
                st.dataframe(df_actual[["district", "cases"]].rename(columns={"cases": "Cases"}), use_container_width=True, hide_index=True)
                total_actual = int(df_actual["cases"].sum())
                st.metric("District(s) total (actual)", f"{total_actual} cases")
                # Bar chart: actual cases only (whole numbers, no uncertainty)
                fig_actual = px.bar(df_actual, x="district", y="cases", labels={"cases": "Cases"}, color_discrete_sequence=["#22c55e"])
                fig_actual.update_layout(title="Actual cases by district", template="plotly_dark", xaxis_tickangle=-45, margin=dict(t=40), showlegend=False, yaxis_tickmode="linear")
                st.plotly_chart(fig_actual, use_container_width=True, key="plot_predictions_bar")
            else:
                st.info("No actual case data for this week in the database. Showing model prediction below.")
                st.markdown('<p class="section-title">Model prediction for that week</p>', unsafe_allow_html=True)

        if not week_in_past or not past_has_actual:
            if not week_in_past and not show_prediction_anywhere:
                st.markdown('<p class="section-title">Predicted case counts</p>', unsafe_allow_html=True)
                st.info("Predictions are only available for the **current week** and **up to 2 weeks ahead** (weather forecast limit). Select a date within that range.")
            else:
                # Past week with no actuals, or future week within forecast window: show prediction
                if not week_in_past:
                    st.markdown('<p class="section-title">Predicted case counts</p>', unsafe_allow_html=True)
                params = {"disease_id": disease_id, "model": model_param, "prediction_date": prediction_week_date.strftime("%Y-%m-%d")}
                if selected_districts and geo == "Select districts":
                    params["districts"] = ",".join(selected_districts)
                pred = api_get("/predict", params, timeout=90)
                if pred and "error" in pred:
                    st.error(pred["error"])
                elif pred and "districts" in pred:
                    if pred.get("weather_fallback"):
                        st.info("Using last available data from model (weather API unavailable). Set **OPENWEATHER_API_KEY** in the backend to use live weather.")
                    st.caption(f"Prediction week: {pred.get('prediction_week_start', '')} to {pred.get('prediction_week_end', '')}")
                    df_pred = pd.DataFrame(pred["districts"])
                    is_ensemble = pred.get("model") == "ensemble"
                    if is_ensemble:
                        st.caption(ENSEMBLE_PLANNING_CAPTION)
                        df_pred = df_pred.copy()
                        capped = df_pred.apply(lambda r: _capped_range(r["lower"], r["median"], r["upper"]), axis=1)
                        df_pred["lower_cap"], df_pred["upper_cap"] = zip(*capped)
                        def _fmt(v):
                            x = float(v)
                            return f"{x:.2f} ({round(x)})"
                        def _fmt_avg(v):
                            x = float(v)
                            lo, hi = math.floor(x), math.ceil(x)
                            return f"{x:.2f} ({lo} - {hi}) cases"
                        df_display = df_pred[["district"]].copy()
                        df_display["Lower"] = df_pred["lower_cap"].apply(_fmt)
                        df_display["Model predicted"] = df_pred["median"].apply(_fmt)
                        df_display["Upper"] = df_pred["upper_cap"].apply(_fmt)
                        df_display["Average cases"] = df_pred["median"].apply(_fmt_avg)
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                        lo, me, hi = pred.get("country_total_lower", 0), pred.get("country_total_median", 0), pred.get("country_total_upper", 0)
                        lo_cap, _ = _capped_range(lo, me, hi)
                        if me != round(me):
                            total_display = f"{me:.1f} ({math.floor(lo_cap):.0f}–{math.ceil(me):.0f}) cases"
                        else:
                            total_display = f"{me:.0f} ({lo_cap:.0f}–{me:.0f}) cases"
                        st.metric("District(s) total (predicted) — use median for planning", total_display)
                        customdata = list(zip(
                            df_pred["lower_cap"].round(2),
                            df_pred["median"].apply(round), df_pred["lower_cap"].apply(round),
                        ))
                        fig = go.Figure(go.Bar(
                            x=df_pred["district"], y=df_pred["median"], name="Predicted (median)",
                            error_y=dict(type="data", array=[0] * len(df_pred), arrayminus=(df_pred["median"] - df_pred["lower_cap"]).tolist(), color="#f97316"),
                            marker_color="#8b5cf6",
                            customdata=customdata,
                            hovertemplate="%{x}<br>Bar: 0 to %{y:.2f} (%{customdata[1]})<br>Range: %{customdata[0]:.2f} – %{y:.2f} (%{customdata[2]}–%{customdata[1]})<extra></extra>",
                        ))
                        fig.update_layout(title="Predicted cases by district (median with range)", template="plotly_dark", xaxis_tickangle=-45, margin=dict(t=40), showlegend=True, legend=dict(orientation="h", x=1, xanchor="right", y=1.02))
                    else:
                        df_pred = df_pred.rename(columns={"predicted_cases": "Predicted cases"})
                        df_display = df_pred[["district", "Predicted cases"]].copy()
                        df_display["Predicted cases"] = df_display["Predicted cases"].apply(
                            lambda x: f"{float(x):.2f} ({round(float(x))})"
                        )
                        st.dataframe(df_display, use_container_width=True, hide_index=True)
                        st.metric("District(s) total (predicted)", f"{pred.get('country_total', 0):.0f} cases")
                        fig = px.bar(df_pred, x="district", y="Predicted cases", color="Predicted cases", color_continuous_scale="Viridis")
                        fig.update_layout(template="plotly_dark", xaxis_tickangle=-45, margin=dict(t=40), showlegend=False)
                    st.plotly_chart(fig, use_container_width=True, key="plot_predictions_bar")
                else:
                    st.info("Start the FastAPI backend to see predictions.")

    # ---------- Charts tab ----------
    with tab4:
        st.markdown('<p class="section-title">Time series & comparison</p>', unsafe_allow_html=True)
        show_prediction_in_charts = show_prediction_anywhere

        past = api_get("/past_cases", {"disease_id": disease_id})
        if past and past.get("data"):
            df_past = pd.DataFrame(past["data"])
            df_past["week_start"] = pd.to_datetime(df_past["week_start"])
            by_week = df_past.groupby("week_start")["cases"].sum().reset_index().sort_values("week_start")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=by_week["week_start"], y=by_week["cases"], mode="lines+markers", name="Past (actual)", line=dict(color="#0ea5e9", width=2)))
            if show_prediction_in_charts:
                pred = api_get("/predict", {"disease_id": disease_id, "model": model_param, "prediction_date": prediction_week_date.strftime("%Y-%m-%d")}, timeout=90)
                if pred and "districts" in pred and "error" not in pred:
                    pred_week = pred.get("prediction_week_start")
                    pred_total = pred.get("country_total_median", pred.get("country_total", 0))
                    if pred_week:
                        fig.add_trace(go.Scatter(x=[pd.Timestamp(pred_week)], y=[pred_total], mode="markers", name="Predicted", marker=dict(size=14, color="#f97316", symbol="diamond")))
                    fig.update_layout(title="Country total: past cases and prediction", xaxis_title="Week", yaxis_title="Cases", template="plotly_dark", legend=dict(orientation="h"), margin=dict(t=40))
                    st.plotly_chart(fig, use_container_width=True, key="plot_charts_country_ts")
                    # District bar only when we show prediction (after last actual week)
                    df_pred = pd.DataFrame(pred["districts"])
                    if pred.get("model") == "ensemble":
                        capped = df_pred.apply(lambda r: _capped_range(r["lower"], r["median"], r["upper"]), axis=1)
                        df_pred["lower_cap"], df_pred["upper_cap"] = zip(*capped)
                        customdata = list(zip(
                            df_pred["lower_cap"].round(2),
                            df_pred["median"].apply(round), df_pred["lower_cap"].apply(round),
                        ))
                        fig2 = go.Figure(go.Bar(
                            x=df_pred["district"], y=df_pred["median"], name="Predicted (median)",
                            error_y=dict(type="data", array=[0] * len(df_pred), arrayminus=(df_pred["median"] - df_pred["lower_cap"]).tolist(), color="#f97316"),
                            marker_color="#8b5cf6",
                            customdata=customdata,
                            hovertemplate="%{x}<br>Bar: 0 to %{y:.2f} (%{customdata[1]})<br>Range: %{customdata[0]:.2f} – %{y:.2f} (%{customdata[2]}–%{customdata[1]})<extra></extra>",
                        ))
                        fig2.update_layout(title="Predicted cases by district (median with range)", template="plotly_dark", xaxis_tickangle=-45, margin=dict(t=40), showlegend=True, legend=dict(orientation="h", x=1, xanchor="right", y=1.02))
                    else:
                        fig2 = px.bar(df_pred, x="district", y="predicted_cases", color="predicted_cases", color_continuous_scale="Plasma", title="Predicted cases by district")
                        fig2.update_layout(template="plotly_dark", xaxis_tickangle=-45, margin=dict(t=40), showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True, key="plot_charts_district_bar")
                else:
                    fig.update_layout(title="Country total: past cases", xaxis_title="Week", yaxis_title="Cases", template="plotly_dark", legend=dict(orientation="h"), margin=dict(t=40))
                    st.plotly_chart(fig, use_container_width=True, key="plot_charts_country_ts")
                    if show_prediction_in_charts and pred and "error" in pred:
                        st.warning(pred.get("error", "Prediction unavailable for this week."))
            else:
                fig.update_layout(title="Country total: past cases", xaxis_title="Week", yaxis_title="Cases", template="plotly_dark", legend=dict(orientation="h"), margin=dict(t=40))
                st.plotly_chart(fig, use_container_width=True, key="plot_charts_country_ts")
                if week_end_sel > LAST_ACTUAL_WEEK_END and not within_forecast_window:
                    st.caption("Predictions only for the **current week** and **up to 2 weeks ahead** (weather forecast limit). Select a date within that range.")
                else:
                    st.caption("Prediction is shown only for weeks after 2025-09-05 and within 2 weeks ahead. Select such a week to see the predicted point and district bar.")
        else:
            st.info("Load past data and run backend to see combined charts.")

    # ---------- Explainability tab (live SHAP for current prediction) ----------
    if explainability_on:
        with tab5:
            st.info("**Explainability uses one district at a time.** When 'Select districts' is chosen, the first selected district is used. When 'All districts' is chosen, the first available district is used.")
            st.markdown('<p class="section-title">Explain this prediction (live SHAP)</p>', unsafe_allow_html=True)
            explain_district = None
            if geo == "Select districts" and selected_districts:
                explain_district = selected_districts[0]
            explain_params = {"disease_id": disease_id, "prediction_date": prediction_week_date.strftime("%Y-%m-%d")}
            if explain_district:
                explain_params["district"] = explain_district
            expl = api_get("/explain", explain_params, timeout=30)
            if expl and expl.get("shap_values"):
                st.caption(f"**Why** did the model predict this for **{expl.get('district', '—')}** on week **{prediction_week_date}**? Positive = pushes cases up, negative = down.")
                df_shap = pd.DataFrame(expl["shap_values"])
                df_shap = df_shap.sort_values("value", key=abs, ascending=False).reset_index(drop=True)
                top_n = 20
                df_top = df_shap.head(top_n)
                fig_ex = px.bar(
                    df_top.sort_values("value", ascending=True),
                    y="feature",
                    x="value",
                    orientation="h",
                    labels={"value": "SHAP contribution", "feature": "Feature"},
                    color="value",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                )
                fig_ex.update_layout(template="plotly_dark", margin=dict(t=30, l=180), height=400 + min(len(df_top), 20) * 18, showlegend=False, xaxis_title="SHAP contribution (positive = higher cases)")
                st.plotly_chart(fig_ex, use_container_width=True, key="plot_explainability_live")
                with st.expander("All feature contributions"):
                    st.dataframe(df_shap.rename(columns={"value": "SHAP value"}), use_container_width=True, hide_index=True)
            elif expl and expl.get("error"):
                st.warning(expl["error"])
                st.markdown("**Fallback: global feature importance** (precomputed)")
                fi = api_get("/feature_importance", {"disease_id": disease_id})
                if fi and fi.get("features"):
                    df_fi = pd.DataFrame(fi["features"])
                    fig_fi = px.bar(df_fi.head(20).sort_values("mean_abs_shap_value", ascending=True), y="feature", x="mean_abs_shap_value", orientation="h", labels={"mean_abs_shap_value": "Importance"}, color="mean_abs_shap_value", color_continuous_scale="Blues")
                    fig_fi.update_layout(template="plotly_dark", margin=dict(t=30, l=180), height=400, showlegend=False)
                    st.plotly_chart(fig_fi, use_container_width=True, key="plot_explainability_fallback")
                else:
                    st.caption("Run step_04_shap_explainability.py to generate shap_explainer.pkl and feature importance CSVs.")
            else:
                st.info("Start the FastAPI backend to see live SHAP explanations.")


if __name__ == "__main__":
    main()

