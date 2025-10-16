# app.py
# Streamlit stock movement tracker — fixed Y-axis, adaptive X-axis ticks, and safe concatenation
# Requires: yfinance, pandas, numpy, streamlit, plotly, matplotlib

import time
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator, AutoMinorLocator, FuncFormatter
from datetime import datetime, timedelta

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="Stock Movement Tracker", layout="wide")

# ----------------------------
# Default tickers & sector mapping
# ----------------------------
STOCKS = [
    "META","TSLA","AMZN","MSFT","NVDA","ORCL","AVGO","GFI","GOOG","GOOGL",
    "AAPL","AMD","PANW","VRT","QCOM","INOD","SMCI","RGTI","QBTS","QUBT",
    "LAES","BBAI","LUMN","IREN","ANET","HIMS","OKTA","MRVL","INTC","SOUN",
    "AGI","AEM","HMY","IAG","NEM","LAC","POET","UNH","JOBY","IONQ","JNJ","LLY"
]

SECTOR_MAPPING = {
    "META": "Tech","TSLA": "Automotive","AMZN": "E-Commerce","MSFT": "Tech","NVDA": "Tech",
    "ORCL": "Tech","AVGO": "Tech","GFI": "Finance","GOOG": "Tech","GOOGL": "Tech",
    "AAPL": "Tech","AMD": "Tech","PANW": "Tech","VRT": "Industrial","QCOM": "Tech",
    "INOD": "Tech","SMCI": "Tech","RGTI": "Tech","QBTS": "Tech","QUBT": "Tech",
    "LAES": "Energy","BBAI": "Tech","LUMN": "Telecom","IREN": "Energy","ANET": "Tech",
    "HIMS": "Healthcare","OKTA": "Tech","MRVL": "Tech","INTC": "Tech","SOUN": "Tech",
    "AGI": "Mining","AEM": "Mining","HMY": "Mining","IAG": "Mining","NEM": "Mining",
    "LAC": "Mining","POET": "Tech","UNH": "Healthcare","JOBY": "Aerospace","IONQ": "Tech"
}

# ----------------------------
# Expanded TIME_WINDOWS (hours)
# ----------------------------
TIME_WINDOWS = {
    "5 Minutes": 5/60,
    "10 Minutes": 10/60,
    "15 Minutes": 15/60,
    "30 Minutes": 30/60,
    "1 Hour": 1,
    "2 Hours": 2,
    "4 Hours": 4,
    "6 Hours": 6,
    "12 Hours": 12,
    "24 Hours": 24,
    "36 Hours": 36,
    "48 Hours": 48,
    "72 Hours": 72,
    "96 Hours": 96,
    "1 Week": 24 * 7,
    "2 Weeks": 24 * 14,
    "1 Month": 24 * 30,
    "2 Months": 24 * 60,
    "6 Months": 24 * 30 * 6,
    "1 Year": 24 * 365,
    "2 Years": 24 * 365 * 2
}

PRESET_LABELS = list(TIME_WINDOWS.keys())
REQUEST_DELAY_SECONDS = 0.12

# ----------------------------
# Map hours -> yfinance period/interval
# ----------------------------
def map_hours_to_period_interval(hours: float):
    if hours <= 10/60:
        return ("1d", "1m")
    if hours <= 0.25:
        return ("2d", "1m")
    if hours <= 1:
        return ("2d", "2m")
    if hours <= 2:
        return ("5d", "5m")
    if hours <= 4:
        return ("5d", "15m")
    if hours <= 6:
        return ("7d", "30m")
    if hours <= 12:
        return ("7d", "60m")
    if hours <= 24:
        return ("14d", "1h")
    if hours <= 36:
        return ("30d", "1h")
    if hours <= 48:
        return ("30d", "1h")
    if hours <= 72:
        return ("60d", "1d")
    if hours <= 96:
        return ("90d", "1d")
    if hours <= 24 * 21:
        return ("3mo", "1d")
    if hours <= 24 * 62:
        return ("6mo", "1d")
    if hours <= 24 * 200:
        return ("1y", "1d")
    return ("2y", "1d")

# ----------------------------
# Fetch close price series
# ----------------------------
def fetch_time_series_for_range(ticker: str, period: str, interval: str, cutoff=None):
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(period=period, interval=interval, auto_adjust=True, prepost=False)
        if df is None or df.empty:
            return None
        if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
            try:
                df.index = df.index.tz_localize("UTC")
            except Exception:
                pass
        if cutoff is not None:
            try:
                df = df.loc[df.index >= cutoff]
            except Exception:
                pass
        if "Close" not in df.columns or df["Close"].shape[0] < 2:
            return None
        return df["Close"]
    except Exception:
        return None

# ----------------------------
# Percent change series
# ----------------------------
def percent_change_series(series: pd.Series):
    if series is None or series.empty:
        return None
    s = series.astype(float)
    base = s.iloc[0]
    if base == 0 or pd.isna(base):
        return None
    return (s / base - 1.0) * 100.0  # percent

# ----------------------------
# Cached helper for multiple tickers
# ----------------------------
@st.cache_data(ttl=30)
def fetch_percent_changes_for_tickers(tickers, period, interval):
    results = []
    warnings = []
    series_dict = {}
    for t in tickers:
        ser = fetch_time_series_for_range(t, period, interval)
        time.sleep(REQUEST_DELAY_SECONDS)
        if ser is None:
            warnings.append(f"No data for {t}")
            continue
        p = percent_change_series(ser)
        if p is None:
            warnings.append(f"Could not compute percent for {t}")
            continue
        series_dict[t] = p
        try:
            final_pct = float(p.iloc[-1])
        except Exception:
            warnings.append(f"Invalid final pct for {t}")
            continue
        results.append((t, round(final_pct, 4)))
    if not results:
        return pd.DataFrame(columns=["Stock", "Change %"]), warnings, series_dict
    df = pd.DataFrame(results, columns=["Stock", "Change %"])
    df = df.sort_values("Change %", ascending=False).reset_index(drop=True)
    return df, warnings, series_dict

# ----------------------------
# Helper: get X-axis tick spacing for Plotly
# ----------------------------
def get_dtick_and_format(hours_span):
    if hours_span <= 0.5:
        return 5*60*1000, "%H:%M"       # 5 min
    if hours_span <= 1:
        return 10*60*1000, "%H:%M"      # 10 min
    if hours_span <= 6:
        return 30*60*1000, "%H:%M"      # 30 min
    if hours_span <= 12:
        return 60*60*1000, "%H:%M"      # 1 hr
    if hours_span <= 24:
        return 2*60*60*1000, "%H:%M"    # 2 hr
    if hours_span <= 24*7:
        return "D1", "%b %d"            # 1 day
    else:
        return "D7", "%b %d"            # 1 week

# ----------------------------
# UI Tabs
# ----------------------------
tab1, tab2 = st.tabs(["Stock Movements", "Sector Percent Trends"])

# ---------- Tab 1: Stock Movements ----------
with tab1:
    st.title("📊 Stock Movement Tracker")
    st.write("Pick a timeframe (presets match the detailed hour-based windows). The static bar chart (green/red) appears above the interactive chart.")

    col_left, col_right = st.columns([3,1])
    with col_left:
        preset_label = st.selectbox("Select Timeframe", PRESET_LABELS, index=3)
        hours = TIME_WINDOWS[preset_label]
        period, interval = map_hours_to_period_interval(hours)
        chosen_label = preset_label

        tickers_text = st.text_area("Tickers (comma separated) — leave blank to use defaults:", value=", ".join(STOCKS), height=120)
        tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
        if not tickers:
            tickers = STOCKS.copy()

    with col_right:
        view_mode = st.radio("View mode", ["Percent change (normalized)", "Raw price series"])
        fetch_btn = st.button("Fetch & Show")

    if fetch_btn:
        st.info(f"Fetching data for {chosen_label} (period={period}, interval={interval}) — this may take a few seconds.")
        df, warnings, series_dict = fetch_percent_changes_for_tickers(tickers, period, interval)

        if df.empty and not series_dict:
            st.warning("No series data retrieved.")
        else:
            # ---------- Matplotlib Bar Chart ----------
            if not df.empty:
                labels = df["Stock"].tolist()
                values = df["Change %"].tolist()
                fig_w = max(10, len(labels)*0.45)
                fig_h = 5.5
                fig, ax = plt.subplots(figsize=(fig_w, fig_h))
                colors = ["green" if v>=0 else "red" for v in values]
                ax.bar(labels, values, color=colors, edgecolor="black", linewidth=0.3)
                ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
                ylim_padding = max(abs(max(values)), abs(min(values)))*0.15
                ax.set_ylim(min(values)-ylim_padding, max(values)+ylim_padding)
                ax.yaxis.set_major_locator(AutoLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_major_formatter(FuncFormatter(lambda y,_: f"{y:.2f}%"))
                ax.grid(axis="y", which="major", linestyle="-", linewidth=0.8, alpha=0.7)
                ax.grid(axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.5)
                ax.set_ylabel("Change %")
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.subheader("📊 Full Stock Movement (Bar Chart)")
                st.pyplot(fig)
                plt.close(fig)

            # ---------- Plotly Interactive Chart ----------
            st.subheader("📈 Interactive Chart")
            fig = go.Figure()
            all_vals = []
            valid_series = {}
            for t,s in series_dict.items():
                if s is None or len(s)==0:
                    continue
                s = s.dropna()
                if len(s)==0:
                    continue
                s.index = s.index.tz_convert("America/New_York")
                all_vals.append(s.values)
                valid_series[t] = s
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=t, hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>"))

            if all_vals:
                combined_vals = np.concatenate(all_vals)
                pad = max(5.0, (combined_vals.max()-combined_vals.min())*0.12)
                y0, y1 = combined_vals.min()-pad, combined_vals.max()+pad
                fig.update_yaxes(range=[y0, y1], tickformat=".2f", ticks="outside")
            else:
                y0, y1 = -5, 5
                fig.update_yaxes(range=[y0, y1])

            if valid_series:
                combined_index = next(iter(valid_series.values())).index
                total_seconds = (combined_index[-1]-combined_index[0]).total_seconds()
                total_hours = total_seconds/3600
                dtick, tickformat = get_dtick_and_format(total_hours)
                fig.update_xaxes(title="Time", dtick=dtick, tickformat=tickformat)
                fig.add_shape(type="line", x0=combined_index[0], x1=combined_index[-1], y0=0, y1=0,
                              line=dict(color="gray", width=1, dash="dash"))
            fig.update_layout(title=f"Percent Change — {chosen_label}", hovermode="x unified", height=600,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
            st.plotly_chart(fig, use_container_width=True)

            if warnings:
                st.subheader("Warnings / Errors")
                for w in warnings:
                    st.write("•", w)

# ---------- Tab 2: Sector Percent Trends ----------
with tab2:
    st.title("📈 Sector Percent Trends")
    sectors = sorted(list(set(SECTOR_MAPPING.values())))
    selected_sector = st.selectbox("Choose sector", sectors)
    sector_tickers = [t for t,s in SECTOR_MAPPING.items() if s==selected_sector]
    sector_range_label = st.selectbox("Range", PRESET_LABELS, index=3)
    show_btn = st.button("Show Sector Percent Chart")

    if show_btn and sector_tickers:
        hours = TIME_WINDOWS[sector_range_label]
        period, interval = map_hours_to_period_interval(hours)
        st.info(f"Fetching {len(sector_tickers)} tickers for {sector_range_label}...")
        pct_series = {}
        warnings = []
        for t in sector_tickers:
            s = fetch_time_series_for_range(t, period, interval)
            time.sleep(REQUEST_DELAY_SECONDS)
            p = percent_change_series(s)
            if p is not None:
                pct_series[t] = p
            else:
                warnings.append(f"{t} skipped")

        fig = go.Figure()
        all_vals = []
        valid_series = {}
        for t,s in pct_series.items():
            if s is None or len(s)==0:
                continue
            s = s.dropna()
            if len(s)==0:
                continue
            s.index = s.index.tz_convert("America/New_York")
            all_vals.append(s.values)
            valid_series[t] = s
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=t,
                                     hovertemplate="%{y:.2f}%<extra>%{fullData.name}</extra>"))

        if all_vals:
            combined_vals = np.concatenate(all_vals)
            pad = max(5.0, (combined_vals.max()-combined_vals.min())*0.12)
            y0, y1 = combined_vals.min()-pad, combined_vals.max()+pad
            fig.update_yaxes(range=[y0, y1])
        else:
            fig.update_yaxes(range=[-5,5])

        if valid_series:
            combined_index = next(iter(valid_series.values())).index
            total_seconds = (combined_index[-1]-combined_index[0]).total_seconds()
            total_hours = total_seconds/3600
            dtick, tickformat = get_dtick_and_format(total_hours)
            fig.update_xaxes(title="Time", dtick=dtick, tickformat=tickformat)
            fig.add_shape(type="line", x0=combined_index[0], x1=combined_index[-1], y0=0, y1=0,
                          line=dict(color="gray", width=1, dash="dash"))

        fig.update_layout(title=f"{selected_sector} — Percent Change ({sector_range_label})",
                          yaxis_title="Change %", hovermode="x unified", height=600,
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
        st.plotly_chart(fig, use_container_width=True)

        if warnings:
            st.subheader("Warnings")
            for w in warnings:
                st.write("•", w)

# EOF
#there is stilll problems with the small time frames