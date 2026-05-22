# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess
import sys
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from qusa.notifications import parse_recipients, send_prediction_email
from qusa.utils.config import load_config

# --- Page Config ---
st.set_page_config(
    page_title="Nocturne Command Center",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Professional Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background-color: #f8fafc;
    }
    
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        border: 1px solid #e2e8f0;
        min-width: 180px;
    }

    /* Metric Label/Value Colors */
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        margin-top: 1.5rem;
        border-left: 4px solid #2563eb;
        padding-left: 1rem;
    }
    
    .status-box {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def get_config():
    config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
    return load_config(str(config_path))

def run_script(script_path, args):
    """Run a Nocturne script via subprocess."""
    cmd = [sys.executable, str(PROJECT_ROOT / script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
    return result

def load_prediction_log(log_path):
    """Load the prediction log if it exists."""
    if not log_path.exists():
        return pd.DataFrame()
    return pd.read_csv(log_path).sort_values("timestamp", ascending=False)

def get_latest_prediction(df_log, ticker):
    """Return the latest prediction row for a ticker as a dict."""
    if df_log.empty or "ticker" not in df_log.columns:
        return None

    df_ticker = df_log[df_log["ticker"] == ticker]
    if df_ticker.empty:
        return None

    return df_ticker.iloc[0].to_dict()

def count_ticker_predictions(df_log, ticker):
    """Count prediction log rows for a ticker."""
    if df_log.empty or "ticker" not in df_log.columns:
        return 0
    return int((df_log["ticker"] == ticker).sum())

def email_config_status(email_config):
    """Summarize whether dashboard email settings are ready."""
    if not email_config.get("enabled", False):
        return False, "Email notifications are disabled in config."

    smtp_user_env = email_config.get("smtp_user_env", "QUSA_SMTP_USER")
    smtp_password_env = email_config.get("smtp_password_env", "QUSA_SMTP_PASSWORD")
    missing = []
    if not email_config.get("smtp_host"):
        missing.append("smtp_host")
    if not email_config.get("smtp_port"):
        missing.append("smtp_port")
    if not email_config.get("smtp_user") and not os.getenv(smtp_user_env):
        missing.append("SMTP username")
    if not email_config.get("smtp_password") and not os.getenv(smtp_password_env):
        missing.append("SMTP password")

    if missing:
        return False, f"Missing email setup: {', '.join(missing)}"

    return True, "Email notifications are configured."

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=48)
    st.title("Nocturne Intelligence")
    
    config = get_config()
    
    # Ticker Selection
    available_tickers = []
    raw_dir = Path(config["data"]["paths"]["raw_data_dir"]).expanduser()
    if raw_dir.exists():
        available_tickers = [f.stem.split('_')[0] for f in raw_dir.glob("*_history.csv")]
    
    selected_ticker = st.selectbox("Active Asset", options=sorted(list(set(available_tickers))) if available_tickers else ["UPRO"])
    
    st.divider()
    if st.button("Refresh Pipeline", use_container_width=True):
        with st.spinner("Processing..."):
            run_script("scripts/run_FE_pipeline.py", ["-ticker", selected_ticker, "--fetch"])

# --- Main Content ---
st.title("Nocturne Command Center")

tab_predict, tab_perf, tab_regime = st.tabs(["Signals", "Performance", "Regimes"])

# --- Tab 1: Predictions ---
with tab_predict:
    st.markdown('<p class="section-header">Latest Strategic Intelligence</p>', unsafe_allow_html=True)
    
    log_path = Path(config["prediction"].get("csv_log", config["prediction"].get("log_file"))).expanduser()
    df_log = load_prediction_log(log_path)
    
    if not df_log.empty:
        df_ticker = df_log[df_log['ticker'] == selected_ticker]
        
        if not df_ticker.empty:
            latest = df_ticker.iloc[0]
            
            # Metric Row
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Market Bias", latest['direction'])
            m_col2.metric("Upward Prob.", f"{latest['probability_up']:.1%}")
            m_col3.metric("Conviction", latest['confidence'])
            
            # Status Box
            if latest['confidence'] == 'HIGH':
                st.success(f"STRATEGIC {latest['direction']} SIGNAL DETECTED")
            else:
                st.warning("NO HIGH-CONVICTION SIGNAL PRESENT")
        else:
            st.info("No intelligence available for selected ticker.")
    
    st.markdown('<p class="section-header">Operational Execution Log</p>', unsafe_allow_html=True)
    if not df_log.empty:
        st.dataframe(df_log.head(15), use_container_width=True, hide_index=True)
    
    st.markdown('<p class="section-header">Email Notification</p>', unsafe_allow_html=True)
    email_config = config.get("email", {}).copy()
    smtp_col1, smtp_col2 = st.columns(2)
    with smtp_col1:
        smtp_user_input = st.text_input(
            "SMTP username",
            key="prediction_smtp_user",
            placeholder="sender@gmail.com",
        )
    with smtp_col2:
        smtp_password_input = st.text_input(
            "SMTP password",
            key="prediction_smtp_password",
            type="password",
            placeholder="App password",
        )

    if smtp_user_input:
        email_config["smtp_user"] = smtp_user_input
    if smtp_password_input:
        email_config["smtp_password"] = smtp_password_input

    email_ready, email_status = email_config_status(email_config)
    recipient_input = st.text_input(
        "Prediction recipients",
        key="prediction_email_recipients",
        placeholder="trader@example.com, analyst@example.com",
    )

    recipients = []
    recipients_valid = True
    try:
        recipients = parse_recipients(recipient_input)
    except ValueError as exc:
        recipients_valid = False
        st.warning(str(exc))

    if email_ready:
        st.success(email_status)
    else:
        st.warning(email_status)

    if recipients:
        st.info(f"Notification will be sent to {len(recipients)} recipient(s) after a successful inference.")
    elif recipients_valid:
        st.warning("Enter at least one recipient to send a notification after inference.")

    email_result = st.session_state.pop("prediction_email_result", None)
    if email_result:
        if email_result.get("sent"):
            st.success(f"Prediction email sent to {', '.join(email_result.get('recipients', []))}.")
        else:
            st.warning(f"Prediction email was not sent: {email_result.get('error')}")

    if st.button("Generate New Inference", type="primary"):
        before_count = count_ticker_predictions(df_log, selected_ticker)
        with st.spinner("Running model..."):
            result = run_script("scripts/model_prediction.py", ["-ticker", selected_ticker, "--fetch"])
            if result.returncode != 0:
                st.error("Prediction failed. Check the predictor logs for details.")
                if result.stderr:
                    st.code(result.stderr)
                st.stop()

            updated_log = load_prediction_log(log_path)
            after_count = count_ticker_predictions(updated_log, selected_ticker)
            latest_prediction = get_latest_prediction(updated_log, selected_ticker)

            if after_count <= before_count or not latest_prediction:
                st.session_state["prediction_email_result"] = {
                    "sent": False,
                    "recipients": recipients,
                    "error": "Prediction completed, but no new prediction log row was found.",
                }
            elif recipients and recipients_valid:
                st.session_state["prediction_email_result"] = send_prediction_email(
                    email_config=email_config,
                    recipients=recipients,
                    prediction=latest_prediction,
                    ticker=selected_ticker,
                )
            else:
                st.session_state["prediction_email_result"] = {
                    "sent": False,
                    "recipients": recipients,
                    "error": "No valid email recipients were provided.",
                }
            st.rerun()

# --- Tab 2: Performance ---
with tab_perf:
    st.markdown('<p class="section-header">Strategy Analytics & Risk Attribution</p>', unsafe_allow_html=True)
    
    fig_dir = Path(config["data"]["paths"]["figures_dir"]).expanduser()
    bt_files = list(fig_dir.glob(f"backtest_results_{selected_ticker}_*.csv"))
    
    if bt_files:
        latest_bt = max(bt_files, key=os.path.getctime)
        df_bt = pd.read_csv(latest_bt)
        df_bt['date'] = pd.to_datetime(df_bt['date'])

        # Robust Column Mapping (Handles legacy 'portfolio_value' or new 'strategy_value')
        if 'strategy_value' not in df_bt.columns and 'portfolio_value' in df_bt.columns:
            df_bt['strategy_value'] = df_bt['portfolio_value']

        required_cols = ['strategy_value', 'buy_hold_value']
        if all(col in df_bt.columns for col in required_cols):
            # 1. Equity Curve
            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['strategy_value'], name='Strategy', line=dict(color='#2563eb', width=2.5)))
            fig_equity.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['buy_hold_value'], name='B&H', line=dict(color='#94a3b8', dash='dash')))
            fig_equity.update_layout(title="Cumulative Equity", hovermode="x unified", template="plotly_white", height=400)
            st.plotly_chart(fig_equity, use_container_width=True)

            # Metrics Implementation
            metrics_file = fig_dir / latest_bt.name.replace("results", "metrics").replace(".csv", ".json")
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    m = json.load(f)
                
                alpha = m.get('alpha', 0)
                bg_color = "#9f1239" if alpha < 0 else "#065f46"
                st.markdown(f"""<style>div[data-testid="stMetric"] {{ background-color: {bg_color} !important; border: none !important; }} div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] {{ color: #ffffff !important; }}</style>""", unsafe_allow_html=True)

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Total Return", f"{m.get('strategy_return', 0)*100:.1f}%", delta=f"{(m.get('strategy_return', 0) - m.get('buy_hold_return', 0))*100:.1f}% vs B&H")
                p2.metric("Sharpe", f"{m.get('sharpe_ratio', 0):.2f}")
                p3.metric("Max DD", f"{abs(m.get('max_draw_down', 0))*100:.1f}%")
                p4.metric("Alpha", f"{alpha:.4f}")

            # 2. Interactive Drawdown Chart
            st.markdown("### Risk Analytics")
            if 'drawdown' not in df_bt.columns:
                peak = df_bt['strategy_value'].cummax()
                df_bt['drawdown'] = (df_bt['strategy_value'] - peak) / peak

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['drawdown'] * 100, fill='tozeroy', name='Drawdown', line=dict(color='#ef4444')))
            fig_dd.update_layout(title="Strategy Drawdown (%)", yaxis_title="Drawdown %", template="plotly_white", height=300)
            st.plotly_chart(fig_dd, use_container_width=True)

            # 3. Interactive Trade Distribution
            st.markdown("### Execution Edge")
            trades = df_bt[df_bt['strategy_return'] != 0].copy()
            if not trades.empty:
                fig_dist = px.histogram(trades, x="strategy_return", nbins=50, title="Trade Return Distribution",
                                       color_discrete_sequence=['#10b981'], labels={'strategy_return': 'Return'})
                fig_dist.add_vline(x=0, line_dash="dash", line_color="#64748b")
                fig_dist.update_layout(template="plotly_white", height=350)
                st.plotly_chart(fig_dist, use_container_width=True)

                # 4. Interactive Trade Timeline
                st.markdown("### Trade Timeline")
                trades['outcome'] = np.where(trades['strategy_return'] > 0, 'Win', 'Loss')
                fig_timeline = px.bar(trades, x="date", y="strategy_return", color="outcome",
                                     title="Individual Trade Outcomes",
                                     color_discrete_map={'Win': '#10b981', 'Loss': '#ef4444'},
                                     labels={'strategy_return': 'Return %', 'date': 'Date'})
                fig_timeline.update_layout(template="plotly_white", height=350, showlegend=False)
                st.plotly_chart(fig_timeline, use_container_width=True)
            else:
                st.info("No active trades in this backtest window.")
        else:
            missing = [c for c in required_cols if c not in df_bt.columns]
            st.error(f"Missing columns for {selected_ticker}: {', '.join(missing)}")
            st.info("Please run the model pipeline to regenerate results for this ticker.")
    else:
        st.warning(f"No backtest results found for {selected_ticker}. Run the model pipeline first.")

# --- Tab 3: Regimes ---
with tab_regime:
    st.markdown('<p class="section-header">Market Regime Analysis</p>', unsafe_allow_html=True)
    proc_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()
    cluster_stats_path = proc_dir / "cluster_statistics.json"
    if cluster_stats_path.exists():
        with open(cluster_stats_path, 'r') as f:
            stats = json.load(f)
        st.dataframe(pd.DataFrame(stats).style.background_gradient(cmap='RdYlGn', subset=['overnight_delta_mean']), use_container_width=True)
