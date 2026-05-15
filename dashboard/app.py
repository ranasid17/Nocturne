# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from qusa.utils.config import load_config

# --- Page Config ---
st.set_page_config(
    page_title="QUSA Command Center",
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
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
    }

    /* Metric Label/Value Colors */
    div[data-testid="stMetricLabel"] {
        font-size: 0.875rem !important;
        font-weight: 600 !important;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.025em;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.875rem !important;
        font-weight: 700 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }

    .stTabs [aria-selected="true"] {
        background-color: transparent;
        border-bottom: 2px solid #2563eb !important;
        color: #2563eb !important;
        font-weight: 700;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 1rem;
        margin-top: 2rem;
        border-left: 4px solid #2563eb;
        padding-left: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data
def get_config():
    config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
    return load_config(str(config_path))

def run_script(script_path, args):
    """Run a QUSA script via subprocess."""
    cmd = [sys.executable, str(PROJECT_ROOT / script_path)] + args
    result = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
    
    if result.returncode == 0 and result.stderr:
        if "UserWarning" in result.stderr and "Error" not in result.stderr and "Exception" not in result.stderr:
            result.stderr = "" 
            
    return result

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/combo-chart.png", width=64)
    st.title("QUSA Core")
    st.caption("Quantitative US Stock Analysis")
    
    st.divider()
    
    config = get_config()
    
    # Ticker Selection
    available_tickers = []
    raw_dir = Path(config["data"]["paths"]["raw_data_dir"]).expanduser()
    if raw_dir.exists():
        available_tickers = [f.stem.split('_')[0] for f in raw_dir.glob("*_history.csv")]
    
    selected_ticker = st.selectbox("Market Ticker", options=sorted(list(set(available_tickers))) if available_tickers else ["UPRO"])
    
    st.divider()
    st.subheader("Operations")
    if st.button("🔄 Sync & Refresh Data", use_container_width=True):
        with st.spinner(f"Synchronizing {selected_ticker}..."):
            res = run_script("scripts/run_FE_pipeline.py", ["-ticker", selected_ticker, "--fetch"])
            if res.returncode == 0:
                st.success("Synchronized")
            else:
                st.error(f"Failed: {res.stderr}")

    st.divider()
    st.info("Version 0.2.0-Alpha | System Stable")

# --- Main Content ---
st.title("📈 Command Center")
st.caption(f"Security: {selected_ticker} | Intelligence Active | {datetime.now().strftime('%H:%M:%S UTC')}")

tab_predict, tab_perf, tab_regime = st.tabs([
    "🎯 Predictive Signals", 
    "📊 Performance Matrix", 
    "🧩 Market Regimes"
])

# --- Tab 1: Predictions ---
with tab_predict:
    col1, col2 = st.columns([1, 2], gap="large")
    
    with col1:
        st.markdown('<p class="section-header">Operational Signal</p>', unsafe_allow_html=True)
        if st.button("Generate Live Inference", type="primary", use_container_width=True):
            with st.spinner("Processing Model..."):
                res = run_script("scripts/model_prediction.py", ["-ticker", selected_ticker, "--fetch"])
                if res.returncode == 0:
                    st.toast("Inference Generated")
                else:
                    st.error(f"Error: {res.stderr}")
        
        st.markdown("### Execution Log")
        log_path = Path(config["prediction"].get("csv_log", config["prediction"].get("log_file"))).expanduser()
        if log_path.exists():
            df_log = pd.read_csv(log_path).sort_values("timestamp", ascending=False)
            st.dataframe(df_log.head(10), use_container_width=True, hide_index=True)
        else:
            st.warning("No signal history found.")

    with col2:
        st.markdown('<p class="section-header">Latest Intelligence</p>', unsafe_allow_html=True)
        if log_path.exists():
            df_ticker = df_log[df_log['ticker'] == selected_ticker]
            if not df_ticker.empty:
                latest = df_ticker.iloc[0]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Bias", latest['direction'])
                m2.metric("Probability", f"{latest['probability_up']:.1%}")
                m3.metric("Confidence", latest['confidence'])
                
                if latest['confidence'] == 'HIGH':
                    color = "green" if "UP" in latest['direction'] else "red"
                    st.success(f"**STRATEGIC {latest['direction']} OPPORTUNITY IDENTIFIED**")
                else:
                    st.info("No high-conviction signal present.")
            else:
                st.info(f"No recent intelligence for {selected_ticker}")

# --- Tab 2: Performance ---
with tab_perf:
    st.markdown('<p class="section-header">Equity Performance & Strategy Attribution</p>', unsafe_allow_html=True)
    
    fig_dir = Path(config["data"]["paths"]["figures_dir"]).expanduser()
    bt_files = list(fig_dir.glob(f"backtest_results_{selected_ticker}_*.csv"))
    
    if bt_files:
        latest_bt = max(bt_files, key=os.path.getctime)
        df_bt = pd.read_csv(latest_bt)
        df_bt['date'] = pd.to_datetime(df_bt['date'])

        required_cols = ['strategy_value', 'buy_hold_value']
        if all(col in df_bt.columns for col in required_cols):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['strategy_value'], name='Active Strategy', line=dict(color='#2563eb', width=2)))
            fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['buy_hold_value'], name='Benchmark (B&H)', line=dict(color='#94a3b8', dash='dash')))

            fig.update_layout(
                title=dict(text="Cumulative Strategy Performance", font=dict(size=18, weight='bold')),
                xaxis_title="Date",
                yaxis_title="Equity Value ($)",
                hovermode="x unified",
                template="plotly_white",
                margin=dict(l=20, r=20, t=60, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("Incomplete data in latest results.")

        # Metrics Implementation
        metrics_file = fig_dir / latest_bt.name.replace("results", "metrics").replace(".csv", ".json")
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                m = json.load(f)
            
            strat_ret = m.get('strategy_return', 0)
            bh_ret = m.get('buy_hold_return', 0)
            alpha = m.get('alpha', 0)
            sharpe = m.get('sharpe_ratio', 0)
            mdd = m.get('max_draw_down', 0)

            # High-Contrast Professional Metric Colors
            # Using deep emerald (#065f46) and deep rose (#9f1239) for readability
            bg_color = "#9f1239" if alpha < 0 else "#065f46"
            text_color = "#ffffff"
            
            st.markdown(f"""
                <style>
                div[data-testid="stMetric"] {{
                    background-color: {bg_color} !important;
                    border: none !important;
                }}
                div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"], div[data-testid="stMetricDelta"] {{
                    color: {text_color} !important;
                }}
                </style>
            """, unsafe_allow_html=True)

            p1, p2, p3, p4 = st.columns(4)
            p1.metric("Strategy Return", f"{strat_ret*100:.1f}%", delta=f"{(strat_ret - bh_ret)*100:.1f}% vs Bench")
            p2.metric("Sharpe Ratio", f"{sharpe:.2f}", delta=f"{sharpe:.2f}", delta_color="off")
            p3.metric("Max Drawdown", f"{abs(mdd)*100:.1f}%", delta=f"{mdd*100:.1f}%", delta_color="off")
            p4.metric("Alpha (Active)", f"{alpha:.4f}", delta=f"{alpha:.4f}", delta_color="off")

        # Static Figures Injection (Drawdown curves, etc.)
        st.markdown('<p class="section-header">Advanced Attribution Plots</p>', unsafe_allow_html=True)
        
        # Look for the generated PNG plot
        timestamp = latest_bt.stem.split('_')[-1]
        static_plot = fig_dir / f"backtest_plot_{selected_ticker}_{timestamp}.png"
        
        if static_plot.exists():
            st.image(str(static_plot), caption="Comprehensive Backtest Visualization", use_container_width=True)
        else:
            # Fallback to any recent plots for this ticker
            other_plots = list(fig_dir.glob(f"backtest_plot_{selected_ticker}_*.png"))
            if other_plots:
                latest_plot = max(other_plots, key=os.path.getctime)
                st.image(str(latest_plot), caption="Most Recent Strategy Attribution Plot", use_container_width=True)
            else:
                st.info("Advanced attribution plots will appear here after the next pipeline run.")

    else:
        st.warning(f"No execution data found for {selected_ticker}.")

# --- Tab 3: Regimes ---
with tab_regime:
    st.markdown('<p class="section-header">Market Regime Classification</p>', unsafe_allow_html=True)
    
    proc_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()
    cluster_stats_path = proc_dir / "cluster_statistics.json"
    
    if cluster_stats_path.exists():
        with open(cluster_stats_path, 'r') as f:
            stats = json.load(f)
        
        df_stats = pd.DataFrame(stats)
        st.dataframe(
            df_stats.style.background_gradient(cmap='RdYlGn', subset=['overnight_delta_mean'])
            .format(precision=4), 
            use_container_width=True
        )
        st.info("💡 Clusters represent distinct market environments identified through unsupervised DBSCAN/K-Means clustering.")
    else:
        st.warning("Regime statistics unavailable.")
