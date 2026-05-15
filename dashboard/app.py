# dashboard/app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess
import sys
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

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
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
    
    # If successful but has warnings in stderr, we filter them for cleaner UI feedback
    if result.returncode == 0 and result.stderr:
        # Check if it's just warnings or something critical
        if "UserWarning" in result.stderr and "Error" not in result.stderr and "Exception" not in result.stderr:
            result.stderr = "" # Clear noise for the UI
            
    return result

# --- Sidebar ---
with st.sidebar:
    st.title("⚙️ Settings")
    
    config = get_config()
    
    # Ticker Selection
    available_tickers = []
    raw_dir = Path(config["data"]["paths"]["raw_data_dir"]).expanduser()
    if raw_dir.exists():
        available_tickers = [f.stem.split('_')[0] for f in raw_dir.glob("*_history.csv")]
    
    selected_ticker = st.selectbox("Select Ticker", options=sorted(list(set(available_tickers))) if available_tickers else ["UPRO"])
    
    st.divider()
    st.subheader("🚀 Quick Actions")
    if st.button("Fetch & Process Latest Data"):
        with st.spinner(f"Fetching data for {selected_ticker}..."):
            res = run_script("scripts/run_FE_pipeline.py", ["-ticker", selected_ticker, "--fetch"])
            if res.returncode == 0:
                st.success("Data updated successfully!")
            else:
                st.error(f"Failed: {res.stderr}")

    st.divider()
    st.info("QUSA Quantitative Analysis Framework v0.1")

# --- Main Content ---
st.title("📉 QUSA Command Center")
st.caption(f"Analyzing {selected_ticker} | Last Refresh: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

tab_predict, tab_perf, tab_regime = st.tabs([
    "🎯 Live Predictions", 
    "📊 Performance Analysis", 
    "🧩 Regime Discovery"
])

# --- Tab 1: Predictions ---
with tab_predict:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("New Prediction")
        if st.button("Run Live Prediction", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                res = run_script("scripts/model_prediction.py", ["-ticker", selected_ticker, "--fetch"])
                if res.returncode == 0:
                    st.toast("Prediction complete!")
                else:
                    st.error(f"Error: {res.stderr}")
        
        st.write("Recent Activity")
        log_path = Path(config["prediction"].get("csv_log", config["prediction"].get("log_file"))).expanduser()
        if log_path.exists():
            df_log = pd.read_csv(log_path).sort_values("timestamp", ascending=False)
            st.dataframe(df_log.head(10), use_container_width=True, hide_index=True)
        else:
            st.warning("No prediction log found.")

    with col2:
        st.subheader("Latest Signal")
        if log_path.exists():
            # Get latest for this ticker
            df_ticker = df_log[df_log['ticker'] == selected_ticker]
            if not df_ticker.empty:
                latest = df_ticker.iloc[0]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Direction", latest['direction'])
                m2.metric("Probability", f"{latest['probability_up']:.1%}")
                m3.metric("Confidence", latest['confidence'])
                
                # Visual Signal
                if latest['confidence'] == 'HIGH':
                    color = "green" if "UP" in latest['direction'] else "red"
                    st.success(f"### 🔥 STRONG {'BUY' if color=='green' else 'SELL'} SIGNAL IDENTIFIED")
                else:
                    st.info("### ⚖️ AMBIGUOUS - No strong signal")
            else:
                st.info(f"No recent predictions for {selected_ticker}")

# --- Tab 2: Performance ---
with tab_perf:
    st.subheader("Equity Curve & Strategy Metrics")
    
    # Load backtest results
    fig_dir = Path(config["data"]["paths"]["figures_dir"]).expanduser()
    bt_files = list(fig_dir.glob(f"backtest_results_{selected_ticker}_*.csv"))
    
    if bt_files:
        # Get most recent
        latest_bt = max(bt_files, key=os.path.getctime)
        df_bt = pd.read_csv(latest_bt)
        df_bt['date'] = pd.to_datetime(df_bt['date'])

        # Defensive check for required columns
        required_cols = ['strategy_value', 'buy_hold_value']
        missing_cols = [c for c in required_cols if c not in df_bt.columns]

        if not missing_cols:
            # Interactive Plotly Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['strategy_value'], name='Strategy', line=dict(color='#2E86C1')))
            fig.add_trace(go.Scatter(x=df_bt['date'], y=df_bt['buy_hold_value'], name='Market (Buy & Hold)', line=dict(color='#ABB2B9', dash='dash')))

            fig.update_layout(
                title=f"Portfolio Performance: {selected_ticker}",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Selected backtest file is missing required data columns: {', '.join(missing_cols)}")
            st.info("Try running the model pipeline for this ticker again to generate fresh results.")

        # Metrics Row

        # Try to find matching JSON metrics
        metrics_file = fig_dir / latest_bt.name.replace("results", "metrics").replace(".csv", ".json")
        if metrics_file.exists():
            import json
            with open(metrics_file, 'r') as f:
                m = json.load(f)
            
            p1, p2, p3, p4 = st.columns(4)
            
            # Extract metrics for coloring
            strat_ret = m.get('strategy_return', 0)
            bh_ret = m.get('buy_hold_return', 0)
            alpha = m.get('alpha', 0)
            sharpe = m.get('sharpe_ratio', 0)
            mdd = m.get('max_draw_down', 0)

            # Dynamic background color based on Alpha
            bg_color = "#f8d7da" if alpha < 0 else "#d4edda"
            st.markdown(f"""
                <style>
                div[data-testid="stMetric"] {{
                    background-color: {bg_color} !important;
                }}
                </style>
            """, unsafe_allow_html=True)

            p1, p2, p3, p4 = st.columns(4)

                "Total Return", 
                f"{strat_ret*100:.1f}%", 
                delta=f"{(strat_ret - bh_ret)*100:.1f}% vs B&H"
            )
            p2.metric(
                "Sharpe Ratio", 
                f"{sharpe:.2f}", 
                delta=f"{sharpe:.2f}", 
                delta_color="off" if sharpe == 0 else "normal"
            )
            p3.metric(
                "Max Drawdown", 
                f"{abs(mdd)*100:.1f}%", 
                delta=f"{mdd*100:.1f}%", 
                delta_color="normal"
            )
            p4.metric(
                "Alpha", 
                f"{alpha:.4f}", 
                delta=f"{alpha:.4f}"
            )
    else:
        st.warning(f"No backtest results found for {selected_ticker}. Run the model pipeline first.")

# --- Tab 3: Regimes ---
with tab_regime:
    st.subheader("Market Regime Clustering")
    
    proc_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()
    cluster_stats_path = proc_dir / "cluster_statistics.json"
    
    if cluster_stats_path.exists():
        import json
        with open(cluster_stats_path, 'r') as f:
            stats = json.load(f)
        
        # Show profiles
        st.write("Cluster Profiles")
        df_stats = pd.DataFrame(stats)
        st.dataframe(df_stats.style.background_gradient(cmap='RdYlGn', subset=['overnight_delta_mean']), use_container_width=True)
        
        # PCA Visualization would go here if we saved the PCA coords
        st.info("💡 Interactive PCA mapping coming in next update. Current view shows statistical distribution across regimes.")
    else:
        st.warning("No cluster statistics found. Run the clustering pipeline first.")
