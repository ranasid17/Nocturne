# flask_dashboard/app.py

import os
import sys
import subprocess
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, jsonify, request

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from qusa.notifications import parse_recipients, send_prediction_email
from qusa.utils.config import load_config, load_env

app = Flask(__name__)

# Load environment variables from .env if it exists
load_env(PROJECT_ROOT / ".env")

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
    log_path = Path(log_path).expanduser()
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

    return df_ticker.iloc[0].replace({np.nan: None}).to_dict()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/prediction/<ticker>")
def api_prediction(ticker):
    config = get_config()
    log_path = Path(config["prediction"].get("csv_log", config["prediction"].get("log_file"))).expanduser()
    df_log = load_prediction_log(log_path)
    
    latest = get_latest_prediction(df_log, ticker.upper())
    history = []
    if not df_log.empty:
        df_ticker = df_log[df_log["ticker"] == ticker.upper()]
        history = df_ticker.head(15).replace({np.nan: None}).to_dict(orient="records")
        
    return jsonify({
        "latest": latest,
        "history": history
    })

@app.route("/api/performance/<ticker>")
def api_performance(ticker):
    config = get_config()
    fig_dir = Path(config["data"]["paths"]["figures_dir"]).expanduser()
    bt_files = list(fig_dir.glob(f"backtest_results_{ticker.upper()}_*.csv"))
    
    if not bt_files:
        return jsonify({"error": f"No backtest results found for {ticker.upper()}"}), 404

    latest_bt = max(bt_files, key=os.path.getctime)
    df_bt = pd.read_csv(latest_bt)
    df_bt['date'] = pd.to_datetime(df_bt['date'])

    # Standardize column names
    if 'strategy_value' not in df_bt.columns and 'portfolio_value' in df_bt.columns:
        df_bt['strategy_value'] = df_bt['portfolio_value']

    # Drawdown
    peak = df_bt['strategy_value'].cummax()
    df_bt['drawdown'] = (df_bt['strategy_value'] - peak) / peak

    # Metrics
    metrics_file = fig_dir / latest_bt.name.replace("results", "metrics").replace(".csv", ".json")
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

    # Prepare chart data (simple version for now, could use plotly.utils.PlotlyJSONEncoder)
    # But for a cleaner API, we can just return the data series and build the chart in JS.
    # However, to maintain parity with Streamlit logic, we'll return structured data.
    
    performance_data = {
        "dates": df_bt['date'].dt.strftime("%Y-%m-%d").tolist(),
        "strategy_value": df_bt['strategy_value'].tolist(),
        "buy_hold_value": df_bt['buy_hold_value'].tolist(),
        "drawdown": (df_bt['drawdown'] * 100).tolist(),
        "metrics": metrics
    }
    
    return jsonify(performance_data)

@app.route("/api/tickers")
def api_tickers():
    config = get_config()
    available_tickers = []
    raw_dir = Path(config["data"]["paths"]["raw_data_dir"]).expanduser()
    if raw_dir.exists():
        available_tickers = sorted(list(set([f.stem.split('_')[0] for f in raw_dir.glob("*_history.csv")])))
    return jsonify(available_tickers)

@app.route("/api/run-inference", methods=["POST"])
def run_inference():
    data = request.json
    ticker = data.get("ticker")
    recipients = data.get("recipients", [])
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    # For now, run synchronously but in a real app this should be async
    result = run_script("scripts/model_prediction.py", ["-ticker", ticker, "--fetch"])
    
    if result.returncode != 0:
        return jsonify({"error": "Prediction failed", "details": result.stderr}), 500

    # Logic to send email if recipients provided
    if recipients:
        config = get_config()
        updated_log = load_prediction_log(Path(config["prediction"].get("csv_log")).expanduser())
        latest_prediction = get_latest_prediction(updated_log, ticker.upper())
        
        if latest_prediction:
            email_result = send_prediction_email(
                email_config=config.get("email"),
                recipients=recipients,
                prediction=latest_prediction,
                ticker=ticker.upper(),
            )
            return jsonify({"status": "success", "email": email_result})

    return jsonify({"status": "success"})

@app.route("/api/refresh-pipeline", methods=["POST"])
def refresh_pipeline():
    data = request.json
    ticker = data.get("ticker")
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    result = run_script("scripts/run_FE_pipeline.py", ["-ticker", ticker, "--fetch"])
    
    if result.returncode != 0:
        return jsonify({"error": "Pipeline refresh failed", "details": result.stderr}), 500

    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(debug=True, port=5001)
