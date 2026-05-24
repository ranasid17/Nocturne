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

# Global task storage (simple dictionary for MVP)
tasks = {}

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

    task_id = f"inference_{ticker}_{datetime.now().strftime('%H%M%S')}"
    
    # Run asynchronously
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/model_prediction.py"), "-ticker", ticker, "--fetch"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
    
    tasks[task_id] = {
        "process": process,
        "ticker": ticker.upper(),
        "recipients": recipients,
        "type": "inference"
    }
    
    return jsonify({"task_id": task_id})

@app.route("/api/refresh-pipeline", methods=["POST"])
def refresh_pipeline():
    data = request.json
    ticker = data.get("ticker")
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400

    task_id = f"pipeline_{ticker}_{datetime.now().strftime('%H%M%S')}"
    
    # Run asynchronously
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/run_FE_pipeline.py"), "-ticker", ticker, "--fetch"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT)})
    
    tasks[task_id] = {
        "process": process,
        "ticker": ticker.upper(),
        "type": "pipeline"
    }
    
    return jsonify({"task_id": task_id})

@app.route("/api/task-status/<task_id>")
def task_status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({"error": "Task not found"}), 404
    
    process = task["process"]
    return_code = process.poll()
    
    if return_code is None:
        return jsonify({"status": "running"})
    
    # Task finished
    stdout, stderr = process.communicate()
    
    if return_code != 0:
        return jsonify({"status": "failed", "error": stderr})

    # If it was an inference and had recipients, send email
    email_result = None
    if task["type"] == "inference" and task.get("recipients"):
        config = get_config()
        log_path = Path(config["prediction"].get("csv_log", config["prediction"].get("log_file"))).expanduser()
        updated_log = load_prediction_log(log_path)
        latest_prediction = get_latest_prediction(updated_log, task["ticker"])
        
        if latest_prediction:
            email_result = send_prediction_email(
                email_config=config.get("email"),
                recipients=task["recipients"],
                prediction=latest_prediction,
                ticker=task["ticker"],
            )

    return jsonify({"status": "success", "email": email_result})

@app.route("/api/regimes")
def api_regimes():
    config = get_config()
    proc_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()
    cluster_stats_path = proc_dir / "cluster_statistics.json"
    
    if not cluster_stats_path.exists():
        return jsonify({"error": "Regime statistics not found"}), 404

    with open(cluster_stats_path, 'r') as f:
        stats = json.load(f)
        
    return jsonify(stats)

if __name__ == "__main__":
    app.run(debug=True, port=5001)
