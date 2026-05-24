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

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True, port=5001)
