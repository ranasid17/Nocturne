from flask import Blueprint, current_app, render_template

from web_app.api import _available_tickers
from qusa.utils.errors import safe_error


pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def dashboard():
    error = None
    try:
        tickers = _available_tickers()
    except Exception as exc:
        current_app.logger.error("Ticker discovery failed: %s", safe_error(exc))
        tickers = []
        error = "Available tickers could not be loaded."
    return render_template("dashboard.html", tickers=tickers, ticker_error=error)
