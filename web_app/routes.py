from flask import Blueprint, current_app, render_template

from web_app.api import _available_tickers


pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def dashboard():
    error = None
    try:
        tickers = _available_tickers()
    except Exception:
        current_app.logger.exception("Ticker discovery failed")
        tickers = []
        error = "Available tickers could not be loaded."
    return render_template("dashboard.html", tickers=tickers, ticker_error=error)
