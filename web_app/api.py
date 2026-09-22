from datetime import date, datetime
from pathlib import Path
import math
import re

from flask import Blueprint, current_app, jsonify, request

from qusa.utils.config import load_config
from qusa.storage.locks import TickerBusyError
from qusa.storage.runs import RunRepository


api_bp = Blueprint("api", __name__, url_prefix="/api")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


_SENSITIVE_QUERY_VALUE = re.compile(
    r"(?i)(api[_-]?key|token|secret|password)=([^&\s]+)"
)


class RequestValidationError(ValueError):
    """A request error that is safe to return to an API caller."""


def _error_response(message, status_code=400, code=None):
    payload = {"success": False, "error": message}
    if code:
        payload["code"] = code
    return jsonify(payload), status_code


def _redact_sensitive_text(value):
    return _SENSITIVE_QUERY_VALUE.sub(r"\1=[REDACTED]", str(value))


def _service_error_response(operation, exc):
    current_app.logger.error("%s failed: %s", operation, _redact_sensitive_text(exc))
    return _error_response(
        f"{operation} could not be completed.", 500, f"{operation}_failed"
    )


def _request_json():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise RequestValidationError("Request body must be a JSON object.")
    if not isinstance(payload.get("fetch_latest", False), bool):
        raise RequestValidationError("fetch_latest must be a boolean.")
    return payload


def _require_ticker(payload):
    ticker = payload.get("ticker")
    if not isinstance(ticker, str) or not ticker.strip():
        raise RequestValidationError("Ticker is required and must be a string.")
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z0-9^][A-Z0-9.^-]*", ticker):
        raise RequestValidationError("Ticker contains invalid characters.")
    return ticker


def _load_app_config():
    return load_config(DEFAULT_CONFIG_PATH)


def _available_tickers():
    config = _load_app_config()
    raw_dir = Path(config["data"]["paths"]["raw_data_dir"]).expanduser()
    if not raw_dir.exists():
        return []

    tickers = {
        path.stem.removesuffix("_history").upper()
        for path in raw_dir.glob("*_history.csv")
        if path.is_file()
    }
    return sorted(tickers)


def _prediction_repository():
    return RunRepository.from_config(_load_app_config())


@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"})


@api_bp.route("/tickers", methods=["GET"])
def tickers():
    try:
        return jsonify({"success": True, "tickers": _available_tickers()})
    except TickerBusyError:
        return _error_response("Ticker is currently busy. Try again shortly.", 409, "ticker_busy")
    except Exception as exc:
        return _service_error_response("ticker_lookup", exc)


@api_bp.route("/pipeline/run", methods=["POST"])
def run_pipeline():
    try:
        payload = _request_json()
        ticker = _require_ticker(payload)
        from qusa.services import run_feature_pipeline

        result = run_feature_pipeline(
            ticker=ticker,
            fetch_latest=bool(payload.get("fetch_latest", False)),
            config_path=DEFAULT_CONFIG_PATH,
        )
        return jsonify(_json_safe(result))
    except RequestValidationError as exc:
        return _error_response(str(exc), 400, "invalid_request")
    except TickerBusyError:
        return _error_response("Ticker is currently busy. Try again shortly.", 409, "ticker_busy")
    except Exception as exc:
        return _service_error_response("feature_pipeline", exc)


@api_bp.route("/predictions/run", methods=["POST"])
def run_prediction():
    try:
        payload = _request_json()
        ticker = _require_ticker(payload)
        volatility = payload.get("volatility")
        volatility_override = None
        if volatility is not None:
            if isinstance(volatility, bool) or not isinstance(volatility, (int, float)):
                raise RequestValidationError("volatility must be a finite non-negative number.")
            volatility_override = float(volatility)
            if not math.isfinite(volatility_override) or volatility_override < 0:
                raise RequestValidationError("volatility must be a finite non-negative number.")
        from qusa.services import make_latest_prediction

        result = make_latest_prediction(
            ticker=ticker,
            fetch_latest=bool(payload.get("fetch_latest", False)),
            volatility_override=volatility_override,
            config_path=DEFAULT_CONFIG_PATH,
        )
        return jsonify(_json_safe(result))
    except RequestValidationError as exc:
        return _error_response(str(exc), 400, "invalid_request")
    except TickerBusyError:
        return _error_response("Ticker is currently busy. Try again shortly.", 409, "ticker_busy")
    except FileNotFoundError as exc:
        current_app.logger.warning("prediction artifact missing: %s", _redact_sensitive_text(exc))
        return _error_response(
            "Required prediction artifact was not found.", 404, "artifact_not_found"
        )
    except Exception as exc:
        return _service_error_response("prediction", exc)


@api_bp.route("/predictions/history", methods=["GET"])
def prediction_history():
    try:
        ticker = request.args.get("ticker", "").strip().upper()
        limit = _bounded_int(request.args.get("limit"), default=50, maximum=100)
        offset = _bounded_int(request.args.get("offset"), default=0, maximum=1_000_000)
        history = _prediction_repository().list_predictions(ticker=ticker or None, limit=limit, offset=offset)
        return jsonify({"success": True, "history": _json_safe(history), "next_offset": offset + len(history) if len(history) == limit else None})
    except RequestValidationError as exc:
        return _error_response(str(exc), 400, "invalid_request")
    except Exception as exc:
        return _service_error_response("prediction_history", exc)


def _bounded_int(raw_value, default, maximum):
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise RequestValidationError("Pagination values must be integers.") from exc
    if value < 0 or value > maximum:
        raise RequestValidationError(f"Pagination values must be between 0 and {maximum}.")
    return value


@api_bp.route("/predictions/latest", methods=["GET"])
def latest_prediction():
    try:
        ticker = request.args.get("ticker", "").strip().upper()
        prediction = _prediction_repository().latest_prediction(ticker=ticker or None)
        return jsonify({"success": True, "prediction": _json_safe(prediction)})
    except Exception as exc:
        return _service_error_response("latest_prediction", exc)


@api_bp.route("/runs/<run_id>", methods=["GET"])
def run_detail(run_id):
    try:
        run = _prediction_repository().get_run(run_id)
        if run is None:
            return _error_response("Run was not found.", 404, "run_not_found")
        return jsonify({"success": True, "run": _json_safe(run)})
    except Exception as exc:
        return _service_error_response("run_detail", exc)
