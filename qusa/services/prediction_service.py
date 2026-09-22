from datetime import datetime
from pathlib import Path

import pandas as pd

from qusa.model import make_prediction
from qusa.services.pipeline_service import DEFAULT_CONFIG_PATH, run_feature_pipeline
from qusa.utils.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def save_prediction_log(prediction_data, log_file_path):
    """
    Save prediction to CSV log.
    """

    log_path = Path(log_file_path).expanduser()
    prediction = pd.DataFrame([prediction_data])

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    except Exception as exc:
        raise IOError(f"Failed to save prediction to {log_file_path}: {exc}") from exc


def _build_volatility_filter(config, volatility_override):
    vol_filter = config["backtest"].get("volatility_filter", {"enabled": False}).copy()
    if volatility_override is not None:
        vol_filter["enabled"] = True
        vol_filter["max_atr_pct"] = volatility_override
    return vol_filter


def make_latest_prediction(
    ticker,
    fetch_latest=False,
    volatility_override=None,
    config_path=None,
    logger=None,
):
    """
    Make and optionally log the latest prediction for one ticker.
    """

    ticker = ticker.upper()
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)
    config = load_config(config_path)

    model_dir = Path(config["model"]["output"]["model_output_path"]).expanduser()
    data_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()
    model_path = model_dir / f"{ticker.lower()}_model.pkl"
    processed_data_path = data_dir / f"{ticker}_processed.csv"

    if fetch_latest:
        if logger:
            logger.info(f"--fetch enabled: preparing data for {ticker}...")
        run_feature_pipeline(
            ticker,
            fetch_latest=True,
            config_path=config_path,
            logger=logger,
        )

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not processed_data_path.exists():
        raise FileNotFoundError(f"Data not found at {processed_data_path}")

    vol_filter = _build_volatility_filter(config, volatility_override)
    if volatility_override is not None and logger:
        logger.info(f"Using command-line volatility filter: {volatility_override}%")

    if logger:
        logger.info(
            f"Predicting using model at {model_path} and data at {processed_data_path}"
        )

    prediction = make_prediction(
        str(model_path),
        str(processed_data_path),
        ticker=ticker,
        logger_obj=logger,
        volatility_filter=vol_filter,
    )

    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ticker": ticker,
        "date": prediction.get("date", "Unknown"),
        "prediction": prediction.get("prediction"),
        "direction": prediction.get("direction"),
        "probability_up": prediction.get("probability_up"),
        "confidence": prediction.get("confidence"),
        "atr_pct": prediction.get("atr_pct"),
        "volatility_filter_triggered": prediction.get("volatility_filter_triggered"),
        "volatility_state": prediction.get("volatility_state"),
        "volatility_threshold": prediction.get("volatility_threshold"),
        "model_id": prediction.get("model_id"),
    }

    should_save = config.get("prediction", {}).get("save", True)
    prediction_csv_file = config.get("prediction", {}).get("csv_log")

    if logger:
        logger.info(
            f"Prediction for {ticker}: {prediction.get('direction')} "
            f"({prediction.get('confidence')} Confidence)"
        )

    if should_save and prediction_csv_file:
        save_prediction_log(log_entry, prediction_csv_file)
        if logger:
            logger.info(f"Prediction appended to log: {prediction_csv_file}")

    return {
        "success": True,
        "ticker": ticker,
        "prediction": prediction,
        "log_entry": log_entry,
        "model_path": str(model_path),
        "processed_data_path": str(processed_data_path),
        "prediction_log_path": prediction_csv_file,
        "volatility_filter": vol_filter,
    }
