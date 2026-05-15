#!/usr/bin/env python3
# qusa/scripts/model_prediction.py

"""
Make prediction on the most recent trading day.
"""

import argparse
import pandas as pd
import sys

from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from qusa.model import make_prediction
from qusa.data.loader import DataLoader
from qusa.features.pipeline import FeaturePipeline
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make QUSA predictions for one or more tickers."
    )
    parser.add_argument(
        "-ticker", "--ticker",
        "--tickers",
        dest="tickers",
        nargs="+",
        required=True,
        help="Ticker symbol(s) to predict, for example -ticker AMZN AAPL",
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch latest data and run feature engineering before prediction.",
    )
    return parser.parse_args()


def save_prediction_log(prediction_data, log_file_path):
    """
    Save prediction to CSV log.

    Parameters:
        1) prediction_data (dict): Prediction details
        2) log_file_path (str): Path to log file
    """

    log_path = Path(log_file_path).expanduser()

    prediction = pd.DataFrame([prediction_data])

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    except Exception as e:
        raise IOError(f"Failed to save prediction to {log_file_path}: {e}")


def main():
    """
    Main function to make prediction.
    """

    args = parse_args()
    tickers = [t.upper() for t in args.tickers]

    try:
        config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
        config = load_config(str(config_path))

        log_file = config.get("prediction", {}).get("log_file", "logs/predictions.log")
        logger = setup_logger("predictor", log_file=log_file)

        logger.info("Configuration loaded successfully.")

    except IOError as e:
        print(f"✗ Configuration file not found: {e}")
        sys.exit(1)

    try:
        model_dir = Path(config["model"]["output"]["model_output_path"]).expanduser()
        data_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()

        should_save = config.get("prediction", {}).get("save", True)
        prediction_csv_file = config.get("prediction", {}).get("csv_log")

    except KeyError as e:
        logger.error(f"✗ Missing configuration key: {e}")
        sys.exit(1)

    success_count = 0

    for ticker in tickers:
        logger.info(f"{'=' * 40}")
        logger.info(f"Processing Ticker: {ticker}")

        try:
            model_path = model_dir / f"{ticker.lower()}_model.pkl"
            processed_data_path = data_dir / f"{ticker}_processed.csv"

            # ---------------------------------------------------------
            # Automated Fetching and Feature Engineering
            # ---------------------------------------------------------
            if args.fetch:
                logger.info(f"--fetch enabled: preparing data for {ticker}...")
                
                # 1. Fetch latest data
                raw_data_dir = config["data"]["paths"]["raw_data_dir"]
                loader = DataLoader(raw_data_dir=raw_data_dir)
                raw_data = loader.load_most_recent(ticker)
                
                # 2. Run feature engineering
                fe_pipeline = FeaturePipeline({
                    "date_col": "date",
                    "open_col": "open",
                    "close_col": "close",
                    "high_col": "high",
                    "low_col": "low",
                    "volume_col": "volume",
                    "overnight": {"abnormal_threshold": config["analysis"]["abnormal_threshold"]},
                    "technical_params": config["features"],
                    "monte_carlo": config.get("monte_carlo", {}),
                })
                processed_data = fe_pipeline.run(raw_data, ticker=ticker)
                
                # 3. Save processed data for prediction
                processed_data.to_csv(processed_data_path, index=False)
                logger.info(f"✓ Data prepared and saved to {processed_data_path}")

            if not model_path.exists():
                logger.warning(f"Skipping {ticker}: Model not found at {model_path}")
                continue
            if not processed_data_path.exists():
                logger.warning(f"Skipping {ticker}: Data not found at {processed_data_path}")
                continue

            logger.info(
                f"Predicting using model at {model_path} and data at {processed_data_path}"
            )
            prediction = make_prediction(
                str(model_path), 
                str(processed_data_path), 
                ticker=ticker, 
                logger_obj=logger
            )

            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "date": prediction.get("date", "Unknown"),
                "prediction": prediction.get("prediction"),
                "direction": prediction.get("direction"),
                "probability_up": prediction.get("probability_up"),
                "confidence": prediction.get("confidence"),
            }

            logger.info(
                f"Prediction for {ticker}: {prediction.get('direction')} ({prediction.get('confidence')} Confidence)"
            )

            if should_save and prediction_csv_file:
                save_prediction_log(log_entry, prediction_csv_file)
                logger.info(f"Prediction appended to log: {prediction_csv_file}")

            success_count += 1

        except Exception as e:
            logger.error(f"✗ Error processing {ticker}: {e}")
            continue

    logger.info(f"{'=' * 40}")
    logger.info(f"Prediction Job Complete. Successful: {success_count}/{len(tickers)}")

    return


if __name__ == "__main__":
    sys.exit(main())
    