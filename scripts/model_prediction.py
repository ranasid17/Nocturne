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

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from qusa.model import make_prediction
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make QUSA predictions for one or more tickers."
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        help="Ticker symbol(s) to predict, for example AMZN or AMZN AAPL",
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

    # convert dict to DataFrame
    prediction = pd.DataFrame([prediction_data])

    try:
        # create log directory if does not exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # append prediction to log file (only write header when file not found)
        prediction.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    except Exception as e:
        raise IOError(f"Failed to save prediction to {log_file_path}: {e}")


def main():
    """
    Main function to make prediction.
    """

    args = parse_args()
    tickers = [ticker.upper() for ticker in args.tickers]

    # 1) set up configuration and logging
    try:
        config_path = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"
        config = load_config(str(config_path))

        log_file = config.get("prediction", {}).get("log_file", "logs/predictions.log")
        logger = setup_logger("predictor", log_file=log_file)

        logger.info("Configuration loaded successfully.")

    except IOError as e:
        print(f"✗ Configuration file not found: {e}")
        sys.exit(1)

    # 2) extract settings
    try:
        # store key paths
        model_dir = Path(config["model"]["output"]["model_output_path"]).expanduser()
        data_dir = Path(config["data"]["paths"]["processed_data_dir"]).expanduser()

        # store prediction settings
        should_save = config.get("prediction", {}).get("save", True)
        prediction_log_file = config.get("prediction", {}).get("log_file")

    except KeyError as e:
        logger.error(f"✗ Missing configuration key: {e}")
        sys.exit(1)

    # 3) main prediction loop
    success_count = 0

    for ticker in tickers:
        logger.info(f"{'=' * 40}")
        logger.info(f"Processing Ticker: {ticker}")

        try:
            # build dynamic paths from pre-defined structure
            model_path = model_dir / f"{ticker.lower()}_model.pkl"
            data_path = data_dir / f"{ticker}_processed.csv"

            # validate paths before calling prediction
            if not model_path.exists():
                logger.warning(f"Skipping {ticker}: Model not found at {model_path}")
                continue  # skip to next ticker
            if not data_path.exists():
                logger.warning(f"Skipping {ticker}: Data not found at {data_path}")
                continue  # skip to next ticker

            # Make Prediction
            ## Note: make_prediction prints to stdout internally,
            ##       but capture the return dict for logging.
            logger.info(
                f"Predicting using model at {model_path} and data at {data_path}"
            )
            prediction = make_prediction(str(model_path), str(data_path))
            logger.info(f"Prediction for {ticker}: {prediction}")

            # Enrich result with metadata for the CSV log
            log_entry = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "date": prediction.get("date", "Unknown"),
                "prediction": prediction.get("prediction"),
                "direction": prediction.get("direction"),
                "probability_up": prediction.get("probability_up"),
                "confidence": prediction.get("confidence"),
            }

            # log result to console
            logger.info(
                f"Prediction for {ticker}: {prediction.get('direction')} ({prediction.get('confidence')} Confidence)"
            )

            # save prediction log if enabled
            if should_save and prediction_log_file:
                save_prediction_log(log_entry, prediction_log_file)
                logger.info(f"Prediction appended to log: {prediction_log_file}")

            success_count += 1

        except Exception as e:
            logger.error(f"✗ Error processing {ticker}: {e}")
            continue

    # 4) summary
    logger.info(f"{'=' * 40}")
    logger.info(f"Prediction Job Complete. Successful: {success_count}/{len(tickers)}")

    return


if __name__ == "__main__":
    sys.exit(main())
