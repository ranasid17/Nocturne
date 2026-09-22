#!/usr/bin/env python3
# qusa/scripts/model_prediction.py

"""
Make prediction on the most recent trading day.
"""

import argparse
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from qusa.services import make_latest_prediction
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Make Nocturne predictions for one or more tickers."
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
    parser.add_argument(
        "--volatility",
        type=float,
        help="Volatility filter threshold (max ATR%%). Overrides config.",
    )
    return parser.parse_args()


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
        config["model"]["output"]["model_output_path"]
        config["data"]["paths"]["processed_data_dir"]
    except KeyError as e:
        logger.error(f"✗ Missing configuration key: {e}")
        return 1

    success_count = 0

    for ticker in tickers:
        logger.info(f"{'=' * 40}")
        logger.info(f"Processing Ticker: {ticker}")

        try:
            make_latest_prediction(
                ticker=ticker,
                fetch_latest=args.fetch,
                volatility_override=args.volatility,
                config_path=config_path,
                logger=logger,
            )
            success_count += 1

        except Exception as e:
            logger.error(f"✗ Error processing {ticker}: {e}")
            continue

    logger.info(f"{'=' * 40}")
    logger.info(f"Prediction Job Complete. Successful: {success_count}/{len(tickers)}")

    return 0 if success_count == len(tickers) else 1


if __name__ == "__main__":
    sys.exit(main())
