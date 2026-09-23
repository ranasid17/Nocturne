# qusa/scripts/run_FE_pipeline.py

import argparse
import sys
from pathlib import Path

# add parent directory to sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.services import run_feature_pipeline
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Nocturne feature engineering for one ticker."
    )
    # Standardized ticker flag with -ticker alias
    parser.add_argument(
        "-ticker", "--ticker", 
        required=True, 
        help="Ticker symbol to process, for example AMZN"
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=(
            "Fetch the most recent trading day and merge it "
            "into history before running feature engineering. "
            "Requires POLYGON_API_KEY env var."
        ),
    )
    return parser.parse_args()


def main():
    """
    Main function to run the feature engineering pipeline.
    """

    args = parse_args()
    ticker = args.ticker.upper()

    logger = setup_logger(
        "FE_pipeline",
        log_file=str(PROJECT_ROOT / "logs" / "fe_pipeline.log"),
    )
    config_path = None

    try:
        run_feature_pipeline(
            ticker=ticker,
            fetch_latest=args.fetch,
            config_path=config_path,
            logger=logger,
        )
        return 0
    except Exception as e:
        logger.error(f"✗ Feature pipeline failed for {ticker}: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
