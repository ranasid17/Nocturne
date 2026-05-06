# qusa/scripts/run_FE_pipeline.py

import argparse
import os
import pandas as pd
import sys
from pathlib import Path

# add parent directory to sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.data.loader import DataLoader
from qusa.features.pipeline import FeaturePipeline
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QUSA feature engineering for one ticker."
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


def log_mc_feature_validation(fe_pipeline, processed_data, logger):
    """
    Log Monte Carlo feature validation and summary statistics when enabled.
    """

    mc_calculator = fe_pipeline.monte_carlo
    if mc_calculator is None:
        return

    validation = mc_calculator.validate_features(processed_data)
    logger.info("Monte Carlo feature validation:")
    logger.info(f"  Total rows: {validation['total_rows']:,}")
    logger.info(f"  Valid MC rows: {validation['valid_rows']:,}")
    logger.info(f"  NaN rows (threshold): {validation['nan_rows']:,}")

    if validation["errors"]:
        for error in validation["errors"]:
            logger.warning(f"  MC validation warning: {error}")
    else:
        logger.info("  No MC validation errors")

    mc_calculator.print_feature_summary(processed_data)


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
    logger.info("=" * 80)
    logger.info("Starting Feature Engineering Pipeline")
    logger.info("=" * 80)

    try:
        logger.info("Loading configuration file...")
        config = load_config(PROJECT_ROOT / "qusa" / "utils" / "config.yaml")
        logger.info("✓ Configuration loaded successfully")
    except Exception as e:
        logger.error(f"✗ Error loading configuration: {e}")
        return 1

    raw_data_dir = os.path.expanduser(config["data"]["paths"]["raw_data_dir"])
    loader = DataLoader(raw_data_dir=raw_data_dir)

    # ------------------------------------------------------------------
    # Data loading: Unified History Strategy
    # ------------------------------------------------------------------
    try:
        if args.fetch:
            logger.info(f"--fetch flag set: pulling latest day for {ticker}...")
            data = loader.load_most_recent(ticker)
            logger.info(f"✓ Latest data prepared: {len(data)} rows")
        else:
            logger.info(f"Locating historical data for {ticker}...")
            # Automatically consolidates any fragmented files found
            data = loader.consolidate_history(ticker)
            
            if data.empty:
                logger.error(f"✗ No historical data found for {ticker} in {raw_data_dir}")
                return 1
                
            logger.info(f"✓ Data loaded successfully: {len(data)} rows")
    except Exception as e:
        logger.error(f"✗ Error loading data: {e}")
        return 1

    try:
        logger.info("Running Feature Engineering Pipeline...")

        fe_pipeline = FeaturePipeline(
            {
                "date_col": "date",
                "open_col": "open",
                "close_col": "close",
                "high_col": "high",
                "low_col": "low",
                "volume_col": "volume",
                "overnight": {
                    "abnormal_threshold": config["analysis"]["abnormal_threshold"]
                },
                "technical_params": config["features"],
                "monte_carlo": config.get("monte_carlo", {}),
            }
        )

        processed_data = fe_pipeline.run(data, ticker=ticker)
        log_mc_feature_validation(fe_pipeline, processed_data, logger)

        logger.info("Feature Engineering Pipeline completed successfully.")
        logger.info(f"  Output shape: {processed_data.shape}")

    except Exception as e:
        logger.error(f"✗ Error during Feature Engineering: {e}")
        logger.exception("Full traceback:")
        return 1

    try:
        logger.info("Saving processed data...")
        processed_dir = os.path.expanduser(
            config["data"]["paths"]["processed_data_dir"]
        )
        os.makedirs(processed_dir, exist_ok=True)

        output_path = os.path.join(processed_dir, f"{ticker}_processed.csv")
        processed_data.to_csv(output_path, index=False)
        logger.info(f"✓ Processed data saved to {output_path}")

    except Exception as e:
        logger.error(f"✗ Error saving processed data: {e}")
        return 1

    logger.info("=" * 80)
    logger.info("Pipeline Execution Summary")
    logger.info("=" * 80)
    logger.info(f"  Ticker: {ticker}")
    logger.info(f"  Source: {raw_data_dir}/{ticker}_history.csv")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Rows processed: {len(processed_data)}")
    logger.info("=" * 80)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
