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
    parser.add_argument("ticker", help="Ticker symbol to process, for example AMZN")
    parser.add_argument(
        "--fetch",
        action="store_true",
        help=(
            "Fetch the most recent trading day from Polygon.io and merge it "
            "onto the historical CSV before running feature engineering. "
            "Requires POLYGON_API_KEY env var."
        ),
    )
    return parser.parse_args()


def validate_dataframe(df, required_columns):
    """
    Validate that the DataFrame contains all required columns.

    Parameters:
        1) df (pd.DataFrame): DataFrame to validate.
        2) required_columns (list): List of required column names.

    Raises:
        1) ValueError: If any required column is missing.
    """

    missing_columns = set(required_columns) - set(df.columns)

    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {missing_columns}")


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
        logger.info(f"  Data directory: {config['data']['paths']['raw_data_dir']}")
        logger.info(
            f"  Output directory: {config['data']['paths']['processed_data_dir']}"
        )
    except Exception as e:
        logger.error(f"✗ Error loading configuration: {e}")
        return 1

    required_columns = [
        config.get("fe_params", {}).get("date_col", "date"),
        config.get("fe_params", {}).get("open_col", "open"),
        config.get("fe_params", {}).get("close_col", "close"),
        config.get("fe_params", {}).get("high_col", "high"),
        config.get("fe_params", {}).get("low_col", "low"),
        config.get("fe_params", {}).get("volume_col", "volume"),
    ]

    raw_data_dir = os.path.expanduser(config["data"]["paths"]["raw_data_dir"])
    start_date = config["data"]["start_date"]
    end_date = config["data"]["end_date"]

    # ------------------------------------------------------------------
    # Data loading: --fetch path vs. local CSV path
    # ------------------------------------------------------------------
    if args.fetch:
        try:
            logger.info(
                f"--fetch flag set: pulling latest day from Polygon.io for {ticker}..."
            )
            loader = DataLoader(raw_data_dir=raw_data_dir)
            data = loader.load_most_recent(ticker, start=start_date, end=end_date)
            logger.info(
                f"✓ Merged dataset ready: {len(data)} rows "
                f"(historical + latest day)"
            )
        except Exception as e:
            logger.error(f"✗ Error fetching data from Polygon: {e}")
            return 1
    else:
        try:
            logger.info("Loading data from local CSV...")
            data_path = os.path.join(
                raw_data_dir, f"{ticker}_{start_date}_{end_date}.csv"
            )

            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found at {data_path}")

            data = pd.read_csv(data_path)
            logger.info("✓ Data loaded successfully")
        except Exception as e:
            logger.error(f"✗ Error loading data: {e}")
            return 1

    try:
        logger.info("Validating input DataFrame...")
        validate_dataframe(data, required_columns)
        logger.info("✓ Input DataFrame validation successful")
    except Exception as e:
        logger.error(f"✗ DataFrame validation error: {e}")
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
        logger.info(f"  Features added: {processed_data.shape[1] - data.shape[1]}")

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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        processed_data.to_csv(output_path, index=False)
        logger.info(f"✓ Processed data saved to {output_path}")

    except Exception as e:
        logger.error(f"✗ Error saving processed data: {e}")
        return 1

    logger.info("=" * 80)
    logger.info("Pipeline Execution Summary")
    logger.info("=" * 80)
    logger.info(f"  Source: {'Polygon.io (--fetch)' if args.fetch else data_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Rows processed: {len(processed_data)}")
    logger.info(
        f"  Features created: {len(fe_pipeline.get_engineered_features(include_monte_carlo=config.get('monte_carlo', {}).get('enabled', False), mc_horizons=config.get('monte_carlo', {}).get('horizons', None)))}"
    )
    logger.info("=" * 80)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
    