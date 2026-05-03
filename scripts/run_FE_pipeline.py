import argparse
import os
import pandas as pd
import sys
from pathlib import Path

# add parent directory to sys.path for module imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.features.pipeline import FeaturePipeline
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run QUSA feature engineering for one ticker."
    )
    parser.add_argument("ticker", help="Ticker symbol to process, for example AMZN")
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

    # store required columns not found in input DataFrame
    missing_columns = set(required_columns) - set(df.columns)

    # raise error when missing columns exist
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

    try:  # load config file
        logger.info("Loading configuration file...")

        config = load_config(PROJECT_ROOT / "qusa" / "utils" / "config.yaml")

        logger.info("✓ Configuration loaded successfully")
        logger.info(f"  Data directory: {config['data']['paths']['raw_data_dir']}")
        logger.info(
            f"  Output directory: {config['data']['paths']['processed_data_dir']}"
        )

    except Exception as e:  # unable to load config
        logger.error(f"✗ Error loading configuration: {e}")
        return 1

    try:  # load data
        logger.info("Loading data...")
        data_path = os.path.expanduser(config["data"]["paths"]["raw_data_dir"])
        start_date = config["data"]["start_date"]
        end_date = config["data"]["end_date"]
        data_path = os.path.join(data_path, f"{ticker}_{start_date}_{end_date}.csv")

        # handle case where path to data does not exist
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        data = pd.read_csv(data_path)
        logger.info("✓ Data loaded successfully")

    except Exception as e:  # unable to load data
        logger.error(f"✗ Error loading data: {e}")
        return 1

    try:  # validate dataframe
        logger.info("Validating input DataFrame...")
        required_columns = [
            config.get("fe_params", {}).get("date_col", "date"),
            config.get("fe_params", {}).get("open_col", "open"),
            config.get("fe_params", {}).get("close_col", "close"),
            config.get("fe_params", {}).get("high_col", "high"),
            config.get("fe_params", {}).get("low_col", "low"),
            config.get("fe_params", {}).get("volume_col", "volume"),
        ]
        validate_dataframe(data, required_columns)
        logger.info("✓ Input DataFrame validation successful")

    except Exception as e:  # DataFrame missing required columns
        logger.error(f"✗ DataFrame validation error: {e}")
        return 1

    try:  # run FE pipeline
        logger.info("Running Feature Engineering Pipeline...")

        # initialize pipeline object and run on DataFrame
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

    except Exception as e:  # FE pipeline error
        logger.error(f"✗ Error during Feature Engineering: {e}")
        logger.exception("Full traceback:")
        return 1

    try:  # save processed data
        logger.info("Saving processed data...")
        # extract output path from config
        processed_dir = os.path.expanduser(
            config["data"]["paths"]["processed_data_dir"]
        )
        os.makedirs(processed_dir, exist_ok=True)

        output_path = os.path.join(processed_dir, f"{ticker}_processed.csv")

        # create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        processed_data.to_csv(output_path, index=False)
        logger.info(f"✓ Processed data saved to {output_path}")

    except Exception as e:  # unable to save data
        logger.error(f"✗ Error saving processed data: {e}")
        return 1

    # Summary
    logger.info("=" * 80)
    logger.info("Pipeline Execution Summary")
    logger.info("=" * 80)
    logger.info(f"Input file: {data_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Rows processed: {len(processed_data)}")
    logger.info(f"Features created: {len(fe_pipeline.get_engineered_features(include_monte_carlo=config.get('monte_carlo', {}).get('enabled', False), mc_horizons=config.get('monte_carlo', {}).get('horizons', None)))}")
    logger.info("=" * 80)
    logger.info("✓ Pipeline completed successfully!")
    logger.info("=" * 80)

    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
