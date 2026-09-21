import os
from pathlib import Path

from qusa.data.loader import DataLoader
from qusa.features.pipeline import FeaturePipeline
from qusa.utils.config import load_config
from qusa.utils.formatting import format_box, format_header


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "qusa" / "utils" / "config.yaml"


def _build_feature_pipeline(config):
    return FeaturePipeline(
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


def run_feature_pipeline(ticker, fetch_latest=False, config_path=None, logger=None):
    """
    Run feature engineering for one ticker and return structured metadata.
    """

    ticker = ticker.upper()
    config_path = Path(config_path or DEFAULT_CONFIG_PATH)

    if logger:
        for line in format_header("Starting Nocturne Feature Pipeline").split("\n"):
            logger.info(line)
        logger.info("Loading configuration file...")

    config = load_config(config_path)

    if logger:
        logger.info("✓ Configuration loaded successfully")

    raw_data_dir = os.path.expanduser(config["data"]["paths"]["raw_data_dir"])
    loader = DataLoader(raw_data_dir=raw_data_dir)

    if fetch_latest:
        if logger:
            logger.info(f"--fetch flag set: pulling latest day for {ticker}...")
        data = loader.load_most_recent(ticker)
        skipped_files = []
        if logger:
            logger.info(f"✓ Latest data prepared: {len(data)} rows")
    else:
        if logger:
            logger.info(f"Locating historical data for {ticker}...")
        data, skipped_files = loader.consolidate_history(ticker)

        if skipped_files and logger:
            logger.warning(
                f"⚠ Skipped {len(skipped_files)} files during consolidation: {skipped_files}"
            )

        if data.empty:
            raise ValueError(f"No historical data found for {ticker} in {raw_data_dir}")

        if logger:
            logger.info(f"✓ Data loaded successfully: {len(data)} rows")

    if logger:
        logger.info("Running Feature Engineering Pipeline...")

    fe_pipeline = _build_feature_pipeline(config)
    processed_data = fe_pipeline.run(data, ticker=ticker)

    if logger:
        log_mc_feature_validation(fe_pipeline, processed_data, logger)
        logger.info("Feature Engineering Pipeline completed successfully.")
        logger.info(f"  Output shape: {processed_data.shape}")
        logger.info("Saving processed data...")

    processed_dir = os.path.expanduser(config["data"]["paths"]["processed_data_dir"])
    os.makedirs(processed_dir, exist_ok=True)

    output_path = Path(processed_dir) / f"{ticker}_processed.csv"
    processed_data.to_csv(output_path, index=False)

    if logger:
        logger.info(f"✓ Processed data saved to {output_path}")
        summary_box = format_box(
            [
                f"Ticker:    {ticker}",
                f"Source:    {raw_data_dir}/{ticker}_history.csv",
                f"Output:    {output_path}",
                f"Rows:      {len(processed_data)}",
                f"Shape:     {processed_data.shape}",
            ],
            title="Pipeline Execution Summary",
        )
        for line in summary_box.split("\n"):
            logger.info(line)

        for line in format_header("✓ Pipeline completed successfully!").split("\n"):
            logger.info(line)

    return {
        "success": True,
        "ticker": ticker,
        "rows": len(processed_data),
        "shape": tuple(processed_data.shape),
        "output_path": str(output_path),
        "source_path": str(Path(raw_data_dir) / f"{ticker}_history.csv"),
        "skipped_files": skipped_files,
    }
