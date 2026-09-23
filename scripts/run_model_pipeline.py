#!/usr/bin/env python3
"""Run model training, evaluation, and backtesting for one or more tickers."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.services import run_model_workflow
from qusa.utils.config import load_config
from qusa.utils.errors import safe_error
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-ticker", "--ticker", "--tickers", dest="tickers", nargs="+", required=True,
        help="Ticker symbol(s) to process, for example -ticker AMZN AAPL",
    )
    parser.add_argument(
        "--volatility", type=float,
        help="Maximum ATR percent for the backtest volatility filter",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        config = load_config()
        logger = setup_logger(
            "pipeline_orchestrator", log_file=str(PROJECT_ROOT / "logs" / "model_pipeline.log")
        )
    except Exception as exc:
        print(f"Error loading configuration: {safe_error(exc)}")
        return 1

    tickers = [ticker.upper() for ticker in args.tickers]
    successful = 0
    for ticker in tickers:
        logger.info("Processing %s", ticker)
        try:
            result = run_model_workflow(
                ticker, config, volatility_override=args.volatility, logger=logger
            )
        except Exception as exc:
            logger.error("Pipeline failed for %s: %s", ticker, safe_error(exc))
            continue

        for name, phase in result["phases"].items():
            logger.info("%s: %s", name, phase["status"])
            if phase["status"] == "failed":
                logger.warning("%s: %s", name, phase.get("error", "Phase failed."))
        for name, report in result.get("reports", {}).items():
            logger.info("%s report: %s", name, report["status"])
            if report["status"] == "failed":
                logger.warning("%s report: %s", name, report.get("error", "Report failed."))
        if result["success"]:
            successful += 1
        else:
            logger.warning("Pipeline completed with failed phases for %s (run %s)", ticker, result["run_id"])

    logger.info("Model pipeline complete: %s/%s succeeded", successful, len(tickers))
    return 0 if successful == len(tickers) else 1


if __name__ == "__main__":
    sys.exit(main())
