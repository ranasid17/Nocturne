#!/usr/bin/env python3
"""Cluster one ticker's processed sessions and save run-owned outputs."""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.services import run_clustering_workflow
from qusa.utils.config import load_config
from qusa.utils.errors import safe_error
from qusa.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-ticker", "--ticker", required=True,
        help="Ticker symbol to cluster, for example -ticker AMZN",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logger(
        "ClusteringPipeline", log_file=str(PROJECT_ROOT / "logs" / "clustering.log")
    )
    try:
        config = load_config()
    except Exception as exc:
        logger.error("Failed to load config: %s", safe_error(exc))
        return 1

    try:
        result = run_clustering_workflow(args.ticker.upper(), config)
    except Exception as exc:
        logger.error("Clustering failed for %s: %s", args.ticker.upper(), safe_error(exc))
        return 1
    logger.info("Clustered data: %s", result["output_path"])
    logger.info("Regime statistics: %s", result["statistics_path"])
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
