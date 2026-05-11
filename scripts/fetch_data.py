#!/usr/bin/env python3
# qusa/scripts/fetch_data.py

"""
Standalone script to fetch OHLCV data from Polygon.io.
Supports fetching a specific date range or the last N trading days.
Automatically consolidates and deconflicts historical data.
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from qusa.data.loader import DataLoader
from qusa.utils.config import load_config
from qusa.utils.logger import setup_logger
from qusa.utils.formatting import format_header


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fetch historical stock data from Polygon.io."
    )
    # Standardized ticker flag with -ticker alias
    parser.add_argument(
        "-ticker", "--ticker", 
        required=True, 
        help="Ticker symbol (e.g., AMZN)"
    )
    parser.add_argument("--days", type=int, help="Number of most recent trading days to fetch")
    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD), defaults to today")
    return parser.parse_args()


def main():
    args = parse_args()
    ticker = args.ticker.upper()
    
    logger = setup_logger("DataFetcher", log_file=str(PROJECT_ROOT / "logs" / "fetch_data.log"))
    
    for line in format_header(f"Fetching Data for {ticker}").split("\n"):
        logger.info(line)
    
    try:
        config = load_config(PROJECT_ROOT / "qusa" / "utils" / "config.yaml")
        raw_data_dir = Path(config["data"]["paths"]["raw_data_dir"]).expanduser()
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        return 1

    loader = DataLoader(raw_data_dir=str(raw_data_dir))
    
    temp_path = None
    try:
        if args.days:
            logger.info(f"Fetching last {args.days} trading days for {ticker}...")
            
            df_new = loader.fetcher.fetch_n_days(ticker, args.days)
            if df_new.empty:
                logger.warning("No data found.")
                return 1
            
            # Save temporary file for consolidation
            start_date = df_new["date"].iloc[0]
            end_date = df_new["date"].iloc[-1]
            temp_path = raw_data_dir / f"{ticker}_{start_date}_{end_date}.csv"
            raw_data_dir.mkdir(parents=True, exist_ok=True)
            df_new.to_csv(temp_path, index=False)
            
            # Consolidate
            _, skipped = loader.consolidate_history(ticker)
            if skipped:
                logger.warning(f"Skipped {len(skipped)} files due to errors: {skipped}")
            
        elif args.start:
            start_date = args.start
            end_date = args.end or datetime.now().strftime("%Y-%m-%d")
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}...")
            loader.load_range(ticker, start_date, end_date)
            
        else:
            logger.error("Either --days or --start must be provided.")
            return 1

        logger.info(f"✓ Data for {ticker} is consolidated and ready in {raw_data_dir}")
        return 0

    except Exception as e:
        logger.error(f"✗ Failed to fetch data: {e}")
        return 1
    
    finally:
        # Clean up temp file if it exists (consolidate_history archives it on success, 
        # but if it fails before consolidation, we should remove it)
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
                logger.debug(f"Cleaned up orphan temp file {temp_path}")
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
