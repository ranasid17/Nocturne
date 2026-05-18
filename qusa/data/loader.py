# qusa/data/loader.py

"""
DataLoader: handles retrieving, merging, and consolidating OHLCV data.
Implements the "Unified History" strategy to prevent data fragmentation.
"""

import glob
import logging
import os
import shutil
import pandas as pd
from pathlib import Path

from qusa.data.fetcher import PolygonFetcher


class DataLoader:
    """
    Load, merge, and consolidate OHLCV data for tickers.
    """

    def __init__(self, raw_data_dir, api_key=None, logger=None):
        """
        Class constructor.

        Parameters:
            1) raw_data_dir (str): Path to data/raw/ directory.
            2) api_key (str, optional): Polygon API key; falls back to env var.
            3) logger (logging.Logger, optional): Logger instance.
        """
        self.raw_dir = Path(raw_data_dir).expanduser()
        self.archive_dir = self.raw_dir / "archive"
        self.fetcher = PolygonFetcher(api_key=api_key)
        self.logger = logger or logging.getLogger(__name__)

    def consolidate_history(self, ticker):
        """
        Scans for all raw data files for a ticker, merges them, 
        deduplicates by date, and saves to {TICKER}_history.csv.
        Moves fragmented source files to an archive directory.

        Parameters:
            1) ticker (str): Stock ticker symbol.

        Returns:
            1) tuple: (pd.DataFrame, list) Consolidated historical data and list of skipped files.
        """
        ticker = ticker.upper()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Find all relevant CSV files for this ticker
        pattern = str(self.raw_dir / f"{ticker}_*.csv")
        all_files = glob.glob(pattern)
        
        exclude_suffixes = ["_processed.csv", "_clustered.csv", "_latest.csv", "_history.csv"]
        source_files = [
            f for f in all_files 
            if not any(f.endswith(suffix) for suffix in exclude_suffixes)
        ]
        
        # 2. Load and merge existing history if it exists
        history_path = self.raw_dir / f"{ticker}_history.csv"
        dfs = []
        skipped_files = []

        if history_path.exists():
            try:
                dfs.append(pd.read_csv(history_path))
            except Exception as e:
                self.logger.error(f"✗ Could not read history file {history_path}: {e}")
                skipped_files.append(str(history_path))
            
        # 3. Load all other source files
        for f in source_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                self.logger.warning(f"⚠ Could not read source file {f}: {e}")
                skipped_files.append(f)

        if not dfs:
            return pd.DataFrame(), skipped_files

        # 4. Consolidate
        merged = pd.concat(dfs, ignore_index=True)
        
        if "date" not in merged.columns:
            self.logger.error("✗ Consolidated data missing 'date' column.")
            return pd.DataFrame(), skipped_files

        merged["date"] = pd.to_datetime(merged["date"])
        consolidated = (
            merged.drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        
        # Ensure date is back to string for CSV
        consolidated["date"] = consolidated["date"].dt.strftime("%Y-%m-%d")

        # 5. Save consolidated file
        consolidated.to_csv(history_path, index=False)
        self.logger.info(f"✓ Consolidated history saved → {history_path} ({len(consolidated)} rows)")

        # 6. Archive source files
        if source_files:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            for f in source_files:
                try:
                    dest = self.archive_dir / os.path.basename(f)
                    shutil.move(f, str(dest))
                except Exception as e:
                    self.logger.warning(f"⚠ Could not archive {f}: {e}")
            self.logger.info(f"✓ Archived {len(source_files)} fragmented source files to {self.archive_dir}")

        return consolidated, skipped_files

    def load_most_recent(self, ticker, start=None, end=None):
        """
        Fetches the latest trading day, merges it into the consolidated 
        history, and returns the full dataset.

        Parameters:
            1) ticker (str): Stock ticker symbol.
            2) start (str, optional): Start date filter (YYYY-MM-DD).
            3) end (str, optional): End date filter (YYYY-MM-DD).

        Returns:
            1) pd.DataFrame: Updated historical data.
        """
        ticker = ticker.upper()

        # 1. Fetch latest day
        latest_df = self.fetcher.fetch_latest_day(ticker)
        latest_path = self.raw_dir / f"{ticker}_latest.csv"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        latest_df.to_csv(latest_path, index=False)
        self.logger.info(f"✓ Latest day fetched → {latest_df['date'].iloc[0]}")

        # 2. Consolidate everything
        history, _ = self.consolidate_history(ticker)
        
        # 3. Manually merge the latest day which was just saved to _latest.csv 
        # (Note: consolidate_history excludes _latest.csv from its scan)
        if history.empty:
            merged = latest_df
        else:
            merged = pd.concat([history, latest_df], ignore_index=True)
            
        merged["date"] = pd.to_datetime(merged["date"])
        final = (
            merged.drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )
        
        # Apply date filters if provided (Task 3.2)
        if start:
            final = final[final["date"] >= pd.to_datetime(start)]
        if end:
            final = final[final["date"] <= pd.to_datetime(end)]

        final = final.copy()
        final["date"] = final["date"].dt.strftime("%Y-%m-%d")
        
        # Note: We return the filtered data, but history file contains full data.
        return final

    def load_range(self, ticker, start, end):
        """
        Fetches a range and consolidates it into the unified history.
        """
        ticker = ticker.upper()
        df = self.fetcher.fetch_historical_range(ticker, start, end)
        
        # Save temporary range file so consolidate_history can pick it up
        temp_path = self.raw_dir / f"{ticker}_{start}_{end}.csv"
        df.to_csv(temp_path, index=False)
        
        history, _ = self.consolidate_history(ticker)
        return history

    def load_intraday(self, ticker):
        """
        Fetches the current real-time snapshot and merges it with 
        consolidated history. Does not persist the snapshot to the 
        main history file (to avoid partial-day data corruption).

        Parameters:
            1) ticker (str): Stock ticker symbol.

        Returns:
            1) pd.DataFrame: Combined historical and intraday data.
        """
        ticker = ticker.upper()

        # 1. Consolidate history to ensure we have the most recent completed days
        history, _ = self.consolidate_history(ticker)
        
        if history.empty:
            self.logger.warning(f"No history found for {ticker}, starting fresh with intraday snapshot.")

        # 2. Fetch real-time snapshot
        try:
            snapshot_df = self.fetcher.fetch_intraday_snapshot(ticker)
            self.logger.info(f"✓ Intraday snapshot fetched for {ticker}")
        except Exception as e:
            self.logger.error(f"✗ Failed to fetch intraday snapshot: {e}")
            return history

        # 3. Merge snapshot into history
        if history.empty:
            merged = snapshot_df
        else:
            merged = pd.concat([history, snapshot_df], ignore_index=True)
            
        merged["date"] = pd.to_datetime(merged["date"])
        
        # Deduplicate by date, keeping the LAST entry (the snapshot) if there's an overlap
        final = (
            merged.drop_duplicates(subset=["date"], keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        
        final["date"] = final["date"].dt.strftime("%Y-%m-%d")
        return final
