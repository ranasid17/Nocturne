# qusa/data/loader.py

"""
DataLoader: handles retrieving, merging, and consolidating OHLCV data.
Implements the "Unified History" strategy to prevent data fragmentation.
"""

import os
import shutil
import pandas as pd
import glob
from pathlib import Path

from qusa.data.fetcher import PolygonFetcher


class DataLoader:
    """
    Load, merge, and consolidate OHLCV data for tickers.
    """

    def __init__(self, raw_data_dir, api_key=None):
        """
        Class constructor.

        Parameters:
            1) raw_data_dir (str): Path to data/raw/ directory.
            2) api_key (str, optional): Polygon API key; falls back to env var.
        """
        self.raw_dir = Path(raw_data_dir).expanduser()
        self.archive_dir = self.raw_dir / "archive"
        self.fetcher = PolygonFetcher(api_key=api_key)

    def consolidate_history(self, ticker):
        """
        Scans for all raw data files for a ticker, merges them, 
        deduplicates by date, and saves to {TICKER}_history.csv.
        Moves fragmented source files to an archive directory.

        Parameters:
            1) ticker (str): Stock ticker symbol.

        Returns:
            1) pd.DataFrame: Consolidated historical data.
        """
        ticker = ticker.upper()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Find all relevant CSV files for this ticker
        # Pattern matches {TICKER}_*.csv but we exclude processed/clustered/latest/history
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
        if history_path.exists():
            dfs.append(pd.read_csv(history_path))
            
        # 3. Load all other source files
        for f in source_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception as e:
                print(f"⚠ Warning: Could not read {f}: {e}")

        if not dfs:
            return pd.DataFrame()

        # 4. Consolidate
        merged = pd.concat(dfs, ignore_index=True)
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
        print(f"✓ Consolidated history saved → {history_path} ({len(consolidated)} rows)")

        # 6. Archive source files
        if source_files:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            for f in source_files:
                try:
                    dest = self.archive_dir / os.path.basename(f)
                    # Use shutil.move to handle cross-device moves if necessary
                    shutil.move(f, str(dest))
                except Exception as e:
                    print(f"⚠ Warning: Could not archive {f}: {e}")
            print(f"✓ Archived {len(source_files)} fragmented source files to {self.archive_dir}")

        return consolidated

    def load_most_recent(self, ticker, start=None, end=None):
        """
        Fetches the latest trading day, merges it into the consolidated 
        history, and returns the full dataset.

        Parameters:
            1) ticker (str): Stock ticker symbol.
            2) start/end (str, optional): Legacy params, no longer strictly needed 
               with consolidation strategy but kept for compatibility.

        Returns:
            1) pd.DataFrame: Updated historical data.
        """
        ticker = ticker.upper()

        # 1. Fetch latest day
        latest_df = self.fetcher.fetch_latest_day(ticker)
        latest_path = self.raw_dir / f"{ticker}_latest.csv"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        latest_df.to_csv(latest_path, index=False)
        print(f"✓ Latest day fetched → {latest_df['date'].iloc[0]}")

        # 2. Use consolidate_history to merge the latest day (via the _latest.csv) 
        # Actually consolidate_history excludes _latest.csv to avoid redundant archiving.
        # Let's just merge it manually here and then save.
        
        history = self.consolidate_history(ticker)
        
        # If no history yet, just use latest
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
        final["date"] = final["date"].dt.strftime("%Y-%m-%d")
        
        # Update history file
        history_path = self.raw_dir / f"{ticker}_history.csv"
        final.to_csv(history_path, index=False)
        
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
        
        return self.consolidate_history(ticker)
