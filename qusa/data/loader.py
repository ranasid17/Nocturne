# qusa/data/loader.py

"""
DataLoader: thin wrapper around PolygonFetcher that handles the
Option-3 storage strategy:

  - Historical data lives in:   data/raw/{TICKER}_{start}_{end}.csv
  - Latest day lives in:        data/raw/{TICKER}_latest.csv

load_most_recent() fetches the latest day from Polygon, saves it to
{TICKER}_latest.csv, then returns a merged DataFrame (history + latest)
ready for FeaturePipeline.run(). The historical CSV is never modified.
"""

import os
import pandas as pd

from pathlib import Path

from qusa.data.fetcher import PolygonFetcher


class DataLoader:
    """
    Load and merge OHLCV data for a ticker.
    """

    def __init__(self, raw_data_dir, api_key=None):
        """
        Class constructor.

        Parameters:
            1) raw_data_dir (str): Path to data/raw/ directory.
            2) api_key (str, optional): Polygon API key; falls back to env var.
        """

        self.raw_dir = Path(raw_data_dir).expanduser()
        self.fetcher = PolygonFetcher(api_key=api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_most_recent(self, ticker, start, end):
        """
        Fetch the most recent trading day from Polygon, persist it to
        {TICKER}_latest.csv, then return a merged DataFrame of the full
        historical CSV plus the latest row — deduplicated and sorted.

        Parameters:
            1) ticker (str): Stock ticker symbol.
            2) start (str): Historical range start (YYYY-MM-DD), used to
               locate the existing raw CSV by filename convention.
            3) end (str): Historical range end (YYYY-MM-DD).

        Returns:
            1) pd.DataFrame: Merged OHLCV DataFrame, sorted ascending by date.
        """

        ticker = ticker.upper()

        # 1. Fetch latest day and save to {TICKER}_latest.csv
        latest_df = self.fetcher.fetch_latest_day(ticker)
        latest_path = self.raw_dir / f"{ticker}_latest.csv"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        latest_df.to_csv(latest_path, index=False)
        print(f"✓ Latest day saved → {latest_path}")

        # 2. Load historical CSV if it exists
        history_path = self.raw_dir / f"{ticker}_{start}_{end}.csv"
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            history_df["date"] = pd.to_datetime(history_df["date"])
            print(f"✓ Historical data loaded from {history_path}")
        else:
            print(
                f"⚠ No historical CSV found at {history_path}. "
                "Returning latest day only."
            )
            history_df = pd.DataFrame()

        # 3. Merge, deduplicate on date, sort ascending
        latest_df["date"] = pd.to_datetime(latest_df["date"])
        merged = (
            pd.concat([history_df, latest_df], ignore_index=True)
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        print(f"✓ Merged DataFrame: {len(merged)} rows")

        return merged

    def load_range(self, ticker, start, end):
        """
        Fetch a full historical range from Polygon and save to the
        conventional raw CSV path. Overwrites any existing file for
        that ticker/range.

        Parameters:
            1) ticker (str): Stock ticker symbol.
            2) start (str): Start date YYYY-MM-DD (inclusive).
            3) end (str): End date YYYY-MM-DD (inclusive).

        Returns:
            1) pd.DataFrame: OHLCV DataFrame sorted ascending by date.
        """

        ticker = ticker.upper()

        df = self.fetcher.fetch_historical_range(ticker, start, end)

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.raw_dir / f"{ticker}_{start}_{end}.csv"
        df.to_csv(out_path, index=False)
        print(f"✓ Historical range saved → {out_path} ({len(df)} rows)")

        return df
    