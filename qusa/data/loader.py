"""Validated, revision-aware local OHLCV history management."""

import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from qusa.data.fetcher import PolygonFetcher
from qusa.storage.artifacts import atomic_write_csv
from qusa.utils.errors import safe_error


REQUIRED_OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")


def validate_ohlcv(data):
    """Return a normalized OHLCV frame or raise before it reaches disk."""

    missing = [column for column in REQUIRED_OHLCV_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError("OHLCV data is missing columns: " + ", ".join(missing))
    result = data.loc[:, REQUIRED_OHLCV_COLUMNS].copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result.empty or result["date"].isna().any():
        raise ValueError("OHLCV data must contain valid dates.")
    for column in REQUIRED_OHLCV_COLUMNS[1:]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    prices = result[["open", "high", "low", "close"]]
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError("OHLC prices must be finite and positive.")
    if not np.isfinite(result["volume"].to_numpy()).all() or (result["volume"] < 0).any():
        raise ValueError("OHLCV volume must be finite and non-negative.")
    if (result["high"] < result[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("OHLCV high must not be below open, close, or low.")
    if (result["low"] > result[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("OHLCV low must not be above open, close, or high.")
    return result


class DataLoader:
    """Load, revise, and publish canonical ticker history on local disk."""

    def __init__(self, raw_data_dir, api_key=None, logger=None, fetcher=None):
        self.raw_dir = Path(raw_data_dir).expanduser()
        self.archive_dir = self.raw_dir / "archive"
        self._api_key = api_key
        self._fetcher = fetcher
        self.logger = logger or logging.getLogger(__name__)

    @property
    def fetcher(self):
        """Create the provider only when a network operation actually needs it."""

        if self._fetcher is None:
            self._fetcher = PolygonFetcher(api_key=self._api_key)
        return self._fetcher

    def _source_paths(self, ticker):
        history_path = self.raw_dir / f"{ticker}_history.csv"
        source_paths = sorted(
            path
            for path in self.raw_dir.glob(f"{ticker}_*.csv")
            if path != history_path
            and not path.name.endswith(("_processed.csv", "_clustered.csv"))
        )
        return history_path, source_paths

    def consolidate_history(self, ticker):
        """Publish a validated canonical history; newer fragments supersede history."""

        ticker = ticker.upper()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        history_path, source_paths = self._source_paths(ticker)
        frames = []
        skipped_files = []

        if history_path.exists():
            try:
                history = validate_ohlcv(pd.read_csv(history_path))
                history["_priority"] = 0
                history["_source"] = history_path.name
                frames.append(history)
            except Exception as exc:
                self.logger.error("Could not read canonical history %s: %s", history_path, safe_error(exc))
                skipped_files.append(str(history_path))

        valid_source_paths = []
        for source_path in source_paths:
            try:
                source = validate_ohlcv(pd.read_csv(source_path))
                source["_priority"] = 1
                source["_source"] = source_path.name
                frames.append(source)
                valid_source_paths.append(source_path)
            except Exception as exc:
                self.logger.warning("Could not read source file %s: %s", source_path, safe_error(exc))
                skipped_files.append(str(source_path))

        if not frames:
            return pd.DataFrame(columns=REQUIRED_OHLCV_COLUMNS), skipped_files

        merged = pd.concat(frames, ignore_index=True)
        # Higher priority fragments supersede history; lexical source order breaks ties.
        merged = merged.sort_values(["date", "_priority", "_source"])
        consolidated = merged.drop_duplicates(subset=["date"], keep="last")
        consolidated = consolidated.loc[:, REQUIRED_OHLCV_COLUMNS].sort_values("date")
        consolidated = consolidated.reset_index(drop=True)
        consolidated["date"] = consolidated["date"].dt.strftime("%Y-%m-%d")

        atomic_write_csv(consolidated, history_path)
        self.logger.info("Consolidated history saved to %s (%s rows)", history_path, len(consolidated))

        if valid_source_paths:
            self.archive_dir.mkdir(parents=True, exist_ok=True)
            for source_path in valid_source_paths:
                destination = self.archive_dir / source_path.name
                try:
                    shutil.move(source_path, destination)
                except Exception as exc:
                    self.logger.warning("Could not archive source file %s: %s", source_path, safe_error(exc))

        return consolidated, skipped_files

    def load_most_recent(self, ticker, start=None, end=None):
        """Fetch a completed provider bar and atomically merge it into history."""

        ticker = ticker.upper()
        latest = validate_ohlcv(self.fetcher.fetch_latest_day(ticker))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        latest_path = self.raw_dir / f"{ticker}_latest.csv"
        latest["date"] = latest["date"].dt.strftime("%Y-%m-%d")
        atomic_write_csv(latest, latest_path)
        history, _ = self.consolidate_history(ticker)
        if history.empty:
            raise ValueError(f"No valid history exists for {ticker} after the provider fetch.")

        filtered = history.copy()
        filtered["date"] = pd.to_datetime(filtered["date"])
        if start:
            filtered = filtered.loc[filtered["date"] >= pd.Timestamp(start)]
        if end:
            filtered = filtered.loc[filtered["date"] <= pd.Timestamp(end)]
        filtered["date"] = filtered["date"].dt.strftime("%Y-%m-%d")
        return filtered.reset_index(drop=True)

    def load_range(self, ticker, start, end):
        """Fetch a range and merge it into canonical history."""

        ticker = ticker.upper()
        fetched = validate_ohlcv(self.fetcher.fetch_historical_range(ticker, start, end))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        temporary_path = self.raw_dir / f"{ticker}_{start}_{end}.csv"
        fetched["date"] = fetched["date"].dt.strftime("%Y-%m-%d")
        atomic_write_csv(fetched, temporary_path)
        history, _ = self.consolidate_history(ticker)
        return history
