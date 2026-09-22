"""
PolygonFetcher: central utility for retrieving OHLCV data from Polygon.io.
Supports fetching latest daily bars, historical ranges, and N-day queries.
"""

import logging
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

from qusa.data.sessions import NyseSessionCalendar

logger = logging.getLogger(__name__)


class LatestBarUnavailableError(ValueError):
    """Raised when a completed session does not yet have provider data."""


class PolygonFetcher:
    """
    Client for the Polygon.io Stocks API.
    """

    def __init__(self, api_key=None, session_calendar=None, clock=None):
        """
        Initialize the fetcher.
        
        Parameters:
            1) api_key (str, optional): Polygon API key. Falls back to POLYGON_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable or api_key parameter is required.")
        
        self.base_url = "https://api.polygon.io"
        self.sessions = session_calendar or NyseSessionCalendar(clock=clock)

    def _get_most_recent_trading_day(self):
        """
        Calculates the date of the most recent completed trading day.
        
        Returns:
            1) str: ISO format date (YYYY-MM-DD).
        """
        completed_session = self.sessions.most_recent_completed_session()
        if completed_session is None:
            raise LatestBarUnavailableError("No completed NYSE session is available yet.")
        return completed_session["date"]

    def fetch_latest_day(self, ticker):
        """
        Fetches the most recent completed daily bar for a ticker.
        
        Parameters:
            1) ticker (str): Ticker symbol.
            
        Returns:
            1) pd.DataFrame: Single-row DataFrame with OHLCV data.
        """
        date = self._get_most_recent_trading_day()
        url = f"{self.base_url}/v1/open-close/{ticker.upper()}/{date}"
        params = {"adjusted": "true", "apiKey": self.api_key}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        status = data.get("status")
        if status not in ["OK", "DELAYED"]:
            raise LatestBarUnavailableError(
                f"Polygon API returned non-OK status for completed session {date}: {status}"
            )
            
        row = {
            "date": date,
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"]
        }
        
        frame = pd.DataFrame([row])
        frame.attrs["completed_session"] = date
        frame.attrs["provider_status"] = status
        return frame

    def fetch_historical_range(self, ticker, start, end):
        """
        Fetches a range of daily aggregates for a ticker.
        
        Parameters:
            1) ticker (str): Ticker symbol.
            2) start (str): Start date (YYYY-MM-DD).
            3) end (str): End date (YYYY-MM-DD).
            
        Returns:
            1) pd.DataFrame: DataFrame with OHLCV data.
        """
        ticker = ticker.upper()
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
        params = {"adjusted": "true", "sort": "asc", "apiKey": self.api_key}
        
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        status = data.get("status")
        if status not in ["OK", "DELAYED"]:
             # It might be 'OK' but empty if no results found
             if data.get("resultsCount", 0) == 0:
                 return pd.DataFrame()
             raise ValueError(f"Polygon API returned non-OK status: {status}")
        
        results = data.get("results", [])
        rows = []
        for r in results:
            # Polygon aggs 't' is Unix msec timestamp
            dt = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc).date()
            rows.append({
                "date": dt.isoformat(),
                "open": r["o"],
                "high": r["h"],
                "low": r["l"],
                "close": r["c"],
                "volume": r["v"]
            })
            
        return pd.DataFrame(rows)

    def fetch_n_days(self, ticker, n_days):
        """
        Fetches the most recent N trading days for a ticker.
        
        Parameters:
            1) ticker (str): Ticker symbol.
            2) n_days (int): Number of trading days to retrieve.
            
        Returns:
            1) pd.DataFrame: DataFrame with OHLCV data.
        """
        # To get N trading days, we fetch a wider range (approx n_days * 1.5) and then tail it.
        # This accounts for weekends and holidays.
        end_date = datetime.now(timezone.utc).date()
        start_date = end_date - timedelta(days=int(n_days * 1.6) + 5)
        
        df = self.fetch_historical_range(ticker, start_date.isoformat(), end_date.isoformat())
        
        if len(df) > n_days:
            return df.tail(n_days).reset_index(drop=True)
        return df
