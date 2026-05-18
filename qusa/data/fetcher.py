"""
PolygonFetcher: central utility for retrieving OHLCV data from Polygon.io.
Supports fetching latest daily bars, historical ranges, and N-day queries.
"""

import logging
import os
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


class PolygonFetcher:
    """
    Client for the Polygon.io Stocks API.
    """

    def __init__(self, api_key=None):
        """
        Initialize the fetcher.
        
        Parameters:
            1) api_key (str, optional): Polygon API key. Falls back to POLYGON_API_KEY env var.
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        if not self.api_key:
            raise ValueError("POLYGON_API_KEY environment variable or api_key parameter is required.")
        
        self.base_url = "https://api.polygon.io"

    def _get_most_recent_trading_day(self):
        """
        Calculates the date of the most recent completed trading day.
        
        Returns:
            1) str: ISO format date (YYYY-MM-DD).
        """
        # Get current UTC time
        now = datetime.now(timezone.utc)
        
        # If it's before 4 PM ET (approx 21:00 UTC), the current day's close data might not be ready.
        # For simplicity, we default to yesterday and roll back weekends.
        target_date = now.date() - timedelta(days=1)
        
        # Roll back Saturday (5) to Friday, Sunday (6) to Friday
        while target_date.weekday() >= 5:
            target_date -= timedelta(days=1)
            
        return target_date.isoformat()

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
            raise ValueError(f"Polygon API returned non-OK status: {status}")
            
        row = {
            "date": date,
            "open": data["open"],
            "high": data["high"],
            "low": data["low"],
            "close": data["close"],
            "volume": data["volume"]
        }
        
        return pd.DataFrame([row])

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

    def fetch_intraday_snapshot(self, ticker):
        """
        Fetches the current day's real-time snapshot for a ticker.
        Used for making predictions while the market is still open.
        
        Parameters:
            1) ticker (str): Ticker symbol.
            
        Returns:
            1) pd.DataFrame: Single-row DataFrame with the day's current OHLCV data.
        """
        ticker = ticker.upper()
        url = f"{self.base_url}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker}"
        params = {"apiKey": self.api_key}
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if data.get("status") != "OK":
            raise ValueError(f"Polygon API returned non-OK status: {data.get('status')}")
            
        ticker_data = data.get("ticker", {})
        day_stats = ticker_data.get("day", {})
        
        if not day_stats:
             raise ValueError(f"No daily snapshot data available for {ticker} at this time.")
             
        # Use current local date for the snapshot bar
        date_str = datetime.now().strftime("%Y-%m-%d")
        
        row = {
            "date": date_str,
            "open": day_stats.get("o"),
            "high": day_stats.get("h"),
            "low": day_stats.get("l"),
            "close": day_stats.get("c"),
            "volume": day_stats.get("v")
        }
        
        return pd.DataFrame([row])
