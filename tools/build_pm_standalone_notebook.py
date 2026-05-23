#!/usr/bin/env python3
"""Build the standalone PM notebook artifact."""

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = ROOT / "notebooks" / "qusa_standalone_pm.ipynb"


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(source).strip() + "\n"}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(source).strip() + "\n",
    }


cells = [
    md(
        """
        # Qusa Standalone Research Notebook

        This notebook is intentionally independent from the Qusa package and repository code. It contains the data fetcher, feature engineering, clustering, model training, backtest, and live prediction logic inline.

        Edit only the setup cell below with a Polygon/Massive API key and ticker, then run the notebook top to bottom.

        Disclaimer: this is research software, not financial advice. Backtests can overfit and do not guarantee future results.
        """
    ),
    md(
        """
        ## 1. Install Dependencies

        Run this once per notebook environment.
        """
    ),
    code(
        """
        # Uncomment if your notebook environment is missing packages.
        # %pip install pandas numpy requests scikit-learn matplotlib joblib
        """
    ),
    md("## 2. Inputs"),
    code(
        """
        API_KEY = ""  # Paste Polygon/Massive API key here
        TICKER = "UPRO"
        DAYS = 504

        RUN_CLUSTERING = True
        RUN_MONTE_CARLO = True
        FETCH_LATEST_FOR_PREDICTION = True

        # Optional email notification after the run completes.
        # For Gmail, use an app password, not your normal account password.
        SEND_EMAIL = False
        EMAIL_SETTINGS = {
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "smtp_user": "",
            "smtp_password": "",
            "from_email": "",
            "to_email": "",
            "use_tls": True,
        }

        # Keep outputs beside this notebook so it can run outside the original repo.
        OUTPUT_ROOT = "qusa_standalone_outputs"
        """
    ),
    md("## 3. Imports, Configuration, and Helpers"),
    code(
        """
        import json
        import math
        import os
        import smtplib
        import warnings
        from dataclasses import dataclass
        from datetime import datetime, timedelta, timezone
        from email.message import EmailMessage
        from pathlib import Path

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import requests
        from sklearn.cluster import DBSCAN, KMeans
        from sklearn.decomposition import PCA
        from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_score
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score, train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        warnings.filterwarnings("ignore", category=RuntimeWarning)

        CONFIG = {
            "features": {
                "rsi_window": 14,
                "atr_window": 14,
                "volume_ma_window": 20,
                "rolling_window_52w": 252,
                "vwap_window": 20,
                "vol_regime_short_window": 5,
                "vol_regime_long_window": 20,
                "advanced_vol_window": 20,
            },
            "analysis": {"abnormal_threshold": 2.0},
            "model": {
                "max_depth": 5,
                "min_samples_leaf": 10,
                "min_samples_split": 20,
                "class_weight": "balanced",
                "random_state": 0,
                "test_size": 0.25,
                "cv": 5,
                "probability_threshold": 0.7,
                "tuning": {"enabled": False},
            },
            "backtest": {
                "initial_capital": 10000,
                "position_size": 0.95,
                "transaction_cost": 0.05,
                "volatility_filter": {"enabled": True, "max_atr_pct": 3.0},
            },
            "monte_carlo": {
                "enabled": RUN_MONTE_CARLO,
                "window_size": 252,
                "iterations": 1000,
                "random_seed": 42,
                "min_data_threshold": 252,
                "horizons": [1],
                "batch_size": 500,
            },
        }

        OUTPUT_ROOT = Path(OUTPUT_ROOT).expanduser().resolve()
        PATHS = {
            "raw": OUTPUT_ROOT / "data" / "raw",
            "processed": OUTPUT_ROOT / "data" / "processed",
            "figures": OUTPUT_ROOT / "outputs" / "figures",
            "predictions": OUTPUT_ROOT / "outputs" / "predictions",
            "models": OUTPUT_ROOT / "saved_models",
        }
        for path in PATHS.values():
            path.mkdir(parents=True, exist_ok=True)

        def require_api_key(api_key):
            if not api_key or api_key.strip() in {"", "PASTE_KEY_HERE"}:
                raise ValueError("Paste a valid Polygon/Massive API key into API_KEY before running.")
            return api_key.strip()

        def clean_ticker(ticker):
            ticker = str(ticker).strip().upper()
            if not ticker:
                raise ValueError("Ticker cannot be empty.")
            return ticker

        def display_table(obj, rows=10):
            if isinstance(obj, pd.DataFrame):
                display(obj.head(rows))
            else:
                display(pd.DataFrame(obj))

        def format_pct(value):
            return f"{value * 100:.2f}%"

        def build_run_summary_email(ticker, history_rows, model_metrics, backtest_metrics, prediction, artifact_paths):
            subject = f"Qusa model run complete: {ticker} {prediction.get('direction', 'UNKNOWN')}"
            lines = [
                f"Qusa standalone model run complete for {ticker}.",
                "",
                "Latest prediction:",
                f"- Date: {prediction.get('date')}",
                f"- Direction: {prediction.get('direction')}",
                f"- Probability up: {prediction.get('probability_up', 0):.3f}",
                f"- Confidence: {prediction.get('confidence')}",
                f"- ATR%: {prediction.get('atr_pct', 0):.3f}",
                f"- Volatility filter triggered: {prediction.get('volatility_filter_triggered')}",
                "",
                "Training:",
                f"- History rows: {history_rows}",
                f"- Test accuracy: {model_metrics.get('accuracy', 0):.3f}",
                f"- CV accuracy: {model_metrics.get('cv_accuracy', 0):.3f}" if model_metrics.get("cv_accuracy") is not None else "- CV accuracy: unavailable",
                "",
                "Backtest:",
                f"- Strategy return: {format_pct(backtest_metrics.get('strategy_return', 0))}",
                f"- Buy and hold return: {format_pct(backtest_metrics.get('buy_hold_return', 0))}",
                f"- Alpha: {format_pct(backtest_metrics.get('alpha', 0))}",
                f"- Sharpe ratio: {backtest_metrics.get('sharpe_ratio', 0):.3f}",
                f"- Max drawdown: {format_pct(backtest_metrics.get('max_draw_down', 0))}",
                f"- Trades: {backtest_metrics.get('total_trades', 0)}",
                f"- Win rate: {format_pct(backtest_metrics.get('win_rate', 0))}",
                "",
                "Artifacts:",
            ]
            lines.extend(f"- {name}: {path}" for name, path in artifact_paths.items())
            lines.extend([
                "",
                "This is an automated research notification, not financial advice.",
            ])
            return subject, "\\n".join(lines)

        def send_email_notification(settings, subject, body):
            required = ["smtp_host", "smtp_port", "smtp_user", "smtp_password", "from_email", "to_email"]
            missing = [key for key in required if not settings.get(key)]
            if missing:
                raise ValueError(f"Email is enabled, but these EMAIL_SETTINGS fields are blank: {missing}")

            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = settings["from_email"]
            msg["To"] = settings["to_email"]
            msg.set_content(body)

            with smtplib.SMTP(settings["smtp_host"], int(settings["smtp_port"]), timeout=30) as server:
                if settings.get("use_tls", True):
                    server.starttls()
                server.login(settings["smtp_user"], settings["smtp_password"])
                server.send_message(msg)
        """
    ),
    md("## 4. Data Fetching"),
    code(
        """
        class PolygonFetcher:
            def __init__(self, api_key=None):
                self.api_key = require_api_key(api_key or os.getenv("POLYGON_API_KEY"))
                self.base_url = "https://api.polygon.io"

            def _get_most_recent_trading_day(self):
                now = datetime.now(timezone.utc)
                target_date = now.date() - timedelta(days=1)
                while target_date.weekday() >= 5:
                    target_date -= timedelta(days=1)
                return target_date.isoformat()

            def fetch_latest_day(self, ticker):
                ticker = clean_ticker(ticker)
                date = self._get_most_recent_trading_day()
                url = f"{self.base_url}/v1/open-close/{ticker}/{date}"
                params = {"adjusted": "true", "apiKey": self.api_key}
                response = requests.get(url, params=params, timeout=20)
                response.raise_for_status()
                data = response.json()
                if data.get("status") not in {"OK", "DELAYED"}:
                    raise ValueError(f"Polygon API returned status {data.get('status')}: {data}")
                return pd.DataFrame([{
                    "date": date,
                    "open": data["open"],
                    "high": data["high"],
                    "low": data["low"],
                    "close": data["close"],
                    "volume": data["volume"],
                }])

            def fetch_historical_range(self, ticker, start, end):
                ticker = clean_ticker(ticker)
                url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
                params = {"adjusted": "true", "sort": "asc", "apiKey": self.api_key}
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                if data.get("status") not in {"OK", "DELAYED"} and data.get("resultsCount", 0) != 0:
                    raise ValueError(f"Polygon API returned status {data.get('status')}: {data}")
                rows = []
                for r in data.get("results", []):
                    dt = datetime.fromtimestamp(r["t"] / 1000, tz=timezone.utc).date()
                    rows.append({
                        "date": dt.isoformat(),
                        "open": r["o"],
                        "high": r["h"],
                        "low": r["l"],
                        "close": r["c"],
                        "volume": r["v"],
                    })
                return pd.DataFrame(rows)

            def fetch_n_days(self, ticker, n_days):
                end_date = datetime.now(timezone.utc).date()
                start_date = end_date - timedelta(days=int(n_days * 1.6) + 5)
                df = self.fetch_historical_range(ticker, start_date.isoformat(), end_date.isoformat())
                if len(df) > n_days:
                    return df.tail(n_days).reset_index(drop=True)
                return df


        class DataLoader:
            def __init__(self, raw_data_dir, api_key):
                self.raw_dir = Path(raw_data_dir)
                self.archive_dir = self.raw_dir / "archive"
                self.fetcher = PolygonFetcher(api_key=api_key)

            def consolidate_history(self, ticker):
                ticker = clean_ticker(ticker)
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                source_files = [
                    p for p in self.raw_dir.glob(f"{ticker}_*.csv")
                    if not p.name.endswith(("_processed.csv", "_clustered.csv", "_history.csv"))
                ]
                history_path = self.raw_dir / f"{ticker}_history.csv"
                frames, skipped = [], []
                if history_path.exists():
                    try:
                        frames.append(pd.read_csv(history_path))
                    except Exception as exc:
                        skipped.append((str(history_path), str(exc)))
                for path in source_files:
                    try:
                        frames.append(pd.read_csv(path))
                    except Exception as exc:
                        skipped.append((str(path), str(exc)))
                if not frames:
                    return pd.DataFrame(), skipped
                merged = pd.concat(frames, ignore_index=True)
                if "date" not in merged.columns:
                    raise ValueError("Consolidated data is missing a date column.")
                merged["date"] = pd.to_datetime(merged["date"])
                consolidated = merged.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                consolidated["date"] = consolidated["date"].dt.strftime("%Y-%m-%d")
                consolidated.to_csv(history_path, index=False)
                if source_files:
                    self.archive_dir.mkdir(parents=True, exist_ok=True)
                    for path in source_files:
                        target = self.archive_dir / path.name
                        if target.exists():
                            target = self.archive_dir / f"{path.stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}{path.suffix}"
                        path.replace(target)
                return consolidated, skipped

            def fetch_days(self, ticker, days):
                ticker = clean_ticker(ticker)
                df = self.fetcher.fetch_n_days(ticker, days)
                if df.empty:
                    raise ValueError(f"No data returned for {ticker}.")
                temp_path = self.raw_dir / f"{ticker}_{df['date'].iloc[0]}_{df['date'].iloc[-1]}.csv"
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(temp_path, index=False)
                history, skipped = self.consolidate_history(ticker)
                return history, skipped

            def load_most_recent(self, ticker):
                ticker = clean_ticker(ticker)
                latest = self.fetcher.fetch_latest_day(ticker)
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                latest.to_csv(self.raw_dir / f"{ticker}_latest.csv", index=False)
                history, _ = self.consolidate_history(ticker)
                return history
        """
    ),
    md("## 5. Feature Engineering"),
    code(
        """
        class OvernightCalculator:
            def __init__(self, date_col="date", open_col="open", close_col="close"):
                self.date = date_col
                self.open = open_col
                self.close = close_col

            def calculate_overnight_delta(self, df):
                df = df.copy()
                df[self.date] = pd.to_datetime(df[self.date])
                df = df.sort_values(self.date).reset_index(drop=True)
                df["overnight_delta"] = df[self.open] - df[self.close].shift(1)
                df["overnight_delta_pct"] = df["overnight_delta"] / df[self.close].shift(1) * 100
                return df

            def identify_abnormal_delta(self, df, threshold=2.0, window=252):
                df = df.copy()
                rolling_mean = df["overnight_delta_pct"].rolling(window=window, min_periods=30).mean()
                rolling_std = df["overnight_delta_pct"].rolling(window=window, min_periods=30).std()
                df["z_score"] = (df["overnight_delta_pct"] - rolling_mean) / rolling_std
                df["abnormal"] = df["z_score"].abs() > threshold
                return df


        class TechnicalIndicators:
            def __init__(self, config, open_col="open", date_col="date", close_col="close", high_col="high", low_col="low", volume_col="volume"):
                self.config = config or {}
                self.period_rsi = self.config.get("rsi_window", 14)
                self.period_atr = self.config.get("atr_window", 14)
                self.period_volume = self.config.get("volume_ma_window", 20)
                self.period_rolling = self.config.get("rolling_window_52w", 252)
                self.date = date_col
                self.open = open_col
                self.close = close_col
                self.high = high_col
                self.low = low_col
                self.volume = volume_col

            def add_all(self, df):
                df = self.calculate_volume_spike(df)
                df = self.calculate_rsi(df)
                df = self.calculate_average_true_range(df)
                df = self.calculate_annual_min_max_proximity(df)
                df = self.calculate_intraday_momentum(df)
                df = self.calculate_late_day_momentum(df)
                return df

            def calculate_volume_spike(self, df, threshold=2.0):
                df = df.copy()
                df["avg_volume_20"] = df[self.volume].rolling(window=self.period_volume).mean()
                df["volume_ratio"] = df[self.volume] / df["avg_volume_20"]
                df["volume_spike"] = df["volume_ratio"] > threshold
                return df

            def calculate_rsi(self, df):
                df = df.copy()
                delta = df[self.close].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                avg_gain = gain.ewm(com=self.period_rsi - 1, min_periods=self.period_rsi).mean()
                avg_loss = loss.ewm(com=self.period_rsi - 1, min_periods=self.period_rsi).mean()
                rs = avg_gain / avg_loss
                df["rsi"] = 100 - (100 / (1 + rs))
                df["rsi_oversold"] = df["rsi"] < 30
                df["rsi_overbought"] = df["rsi"] > 70
                return df

            def calculate_average_true_range(self, df):
                df = df.copy()
                hl = df[self.high] - df[self.low]
                hc = (df[self.high] - df[self.close].shift()).abs()
                lc = (df[self.low] - df[self.close].shift()).abs()
                true_range = pd.concat([hl, hc, lc], axis=1).max(axis=1)
                df["atr"] = true_range.rolling(window=self.period_atr).mean()
                df["atr_pct"] = df["atr"] / df[self.close] * 100
                return df

            def calculate_annual_min_max_proximity(self, df):
                df = df.copy()
                df["52_week_high"] = df[self.high].rolling(window=self.period_rolling).max()
                df["52_week_low"] = df[self.low].rolling(window=self.period_rolling).min()
                df["52_week_high_proximity"] = (df["52_week_high"] - df[self.close]) / df["52_week_high"] * 100
                df["52_week_low_proximity"] = (df[self.close] - df["52_week_low"]) / df["52_week_low"] * 100
                df["52_week_high_threshold"] = df["52_week_high_proximity"] < 5
                df["52_week_low_threshold"] = df["52_week_low_proximity"] < 5
                return df

            def calculate_intraday_momentum(self, df, threshold=2.0):
                df = df.copy()
                df["intraday_return"] = (df[self.close] - df[self.open]) / df[self.open] * 100
                df["intraday_return_strong_positive"] = df["intraday_return"] > threshold
                df["intraday_return_strong_negative"] = df["intraday_return"] < -threshold
                return df

            def calculate_late_day_momentum(self, df):
                df = df.copy()
                daily_range = (df[self.high] - df[self.low]).replace(0, np.nan)
                df["close_position"] = ((df[self.close] - df[self.low]) / daily_range).fillna(0.5)
                return df


        class VolatilityCalculator:
            def __init__(self, config=None):
                self.config = config or {}
                self.vwap_window = self.config.get("vwap_window", 20)
                self.vol_regime_short = self.config.get("vol_regime_short_window", 5)
                self.vol_regime_long = self.config.get("vol_regime_long_window", 20)
                self.advanced_vol_window = self.config.get("advanced_vol_window", 20)

            def add_all(self, df):
                df = df.copy()
                tp_v = df["close"] * df["volume"]
                vwap = tp_v.rolling(self.vwap_window).sum() / df["volume"].rolling(self.vwap_window).sum()
                df["vwap_deviation"] = (df["close"] - vwap) / vwap * 100
                log_returns = np.log(df["close"] / df["close"].shift(1))
                df["vol_regime"] = log_returns.rolling(self.vol_regime_short).std() / log_returns.rolling(self.vol_regime_long).std()
                pk_val = (1.0 / (4.0 * np.log(2.0))) * np.square(np.log(df["high"] / df["low"]))
                df["vol_parkinson"] = np.sqrt(pk_val.rolling(self.advanced_vol_window).mean())
                log_hl = np.square(np.log(df["high"] / df["low"]))
                log_co = np.square(np.log(df["close"] / df["open"]))
                gk_val = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
                df["vol_garman_klass"] = np.sqrt(gk_val.rolling(self.advanced_vol_window).mean())
                return df


        class CalendarFeatures:
            def __init__(self, date_col="date"):
                self.date = date_col

            def add_all(self, df):
                df = df.copy()
                df[self.date] = pd.to_datetime(df[self.date])
                df["day_of_week"] = df[self.date].dt.day_of_week
                df["is_monday"] = df["day_of_week"] == 0
                df["is_tuesday"] = df["day_of_week"] == 1
                df["is_wednesday"] = df["day_of_week"] == 2
                df["is_thursday"] = df["day_of_week"] == 3
                df["is_friday"] = df["day_of_week"] == 4
                df["month_of_year"] = df[self.date].dt.month
                for i, name in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1):
                    df[f"is_{name}"] = df["month_of_year"] == i
                df["day_of_month"] = df[self.date].dt.day
                df["first_5d_month"] = df["day_of_month"] <= 5
                df["final_5d_month"] = df["day_of_month"] >= 25
                return df


        class MonteCarloFeatures:
            def __init__(self, config=None):
                self.config = config or {}
                self.window_size = int(self.config.get("window_size", 252))
                self.iterations = int(self.config.get("iterations", 1000))
                self.random_seed = self.config.get("random_seed", 42)
                self.min_data_threshold = int(self.config.get("min_data_threshold", 252))
                self.price_col = self.config.get("price_col", "close")
                self.horizons = self.config.get("horizons", [1])
                self.batch_size = int(self.config.get("batch_size", 500))
                self.features = self.get_feature_names(self.horizons)

            @staticmethod
            def get_feature_names(horizons=None):
                horizons = horizons or [1]
                suffixes = ["q1", "q5", "q10", "q50", "q95", "return_pct", "prob_breakeven"]
                return [f"mc_{h}d_{suffix}" for h in horizons for suffix in suffixes]

            def add_all(self, df):
                df = df.copy()
                for col in self.features:
                    if col not in df.columns:
                        df[col] = np.nan
                if "mc_1d_expected_value" not in df.columns:
                    df["mc_1d_expected_value"] = np.nan
                if len(df) < self.min_data_threshold:
                    return df
                safe_prices = df[self.price_col].replace(0, np.nan)
                log_returns = np.log(safe_prices / safe_prices.shift(1))
                rolling_mean = log_returns.rolling(self.window_size).mean()
                rolling_var = log_returns.rolling(self.window_size).var()
                rolling_std = log_returns.rolling(self.window_size).std()
                drifts = (rolling_mean - 0.5 * rolling_var).values
                vols = rolling_std.values
                current_prices = df[self.price_col].values
                np.random.seed(self.random_seed)
                dt = 1 / 252
                valid_indices = np.arange(self.min_data_threshold, len(df))
                for start in range(0, len(valid_indices), self.batch_size):
                    batch_indices = valid_indices[start:start + self.batch_size]
                    comp_drifts = drifts[batch_indices].reshape(1, -1)
                    comp_vols = vols[batch_indices].reshape(1, -1)
                    comp_prices = current_prices[batch_indices].reshape(1, -1)
                    for h in self.horizons:
                        shocks = np.random.normal(size=(self.iterations, len(batch_indices)))
                        simulated = comp_prices * np.exp(comp_drifts * (h * dt) + comp_vols * np.sqrt(h * dt) * shocks)
                        q1, q5, q10, q50, q95 = np.percentile(simulated, [1, 5, 10, 50, 95], axis=0)
                        expected = simulated.mean(axis=0)
                        prob_breakeven = (simulated > comp_prices).mean(axis=0)
                        denom = comp_prices.flatten()
                        returns_pct = np.where(denom != 0, (expected - denom) / denom * 100, 0.0)
                        idx = df.index[batch_indices]
                        df.loc[idx, f"mc_{h}d_q1"] = q1
                        df.loc[idx, f"mc_{h}d_q5"] = q5
                        df.loc[idx, f"mc_{h}d_q10"] = q10
                        df.loc[idx, f"mc_{h}d_q50"] = q50
                        df.loc[idx, f"mc_{h}d_q95"] = q95
                        df.loc[idx, f"mc_{h}d_return_pct"] = returns_pct
                        df.loc[idx, f"mc_{h}d_prob_breakeven"] = prob_breakeven
                        if h == 1:
                            df.loc[idx, "mc_1d_expected_value"] = expected
                return df


        class FeaturePipeline:
            def __init__(self, config=None):
                self.config = config or {}
                self.overnight = OvernightCalculator()
                self.technical = TechnicalIndicators(self.config.get("technical_params", {}))
                self.volatility = VolatilityCalculator(self.config.get("technical_params", {}))
                self.calendar = CalendarFeatures()
                mc_config = self.config.get("monte_carlo", {})
                self.monte_carlo = MonteCarloFeatures({**mc_config, "price_col": "close"}) if mc_config.get("enabled", False) else None

            def run(self, df):
                df = self.overnight.calculate_overnight_delta(df)
                df = self.overnight.identify_abnormal_delta(
                    df,
                    threshold=self.config.get("overnight", {}).get("abnormal_threshold", 2.0),
                )
                df = self.technical.add_all(df)
                df = self.volatility.add_all(df)
                df = self.calendar.add_all(df)
                if self.monte_carlo is not None:
                    df = self.monte_carlo.add_all(df)
                return df
        """
    ),
    md("## 6. Modeling"),
    code(
        """
        BASE_SAFE_FEATURES = [
            "52_week_high_proximity", "52_week_low_proximity", "atr_pct", "close_position",
            "rsi", "volume_ratio", "day_of_week", "day_of_month", "month_of_year",
            "first_5d_month", "final_5d_month", "is_monday", "is_tuesday", "is_wednesday",
            "is_thursday", "is_friday", "is_jan", "is_feb", "is_mar", "is_apr", "is_may",
            "is_jun", "is_jul", "is_aug", "is_sep", "is_oct", "is_nov", "is_dec",
            "vwap_deviation", "vol_regime", "mc_1d_q1", "mc_1d_q5", "mc_1d_q10",
            "mc_1d_q50", "mc_1d_q95", "mc_1d_return_pct", "mc_1d_prob_breakeven",
        ]
        BASE_SAFE_FEATURES = list(dict.fromkeys(BASE_SAFE_FEATURES))

        CONFOUND_FEATURES = [
            "overnight_delta", "overnight_delta_pct", "date", "z_score", "abnormal",
            "intraday_returns", "intraday_return_strong_positive", "intraday_return_strong_negative",
        ]

        def get_safe_features(include_monte_carlo=True, mc_horizons=None):
            features = BASE_SAFE_FEATURES.copy()
            if include_monte_carlo:
                features.extend(MonteCarloFeatures.get_feature_names(mc_horizons))
            else:
                features = [f for f in features if not f.startswith("mc_")]
            return list(dict.fromkeys(features))

        def prepare_model_features(data, feature_names):
            available = [f for f in feature_names if f in data.columns]
            missing = [f for f in feature_names if f not in data.columns]
            if missing:
                print(f"Missing {len(missing)} model features; filling them with 0: {missing}")
                data = data.copy()
                for col in missing:
                    data[col] = 0
            return data[feature_names].replace([np.inf, -np.inf], 0).fillna(0)


        class OvernightDirectionModel:
            def __init__(self, config=None):
                self.config = config or {}
                mc_conf = CONFIG.get("monte_carlo", {})
                self.feature_names = get_safe_features(mc_conf.get("enabled", False), mc_conf.get("horizons"))
                self.model = None
                self.trained_date = None
                self.metrics = {}
                self.best_params = None

            def load_data(self, data_path):
                data = pd.read_csv(data_path)
                data["target"] = (data["overnight_delta"].shift(-1) > 0).astype(int)
                data = data.iloc[:-1].dropna(subset=["overnight_delta"])
                data = data.drop(columns=CONFOUND_FEATURES, errors="ignore")
                return data

            def prepare_features(self, data):
                X = prepare_model_features(data, self.feature_names)
                y = data["target"]
                return X, y

            def train(self, X_train, y_train):
                base_model = DecisionTreeClassifier(
                    max_depth=self.config.get("max_depth", 5),
                    min_samples_leaf=self.config.get("min_samples_leaf", 10),
                    min_samples_split=self.config.get("min_samples_split", 20),
                    class_weight=self.config.get("class_weight", "balanced"),
                    random_state=self.config.get("random_state", 42),
                )
                n_splits = min(self.config.get("cv", 5), max(2, len(X_train) // 50))
                tscv = TimeSeriesSplit(n_splits=n_splits)
                tuning_config = self.config.get("tuning", {})
                if tuning_config.get("enabled", False):
                    grid_search = GridSearchCV(
                        estimator=base_model,
                        param_grid=tuning_config.get("param_grid", {
                            "max_depth": [3, 5, 8, 12],
                            "min_samples_leaf": [5, 10, 20],
                            "min_samples_split": [10, 20, 40],
                        }),
                        cv=tscv,
                        scoring="accuracy",
                        n_jobs=-1,
                    )
                    grid_search.fit(X_train, y_train)
                    self.model = grid_search.best_estimator_
                    self.best_params = grid_search.best_params_
                    self.cv_accuracy = grid_search.best_score_
                else:
                    self.model = base_model
                    scores = cross_val_score(self.model, X_train, y_train, cv=tscv)
                    self.cv_accuracy = float(scores.mean())
                    self.model.fit(X_train, y_train)
                self.trained_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                return self

            def evaluate(self, X_test, y_test):
                y_pred = self.model.predict(X_test)
                cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
                importance = pd.Series(self.model.feature_importances_, index=self.feature_names).sort_values(ascending=False)
                self.metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "cv_accuracy": getattr(self, "cv_accuracy", None),
                    "confusion_matrix": cm,
                    "true_negatives": int(cm[0, 0]),
                    "false_positives": int(cm[0, 1]),
                    "false_negatives": int(cm[1, 0]),
                    "true_positives": int(cm[1, 1]),
                    "feature_importance": importance.to_dict(),
                }
                return self.metrics

            def save_model(self, save_path):
                bundle = {
                    "model": self.model,
                    "features": self.feature_names,
                    "threshold": self.config["probability_threshold"],
                    "target": "next_overnight_delta_positive",
                    "trained_date": self.trained_date,
                    "config": self.config,
                    "metrics": self.metrics,
                }
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(bundle, save_path)
                return save_path


        def train_model(data_path, save_path, config):
            model = OvernightDirectionModel(config)
            data = model.load_data(data_path)
            X, y = model.prepare_features(data)
            if len(X) < 60:
                raise ValueError(f"Need at least 60 model rows after feature engineering; got {len(X)}.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=config["test_size"], shuffle=False
            )
            model.train(X_train, y_train)
            model.evaluate(X_test, y_test)
            model.save_model(save_path)
            return model, X_train, X_test, y_train, y_test
        """
    ),
    md("## 7. Backtesting and Prediction"),
    code(
        """
        class ModelBacktester:
            def __init__(self, model_path, backtest_data_path):
                bundle = joblib.load(model_path)
                self.model = bundle["model"]
                self.features = bundle["features"]
                self.threshold = bundle["threshold"]
                self.data = pd.read_csv(backtest_data_path)
                self.data["date"] = pd.to_datetime(self.data["date"])
                self.results = None

            def run_backtest(self, initial_capital, position_size, transaction_cost, volatility_filter=None):
                X = prepare_model_features(self.data, self.features)
                y_prob = self.model.predict_proba(X)[:, 1]
                results = self.data[["date", "close", "overnight_delta"]].copy()
                if "atr_pct" in self.data.columns:
                    results["atr_pct"] = self.data["atr_pct"]
                results["probability_up"] = y_prob
                results["signal"] = 0
                results.loc[results["probability_up"] >= self.threshold, "signal"] = 1
                results.loc[results["probability_up"] <= (1 - self.threshold), "signal"] = -1
                results["vol_skipped"] = False
                if volatility_filter and volatility_filter.get("enabled", False) and "atr_pct" in results.columns:
                    mask = (results["signal"] != 0) & (results["atr_pct"] > volatility_filter.get("max_atr_pct", 100.0))
                    results.loc[mask, "vol_skipped"] = True
                    results.loc[mask, "signal"] = 0
                results["overnight_return"] = results["overnight_delta"].shift(-1) / 100
                results = results.iloc[:-1].copy()
                results["strategy_return"] = 0.0
                results["trade_count"] = 0
                cost = (transaction_cost / 100) * 2
                results.loc[results["signal"] == 1, "strategy_return"] = (results["overnight_return"] - cost) * position_size
                results.loc[results["signal"] == -1, "strategy_return"] = (-results["overnight_return"] - cost) * position_size
                results.loc[results["signal"] != 0, "trade_count"] = 1
                results["strategy_cumulative"] = (1 + results["strategy_return"]).cumprod()
                results["strategy_value"] = initial_capital * results["strategy_cumulative"]
                results["buy_hold_value"] = initial_capital * (results["close"] / results["close"].iloc[0])
                self.results = results
                return results

            def calculate_metrics(self, initial_capital):
                r = self.results
                total_trades = int(r["trade_count"].sum())
                winning = int(((r["trade_count"] == 1) & (r["strategy_return"] > 0)).sum())
                losing = int(((r["trade_count"] == 1) & (r["strategy_return"] < 0)).sum())
                final_val = float(r["strategy_value"].iloc[-1])
                bh_final = float(r["buy_hold_value"].iloc[-1])
                strategy_return = (final_val - initial_capital) / initial_capital
                bh_return = (bh_final - initial_capital) / initial_capital
                daily = r["strategy_return"]
                sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
                peak = r["strategy_value"].cummax()
                drawdown = (r["strategy_value"] - peak) / peak
                return {
                    "initial_capital": initial_capital,
                    "strategy_value": final_val,
                    "buy_hold_value": bh_final,
                    "strategy_return": strategy_return,
                    "buy_hold_return": bh_return,
                    "alpha": strategy_return - bh_return,
                    "annual_volatility": float(daily.std() * np.sqrt(252)),
                    "sharpe_ratio": sharpe,
                    "max_draw_down": float(drawdown.min()),
                    "total_trades": total_trades,
                    "skipped_vol_trades": int(r["vol_skipped"].sum()) if "vol_skipped" in r else 0,
                    "winning_trades": winning,
                    "losing_trades": losing,
                    "win_rate": winning / total_trades if total_trades else 0.0,
                }

            def plot_results(self, save_path):
                r = self.results
                fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={"height_ratios": [2, 1, 1]})
                axes[0].plot(r["date"], r["strategy_value"], label="Overnight Strategy", linewidth=2)
                axes[0].plot(r["date"], r["buy_hold_value"], label="Buy & Hold", linestyle="--", alpha=0.8)
                axes[0].set_title("Strategy Performance vs Benchmark")
                axes[0].set_ylabel("Portfolio Value")
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                peak = r["strategy_value"].cummax()
                drawdown = (r["strategy_value"] - peak) / peak * 100
                axes[1].fill_between(r["date"], drawdown, 0, alpha=0.25)
                axes[1].plot(r["date"], drawdown, linewidth=1)
                axes[1].set_title("Strategy Drawdown")
                axes[1].set_ylabel("Drawdown %")
                axes[1].grid(True, alpha=0.3)
                trades = r.loc[r["strategy_return"] != 0, "strategy_return"]
                if len(trades):
                    axes[2].hist(trades, bins=40, alpha=0.75)
                    axes[2].axvline(0, linestyle="--", linewidth=1)
                else:
                    axes[2].text(0.5, 0.5, "No trades executed", ha="center", va="center")
                axes[2].set_title("Trade Return Distribution")
                axes[2].grid(True, alpha=0.3)
                plt.tight_layout()
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=160, bbox_inches="tight")
                plt.show()
                return save_path


        class LivePredictor:
            def __init__(self, model_path):
                bundle = joblib.load(model_path)
                self.model = bundle["model"]
                self.features = bundle["features"]
                self.threshold = bundle["threshold"]
                self.trained_date = bundle.get("trained_date", "Unknown")

            def predict(self, data, volatility_filter=None):
                latest = data.tail(1).copy()
                X = prepare_model_features(latest, self.features)
                y_pred = int(self.model.predict(X)[0])
                y_prob = float(self.model.predict_proba(X)[0, 1])
                confidence = "HIGH" if y_prob >= self.threshold or y_prob <= (1 - self.threshold) else "LOW"
                atr_pct = float(latest["atr_pct"].iloc[0]) if "atr_pct" in latest.columns and pd.notna(latest["atr_pct"].iloc[0]) else 0.0
                vol_triggered = bool(volatility_filter and volatility_filter.get("enabled", False) and atr_pct > volatility_filter.get("max_atr_pct", 100.0))
                return {
                    "date": str(latest["date"].iloc[0]) if "date" in latest.columns else None,
                    "prediction": y_pred,
                    "direction": "UP" if y_pred == 1 else "DOWN",
                    "probability_up": y_prob,
                    "confidence": confidence,
                    "volatility_filter_triggered": vol_triggered,
                    "atr_pct": atr_pct,
                    "trained_date": self.trained_date,
                }
        """
    ),
    md("## 8. Clustering"),
    code(
        """
        class ClusterAnalyzer:
            def __init__(self, n_clusters=4, algorithm="kmeans", random_state=42):
                self.n_clusters = n_clusters
                self.algorithm = algorithm
                self.random_state = random_state
                self.scaler = None
                self.pca = None
                self.model = None
                self.feature_columns = None
                self.cluster_profiles = None
                self.cluster_labels = None

            def prepare_features(self, df, feature_cols=None):
                feature_cols = feature_cols or [
                    "overnight_delta_pct", "vol_regime", "atr_pct", "volume_ratio", "rsi",
                    "close_position", "52_week_high_proximity", "52_week_low_proximity",
                ]
                available = [c for c in feature_cols if c in df.columns]
                if not available:
                    raise ValueError("No valid feature columns found for clustering.")
                self.feature_columns = available
                df_features = df[available].replace([np.inf, -np.inf], np.nan).dropna().reset_index()
                if len(df_features) < self.n_clusters:
                    raise ValueError(f"Not enough valid samples for clustering: {len(df_features)}.")
                self.scaler = self.scaler or StandardScaler()
                X_scaled = self.scaler.fit_transform(df_features[available])
                return X_scaled, available, df_features["index"].values

            def find_optimal_clusters(self, df, feature_cols=None, max_k=10):
                X, _, _ = self.prepare_features(df, feature_cols)
                max_k = min(max_k, max(2, len(X) - 1))
                results = {"k": [], "inertia": [], "silhouette_score": []}
                for k in range(2, max_k + 1):
                    model = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                    labels = model.fit_predict(X)
                    results["k"].append(k)
                    results["inertia"].append(model.inertia_)
                    results["silhouette_score"].append(silhouette_score(X, labels))
                results["optimal_k"] = results["k"][int(np.argmax(results["silhouette_score"]))]
                return results

            def fit_clusters(self, df, feature_cols=None):
                X, features, valid_indices = self.prepare_features(df, feature_cols)
                if self.algorithm == "kmeans":
                    model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
                elif self.algorithm == "dbscan":
                    model = DBSCAN(eps=0.5, min_samples=5)
                else:
                    raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")
                labels = model.fit_predict(X)
                self.model = model
                self.cluster_labels = labels
                df = df.copy()
                df["cluster"] = -1
                df.loc[valid_indices, "cluster"] = labels
                self._calculate_cluster_profiles(df)
                return df

            def _calculate_cluster_profiles(self, df):
                valid = df[df["cluster"] != -1]
                profiles = []
                means = valid.groupby("cluster")[self.feature_columns].mean()
                for cluster_label in sorted(valid["cluster"].unique()):
                    rows = valid[valid["cluster"] == cluster_label]
                    profile = {"cluster": int(cluster_label), "count": len(rows), "proportion": len(rows) / len(valid)}
                    for feature in self.feature_columns:
                        profile[f"mean_{feature}"] = means.loc[cluster_label, feature]
                    profiles.append(profile)
                self.cluster_profiles = pd.DataFrame(profiles)

            def perform_pca(self, df, n_components=2):
                X, _, _ = self.prepare_features(df, self.feature_columns)
                self.pca = PCA(n_components=n_components, random_state=self.random_state)
                return self.pca.fit_transform(X), self.pca

            def interpret_clusters(self):
                interpretations = {}
                for _, row in self.cluster_profiles.iterrows():
                    label = int(row["cluster"])
                    overnight = row.get("mean_overnight_delta_pct", 0)
                    vol_regime = row.get("mean_vol_regime", 1.0)
                    atr = row.get("mean_atr_pct", 1.0)
                    volume = row.get("mean_volume_ratio", 1.0)
                    rsi = row.get("mean_rsi", 50)
                    if (vol_regime > 1.2) or (atr > 2.0):
                        text = "High volatility breakout or panic regime" if volume > 1.5 else "Elevated volatility regime"
                    elif (vol_regime < 0.8) and (volume < 1.0):
                        text = "Low volatility consolidation"
                    elif (overnight > 1.0) and (rsi > 65):
                        text = "Bullish momentum regime"
                    elif (overnight < -1.0) and (rsi < 35):
                        text = "Bearish exhaustion regime"
                    elif rsi > 70:
                        text = "Overbought regime"
                    elif rsi < 30:
                        text = "Oversold regime"
                    else:
                        text = "Standard market dynamics"
                    interpretations[label] = text
                return interpretations

            def plot_pca(self, clustered_df, save_path):
                pca_X, pca_model = self.perform_pca(clustered_df)
                valid = clustered_df[clustered_df["cluster"] != -1]
                fig, ax = plt.subplots(figsize=(9, 6))
                scatter = ax.scatter(pca_X[:, 0], pca_X[:, 1], c=valid["cluster"], cmap="coolwarm", alpha=0.8)
                plt.colorbar(scatter, ax=ax, label="Regime")
                ax.set_title("PCA Cluster Visualization")
                ax.set_xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0] * 100:.1f}% variance)")
                ax.set_ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1] * 100:.1f}% variance)")
                ax.grid(True, alpha=0.2)
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=160, bbox_inches="tight")
                plt.show()
                return save_path
        """
    ),
    md("## 9. One-Button Pipeline"),
    code(
        """
        def run_full_pipeline(
            api_key,
            ticker,
            days=504,
            run_clustering=True,
            run_monte_carlo=True,
            fetch_latest_for_prediction=True,
            send_email=False,
            email_settings=None,
        ):
            ticker = clean_ticker(ticker)
            CONFIG["monte_carlo"]["enabled"] = bool(run_monte_carlo)
            print(f"Running standalone Qusa pipeline for {ticker}")
            print(f"Outputs: {OUTPUT_ROOT}")

            loader = DataLoader(PATHS["raw"], api_key=api_key)
            history, skipped = loader.fetch_days(ticker, days)
            print(f"Fetched/consolidated {len(history)} rows.")
            if skipped:
                print(f"Skipped unreadable files: {skipped}")

            pipeline = FeaturePipeline({
                "overnight": {"abnormal_threshold": CONFIG["analysis"]["abnormal_threshold"]},
                "technical_params": CONFIG["features"],
                "monte_carlo": CONFIG["monte_carlo"],
            })
            processed = pipeline.run(history)
            processed_path = PATHS["processed"] / f"{ticker}_processed.csv"
            processed.to_csv(processed_path, index=False)
            print(f"Processed data saved: {processed_path}")

            model_path = PATHS["models"] / f"{ticker.lower()}_model.pkl"
            model, X_train, X_test, y_train, y_test = train_model(processed_path, model_path, CONFIG["model"])
            print(f"Model saved: {model_path}")
            print(f"Test accuracy: {model.metrics['accuracy']:.3f}")
            display(pd.DataFrame(model.metrics["confusion_matrix"], index=["Actual DOWN", "Actual UP"], columns=["Pred DOWN", "Pred UP"]))
            display(pd.Series(model.metrics["feature_importance"]).sort_values(ascending=False).head(15).to_frame("importance"))

            backtester = ModelBacktester(model_path, processed_path)
            backtest_results = backtester.run_backtest(
                initial_capital=CONFIG["backtest"]["initial_capital"],
                position_size=CONFIG["backtest"]["position_size"],
                transaction_cost=CONFIG["backtest"]["transaction_cost"],
                volatility_filter=CONFIG["backtest"]["volatility_filter"],
            )
            backtest_metrics = backtester.calculate_metrics(CONFIG["backtest"]["initial_capital"])
            backtest_csv = PATHS["figures"] / f"{ticker}_backtest_results.csv"
            backtest_results.to_csv(backtest_csv, index=False)
            backtest_plot = backtester.plot_results(PATHS["figures"] / f"{ticker}_backtest.png")
            display(pd.Series(backtest_metrics).to_frame("value"))

            clustering_output = None
            if run_clustering:
                analyzer = ClusterAnalyzer(n_clusters=4, algorithm="kmeans", random_state=42)
                optimal = analyzer.find_optimal_clusters(processed, max_k=10)
                analyzer.n_clusters = optimal["optimal_k"]
                clustered = analyzer.fit_clusters(processed)
                clustered_path = PATHS["processed"] / f"{ticker}_clustered.csv"
                clustered.to_csv(clustered_path, index=False)
                pca_plot = analyzer.plot_pca(clustered, PATHS["figures"] / f"{ticker}_pca_clusters.png")
                interpretations = analyzer.interpret_clusters()
                print("Cluster interpretations:")
                for label, text in interpretations.items():
                    print(f"  Regime {label}: {text}")
                display(analyzer.cluster_profiles)
                clustering_output = {
                    "optimal": optimal,
                    "clustered_path": str(clustered_path),
                    "pca_plot": str(pca_plot),
                    "profiles": analyzer.cluster_profiles,
                    "interpretations": interpretations,
                }

            if fetch_latest_for_prediction:
                latest_history = loader.load_most_recent(ticker)
                latest_processed = pipeline.run(latest_history)
                latest_processed.to_csv(processed_path, index=False)
            else:
                latest_processed = processed

            predictor = LivePredictor(model_path)
            prediction = predictor.predict(latest_processed, volatility_filter=CONFIG["backtest"]["volatility_filter"])
            prediction_log = PATHS["predictions"] / "prediction_log.csv"
            pd.DataFrame([{"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "ticker": ticker, **prediction}]).to_csv(
                prediction_log,
                mode="a",
                header=not prediction_log.exists(),
                index=False,
            )
            print("Latest prediction:")
            display(pd.DataFrame([prediction]))

            results = {
                "ticker": ticker,
                "history_rows": len(history),
                "processed_path": str(processed_path),
                "model_path": str(model_path),
                "training_metrics": model.metrics,
                "backtest_metrics": backtest_metrics,
                "backtest_csv": str(backtest_csv),
                "backtest_plot": str(backtest_plot),
                "prediction": prediction,
                "prediction_log": str(prediction_log),
                "clustering": clustering_output,
            }

            if send_email:
                artifact_paths = {
                    "processed_data": str(processed_path),
                    "model": str(model_path),
                    "backtest_csv": str(backtest_csv),
                    "backtest_plot": str(backtest_plot),
                    "prediction_log": str(prediction_log),
                }
                if clustering_output:
                    artifact_paths["clustered_data"] = clustering_output["clustered_path"]
                    artifact_paths["pca_plot"] = clustering_output["pca_plot"]

                subject, body = build_run_summary_email(
                    ticker=ticker,
                    history_rows=len(history),
                    model_metrics=model.metrics,
                    backtest_metrics=backtest_metrics,
                    prediction=prediction,
                    artifact_paths=artifact_paths,
                )
                try:
                    send_email_notification(email_settings or {}, subject, body)
                    print(f"Email notification sent to {(email_settings or {}).get('to_email')}.")
                    results["email_sent"] = True
                except Exception as exc:
                    print(f"Email notification failed: {exc}")
                    results["email_sent"] = False
                    results["email_error"] = str(exc)

            return results
        """
    ),
    md("## 10. Run Everything"),
    code(
        """
        results = run_full_pipeline(
            api_key=API_KEY,
            ticker=TICKER,
            days=DAYS,
            run_clustering=RUN_CLUSTERING,
            run_monte_carlo=RUN_MONTE_CARLO,
            fetch_latest_for_prediction=FETCH_LATEST_FOR_PREDICTION,
            send_email=SEND_EMAIL,
            email_settings=EMAIL_SETTINGS,
        )
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def main():
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    print(NOTEBOOK_PATH)


if __name__ == "__main__":
    main()
