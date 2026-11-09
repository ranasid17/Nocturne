"""Volatility forecasting module for stock price floor analysis.

This module starts from the recommended order:
1. baseline rolling realized volatility,
2. EWMA volatility model,
3. GARCH(1,1) volatility model,
4. walk-forward time-series validation for model selection.

The idea is to forecast future realized volatility and only use the most reliable
volatility model in the Monte Carlo floor generator once it has been validated
out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def fetch_prices(ticker: str, start_date: str = "2016-01-01", end_date: str | None = None) -> pd.DataFrame:
    """Download adjusted close prices for a ticker."""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker}")

    if "Close" not in data.columns:
        raise ValueError(f"Close price column missing for {ticker}")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        if close.shape[1] != 1:
            raise ValueError(f"Expected one close-price column for {ticker}")
        close = close.iloc[:, 0]

    close = close.dropna().astype(float)
    close.name = "close"
    return close.to_frame()


def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series:
    """Compute log returns."""
    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError("Expected a Series or one-column DataFrame")
        prices = prices.iloc[:, 0]

    prices = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if (prices <= 0).any():
        raise ValueError("Prices must be positive to compute log returns")

    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Baseline volatility estimators
# ---------------------------------------------------------------------------


def realized_volatility_annualized(prices: pd.Series | pd.DataFrame, window: int = 20) -> pd.Series:
    """Annualized realized volatility using rolling standard deviation of log returns."""
    rets = log_returns(prices)
    return rets.rolling(window).std(ddof=1) * np.sqrt(252)


# ---------------------------------------------------------------------------
# EWMA estimator
# ---------------------------------------------------------------------------

def ewma_volatility_forecast(prices: pd.Series | pd.DataFrame, span: int = 20) -> pd.Series:
    """EWMA forecast of annualized volatility from squared returns."""
    rets = log_returns(prices)
    variance = rets.pow(2).ewm(span=span, adjust=False).mean()
    return np.sqrt(variance * 252)


# ---------------------------------------------------------------------------
# GARCH(1,1) estimator (custom numerical implementation)
# ---------------------------------------------------------------------------


def garch11_fit(returns: pd.Series) -> Tuple[float, float, float, np.ndarray]:
    """Estimate GARCH(1,1) parameters using a simple Gaussian likelihood.

    Returns:
        omega, alpha, beta, variance_series
    """
    r = returns.to_numpy(dtype=float)
    if len(r) < 50:
        raise ValueError("Need at least 50 observations to estimate GARCH(1,1)")

    def neg_log_likelihood(params: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 1e-8 or alpha <= 1e-8 or beta <= 1e-8:
            return 1e12
        if alpha + beta >= 0.999:
            return 1e12

        var = np.empty(len(r), dtype=float)
        var[0] = np.var(r)
        for i in range(1, len(r)):
            var[i] = omega + alpha * (r[i - 1] ** 2) + beta * var[i - 1]

        if np.any(var <= 0):
            return 1e12

        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (r ** 2) / var)
        return -ll

    # Reasonable starting values.
    init = np.array([1e-6, 0.08, 0.90], dtype=float)
    bounds = [
        (1e-8, 1e-2),
        (1e-8, 0.5),
        (1e-8, 0.98),
    ]

    result = minimize(neg_log_likelihood, init, method="L-BFGS-B", bounds=bounds)
    if not result.success:
        omega, alpha, beta = init
    else:
        omega, alpha, beta = result.x

    var = np.empty(len(r), dtype=float)
    var[0] = np.var(r)
    for i in range(1, len(r)):
        var[i] = omega + alpha * (r[i - 1] ** 2) + beta * var[i - 1]

    return float(omega), float(alpha), float(beta), var


def garch11_forecast(prices: pd.Series | pd.DataFrame, horizon_days: int = 30) -> pd.Series:
    """Generate a GARCH(1,1) volatility forecast series using the current estimated state."""
    rets = log_returns(prices)
    omega, alpha, beta, variance = garch11_fit(rets)

    # Build a series of forecast values using the estimated recursive variance.
    forecast_series = pd.Series(np.sqrt(variance * 252), index=rets.index, name="garch11_vol")

    # For the next horizon, use the most recent forecasted variance.
    most_recent_variance = variance[-1]
    forecast_scalar = np.sqrt(most_recent_variance * 252) * np.sqrt(horizon_days / 252)
    forecast_series.iloc[-1] = forecast_scalar

    return forecast_series


# ---------------------------------------------------------------------------
# Forecast-target construction for validation
# ---------------------------------------------------------------------------


def annualized_horizon_realized_volatility(returns: pd.Series, horizon_days: int = 30) -> pd.Series:
    """Create target: realised annualized volatility over the next horizon period."""
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")

    target = (
        returns.shift(-horizon_days)
        .rolling(window=horizon_days, min_periods=horizon_days)
        .std(ddof=1)
        * np.sqrt(252)
    )
    return target.rename(f"target_vol_{horizon_days}d")


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def rolling_vol_forecast(prices: pd.Series | pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling realized-vol forecast using the latest window."""
    rets = log_returns(prices)
    return rets.rolling(window).std(ddof=1) * np.sqrt(252)


def validate_volatility_models(
    prices: pd.Series | pd.DataFrame,
    horizon_days: int = 30,
    train_min_size: int = 252,
    validation_window: int = 250,
    forecast_windows: Iterable[int] = (20, 60, 126, 252),
) -> pd.DataFrame:
    """Walk-forward validation comparing baseline, EWMA, and GARCH models.

    This follows the recommended time-series validation process instead of a random split.
    """
    rets = log_returns(prices)
    actual = annualized_horizon_realized_volatility(rets, horizon_days=horizon_days)

    rows: List[Dict[str, float]] = []

    for cutoff in range(train_min_size, len(rets) - horizon_days, validation_window):
        train_rets = rets.iloc[:cutoff]
        forecast_date = train_rets.index[-1]

        actual_value = actual.loc[forecast_date]
        if pd.isna(actual_value):
            continue

        baseline_forecast = (
            train_rets.tail(max(forecast_windows)).std(ddof=1) * np.sqrt(252)
        )

        ewma_forecast_value = (
            train_rets.pow(2).ewm(span=max(forecast_windows), adjust=False).mean().iloc[-1] * 252
        )
        ewma_forecast_value = np.sqrt(ewma_forecast_value)

        garch_omega, garch_alpha, garch_beta, garch_variance = garch11_fit(train_rets)
        garch_forecast_value = np.sqrt(garch_variance[-1] * 252) * np.sqrt(horizon_days / 252)

        for model_name, pred in {
            "rolling_baseline": baseline_forecast,
            "ewma": ewma_forecast_value,
            "garch11": garch_forecast_value,
        }.items():
            rows.append(
                {
                    "cutoff_date": forecast_date,
                    "model": model_name,
                    "actual_annualized_vol": float(actual_value),
                    "forecast_annualized_vol": float(pred),
                    "abs_error": float(abs(float(actual_value) - float(pred))),
                    "pct_error": float((float(pred) - float(actual_value)) / float(actual_value)),
                }
            )

    if not rows:
        raise ValueError("No validation rows were produced")

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("model")
        .agg(
            mae=("abs_error", "mean"),
            rmse=("abs_error", lambda x: np.sqrt(np.mean(np.square(x)))),
            mean_pct_error=("pct_error", "mean"),
            avg_forecast=("forecast_annualized_vol", "mean"),
            avg_actual=("actual_annualized_vol", "mean"),
        )
        .reset_index()
    )

    return df, summary


# ---------------------------------------------------------------------------
# Monte Carlo integration helpers (planned next step)
# ---------------------------------------------------------------------------


def forecasted_volatility_for_mc(
    prices: pd.Series | pd.DataFrame,
    method: str = "rolling_baseline",
    window: int = 252,
    span: int = 20,
    horizon_days: int = 30,
) -> float:
    """Return the selected volatility estimate for the next MC run.

    This is the bridge between the volatility forecast module and the Monte Carlo floor model.
    """
    rets = log_returns(prices)

    if method == "rolling_baseline":
        return float(rets.tail(window).std(ddof=1) * np.sqrt(252) * np.sqrt(horizon_days / 252))

    if method == "ewma":
        variance = rets.pow(2).ewm(span=span, adjust=False).mean().iloc[-1]
        return float(np.sqrt(variance * 252) * np.sqrt(horizon_days / 252))

    if method == "garch11":
        omega, alpha, beta, variance = garch11_fit(rets)
        return float(np.sqrt(variance[-1] * 252) * np.sqrt(horizon_days / 252))

    raise ValueError("Unknown forecasting method")


# ---------------------------------------------------------------------------
# Example runner
# ---------------------------------------------------------------------------


def main() -> None:
    ticker = "AAPL"
    prices = fetch_prices(ticker, start_date="2016-01-01")
    close_prices = prices["close"]

    print(f"Loaded {len(close_prices)} observations for {ticker}")

    rolling = realized_volatility_annualized(close_prices, window=252).tail(10)
    print("\nRecent rolling 252-day realized volatility:")
    print(rolling.tail(10).to_string())

    ewma = ewma_volatility_forecast(close_prices, span=20).tail(10)
    print("\nRecent EWMA volatility estimate:")
    print(ewma.tail(10).to_string())

    horizon_days = 30
    validation_rows, validation_summary = validate_volatility_models(
        close_prices,
        horizon_days=horizon_days,
        train_min_size=252,
        validation_window=90,
        forecast_windows=(20, 60, 252),
    )

    print(f"\nWalk-forward validation summary for {horizon_days}-day volatility forecast:")
    print(validation_summary.sort_values("mae").to_string(index=False))

    print("\nExample current forecasted vol estimates:")
    for method in ("rolling_baseline", "ewma", "garch11"):
        value = forecasted_volatility_for_mc(close_prices, method=method, window=252, span=20, horizon_days=horizon_days)
        print(f"  {method}: {value:.4f} annualized")


if __name__ == "__main__":
    main()
