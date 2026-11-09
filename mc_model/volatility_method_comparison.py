"""Volatility forecast comparison module.

This file is intentionally separate from the existing floor-model workflow so the
Monte Carlo floor scripts remain stable while we compare candidate volatility
forecasting methods.

Recommended model order:
1. rolling realized volatility baseline
2. EWMA volatility forecast
3. GARCH(1,1) volatility forecast
4. optional feature-enriched forecast layer (not yet wired into the MC floor)

The purpose is to compare each method using time-series walk-forward validation
before deciding which volatility estimate should feed the Monte Carlo floor model.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def fetch_close_prices(ticker: str, start_date: str = "2016-01-01", end_date: str | None = None) -> pd.Series:
    """Download adjusted close prices and return them as a Series."""
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No price data returned for {ticker}")

    if "Close" not in data.columns:
        raise ValueError(f"Close column missing for {ticker}")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    close = close.dropna().astype(float)
    close.name = "close"
    return close


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns."""
    prices = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if (prices <= 0).any():
        raise ValueError("Prices must be positive to compute log returns")
    return np.log(prices / prices.shift(1)).dropna()


def realized_volatility_annualized(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling annualized realized volatility from log returns."""
    return returns.rolling(window).std(ddof=1) * np.sqrt(252)


# ---------------------------------------------------------------------------
# Candidate forecast methods
# ---------------------------------------------------------------------------

def rolling_vol_predictor(train_returns: pd.Series, horizon_days: int, window: int = 252) -> float:
    """Baseline: use the most recent realized annualized volatility."""
    if len(train_returns) < window:
        window = max(5, len(train_returns))
    vol = train_returns.tail(window).std(ddof=1) * np.sqrt(252)
    return float(vol)


def ewma_vol_predictor(train_returns: pd.Series, horizon_days: int, span: int = 20) -> float:
    """EWMA forecast of annualized volatility."""
    variance = train_returns.pow(2).ewm(span=span, adjust=False).mean().iloc[-1]
    return float(np.sqrt(variance * 252))


def garch11_fit(returns: pd.Series) -> Tuple[float, float, float, np.ndarray]:
    """Estimate GARCH(1,1) parameters using a Gaussian quasi-likelihood."""
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
            var[i] = omega + alpha * r[i - 1] ** 2 + beta * var[i - 1]

        if np.any(var <= 0):
            return 1e12

        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (r ** 2) / var)
        return -ll

    init = np.array([1e-6, 0.08, 0.90], dtype=float)
    bounds = [(1e-8, 1e-2), (1e-8, 0.5), (1e-8, 0.98)]
    result = minimize(neg_log_likelihood, init, method="L-BFGS-B", bounds=bounds)

    if result.success:
        omega, alpha, beta = result.x
    else:
        omega, alpha, beta = init

    var = np.empty(len(r), dtype=float)
    var[0] = np.var(r)
    for i in range(1, len(r)):
        var[i] = omega + alpha * r[i - 1] ** 2 + beta * var[i - 1]

    return float(omega), float(alpha), float(beta), var


def garch11_predictor(train_returns: pd.Series, horizon_days: int, omega_init: float | None = None) -> float:
    """GARCH(1,1) forecast using the most recent conditional variance."""
    omega, alpha, beta, var = garch11_fit(train_returns)
    latest_variance = var[-1]
    annualized_vol = np.sqrt(latest_variance * 252)
    return float(annualized_vol)


def feature_enriched_vol_predictor(train_returns: pd.Series, horizon_days: int, price_series: pd.Series | None = None) -> float:
    """A lightweight feature-enriched forecast based on multiple volatility regimes and recent shocks.

    This is the first practical upgrade beyond the baseline/EWMA/GARCH methods and is meant to
    reflect the feature set we discussed earlier: lagged vol windows, recent shock intensity, and
    trend/stress indicators.
    """
    if price_series is None:
        # Use a purely return-based approximation when no price series is available.
        price_series = pd.Series(np.ones(len(train_returns) + 1), index=pd.RangeIndex(len(train_returns) + 1))

    rets = train_returns.dropna()
    if len(rets) < 20:
        return float(np.sqrt(np.var(rets) * 252))

    rolling_5 = rets.tail(5).std(ddof=1) * np.sqrt(252)
    rolling_10 = rets.tail(10).std(ddof=1) * np.sqrt(252)
    rolling_20 = rets.tail(20).std(ddof=1) * np.sqrt(252)
    rolling_40 = rets.tail(40).std(ddof=1) * np.sqrt(252)
    rolling_60 = rets.tail(60).std(ddof=1) * np.sqrt(252)
    ewma = np.sqrt(rets.pow(2).ewm(span=20, adjust=False).mean().iloc[-1] * 252)
    recent_abs_return = rets.abs().tail(20).mean() * np.sqrt(252)

    if price_series is not None and len(price_series) >= 20:
        recent_prices = price_series.tail(20)
        drawdown = float((recent_prices / recent_prices.max() - 1.0).iloc[-1])
        trend = float(rets.tail(20).mean() * 252)  # rough normalized stress signal
    else:
        drawdown = 0.0
        trend = 0.0

    # Blend several regimes: recent volatility, medium-term volatility, stress sensitivity.
    forecast = (
        0.35 * rolling_20
        + 0.20 * rolling_40
        + 0.15 * rolling_60
        + 0.20 * ewma
        + 0.10 * recent_abs_return
    )

    if drawdown < -0.10:
        forecast *= 1.12
    elif drawdown > 0.10:
        forecast *= 0.92

    if trend < -0.02:
        forecast *= 1.05

    return float(max(forecast, 0.02))


# ---------------------------------------------------------------------------
# Feature matrix for a richer, future model
# ---------------------------------------------------------------------------

def build_feature_frame(prices: pd.Series, horizon_days: int = 30) -> pd.DataFrame:
    """Create a feature matrix with lagged realized-vol and shock-like features.

    These are the features I would recommend adding next after the baseline models.
    """
    rets = log_returns(prices)
    frame = pd.DataFrame(index=rets.index)

    for window in [5, 10, 20, 40, 60, 126, 252]:
        frame[f"realized_vol_{window}"] = realized_volatility_annualized(rets, window)
        frame[f"abs_return_{window}"] = rets.abs().rolling(window).mean()
        frame[f"ret_std_{window}"] = rets.rolling(window).std(ddof=1)

    frame["return_trend_20"] = rets.rolling(20).mean()
    frame["return_trend_60"] = rets.rolling(60).mean()
    frame["drawdown_20"] = (prices / prices.rolling(20).max() - 1.0)
    frame["volume_regime"] = 0.0

    frame["target_vol"] = (
        rets.shift(-horizon_days)
        .rolling(window=horizon_days, min_periods=horizon_days)
        .std(ddof=1)
        * np.sqrt(252)
    )

    return frame.dropna()


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------

def evaluate_method(
    prices: pd.Series,
    method: Callable[[pd.Series, int], float],
    horizon_days: int = 30,
    train_min_size: int = 252,
    step_size: int = 30,
) -> pd.DataFrame:
    """Run time-series walk-forward validation for a single model."""
    rets = log_returns(prices)
    rows: List[Dict[str, float]] = []

    for cutoff in range(train_min_size, len(rets) - horizon_days, step_size):
        train_returns = rets.iloc[:cutoff]
        validation_returns = rets.iloc[cutoff:cutoff + horizon_days]

        forecast = method(train_returns, horizon_days)
        actual_vol = validation_returns.std(ddof=1) * np.sqrt(252)

        if np.isnan(actual_vol) or actual_vol <= 0:
            continue

        error = forecast - actual_vol
        abs_error = abs(error)
        pct_error = error / actual_vol
        mape = abs_error / actual_vol

        rows.append(
            {
                "cutoff_date": train_returns.index[-1],
                "forecast_vol": float(forecast),
                "actual_vol": float(actual_vol),
                "abs_error": float(abs_error),
                "pct_error": float(pct_error),
                "mape": float(mape),
            }
        )

    if not rows:
        raise ValueError("No validation rows were produced.")

    return pd.DataFrame(rows)


def summarize_model_results(results_df: pd.DataFrame, model_name: str) -> pd.Series:
    """Summarize a model's validation metrics."""
    summary = {
        "model": model_name,
        "mae": float(results_df["abs_error"].mean()),
        "rmse": float(np.sqrt((results_df["abs_error"] ** 2).mean())),
        "mape": float(results_df["mape"].mean()),
        "mean_pct_error": float(results_df["pct_error"].mean()),
        "avg_forecast_vol": float(results_df["forecast_vol"].mean()),
        "avg_actual_vol": float(results_df["actual_vol"].mean()),
    }
    return pd.Series(summary)


def compare_volatility_models(
    ticker: str,
    start_date: str = "2016-01-01",
    horizons: Iterable[int] = (7, 14, 30, 40, 45),
    train_min_size: int = 252,
    step_size: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare candidate methods across the same horizon buckets used by the floor model.

    Returns:
        (results_by_model_and_horizon, summary_table, feature_frame)
    """
    prices = fetch_close_prices(ticker, start_date=start_date)
    horizons = list(horizons)

    model_methods = {
        "rolling_baseline": lambda train_returns, horizon: rolling_vol_predictor(train_returns, horizon, window=252),
        "ewma": lambda train_returns, horizon: ewma_vol_predictor(train_returns, horizon, span=20),
        "garch11": lambda train_returns, horizon: garch11_predictor(train_returns, horizon),
        "feature_enriched": lambda train_returns, horizon: feature_enriched_vol_predictor(train_returns, horizon, price_series=prices),
    }

    all_results: List[pd.DataFrame] = []
    summaries: List[pd.Series] = []

    for horizon_days in horizons:
        for model_name, method in model_methods.items():
            result_df = evaluate_method(
                prices,
                method=method,
                horizon_days=horizon_days,
                train_min_size=train_min_size,
                step_size=step_size,
            )
            result_df["model"] = model_name
            result_df["horizon_days"] = horizon_days
            all_results.append(result_df)

            summary = summarize_model_results(result_df, model_name)
            summary["horizon_days"] = horizon_days
            summaries.append(summary)

    combined = pd.concat(all_results, ignore_index=True)
    summary_table = pd.DataFrame(summaries).sort_values(["horizon_days", "mae"]).reset_index(drop=True)
    feature_frame = build_feature_frame(prices, horizon_days=max(horizons))
    return combined, summary_table, feature_frame


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison_report(summary_table: pd.DataFrame, feature_frame: pd.DataFrame) -> None:
    """Display a horizon-by-horizon comparison of method quality and model selection signals."""
    print("\nVolatility model comparison (walk-forward validation by horizon):")
    print(summary_table[["horizon_days", "model", "mae", "rmse", "mape", "mean_pct_error", "avg_forecast_vol", "avg_actual_vol"]].to_string(index=False))

    print("\nRecommended features for the next volatility model:")
    feature_names = list(feature_frame.columns[:-1])
    for item in feature_names[:12]:
        print(f"  - {item}")

    print("\nInterpretation:")
    print("  - Lower MAE/RMSE means better point forecast accuracy.")
    print("  - Lower MAPE means better relative accuracy across different vol regimes.")
    print("  - The model with the lowest MAE/MAPE for a given horizon is the best candidate for that horizon's MC input.")
    print("  - If a model does not meaningfully beat the rolling baseline, keep the historical baseline for that horizon.")

#%%z
# ---------------------------------------------------------------------------
# Example runner
# ---------------------------------------------------------------------------

def main() -> None:
    ticker = "AAPL"
    horizons = [7, 14, 30, 40, 45]
    results_df, summary_table, feature_frame = compare_volatility_models(
        ticker=ticker,
        start_date="2016-01-01",
        horizons=horizons,
        train_min_size=252,
        step_size=30,
    )

    print(f"Loaded data for {ticker} and ran walk-forward validation across horizons: {horizons}.")
    print_comparison_report(summary_table, feature_frame)

    print("\nSample validation rows:")
    print(results_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

#%%