"""Compare the Monte Carlo floor under different volatility inputs.

This file is intentionally separate from the main floor model workflow so we can
compare the impact of using different volatility assumptions across the same
horizons and backtest windows.

Methods compared:
- historical_vol
- rolling_baseline
- ewma
- garch11
- feature_enriched

The script keeps the same generic Monte Carlo logic but swaps in the volatility
assumption. This lets us measure whether a forecasted volatility model improves
the downside floor estimate compared to the historical-vol baseline.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

from volatility_method_comparison import (
    compare_volatility_models,
    ewma_vol_predictor,
    feature_enriched_vol_predictor,
    fetch_close_prices as compare_fetch_close_prices,
    garch11_predictor,
    log_returns as compare_log_returns,
    rolling_vol_predictor,
)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def fetch_close_prices(ticker: str, start_date: str = "2016-01-01", end_date: str | None = None) -> pd.Series:
    if end_date is None:
        end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False, progress=False)
    if data.empty:
        raise ValueError(f"No price data found for {ticker}")
    close = data["Close"].dropna().astype(float)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.name = "close"
    return close


def log_returns(prices: pd.Series) -> pd.Series:
    prices = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    if (prices <= 0).any():
        raise ValueError("Prices must be positive")
    return np.log(prices / prices.shift(1)).dropna()


def annualized_volatility_from_returns(returns: pd.Series, window: int = 20) -> float:
    return float(returns.tail(window).std(ddof=1) * np.sqrt(252))


def ewma_volatility(returns: pd.Series, span: int = 20) -> float:
    variance = returns.pow(2).ewm(span=span, adjust=False).mean().iloc[-1]
    return float(np.sqrt(variance * 252))


def historical_volatility(returns: pd.Series, window: int = 252) -> float:
    return annualized_volatility_from_returns(returns, window=window)


def build_mc_paths(
    current_price: float,
    annualized_vol: float,
    horizon_days: int,
    iterations: int = 5000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate terminal prices under geometric Brownian motion."""
    if rng is None:
        rng = np.random.default_rng(42)

    daily_vol = annualized_vol / np.sqrt(252)
    drift = -0.5 * daily_vol ** 2
    shocks = rng.normal(0.0, daily_vol, size=(horizon_days, iterations))
    log_paths = np.log(current_price) + drift * np.arange(1, horizon_days + 1)[:, None] + shocks
    return np.exp(log_paths[-1, :])


# ---------------------------------------------------------------------------
# Candidate volatility methods for comparison
# ---------------------------------------------------------------------------

def vol_method_historical(returns: pd.Series, horizon_days: int) -> float:
    return historical_volatility(returns, window=252)


def vol_method_rolling(returns: pd.Series, horizon_days: int) -> float:
    return annualized_volatility_from_returns(returns, window=252)


def vol_method_ewma(returns: pd.Series, horizon_days: int) -> float:
    return ewma_volatility(returns, span=20)


def vol_method_garch11(returns: pd.Series, horizon_days: int) -> float:
    # Lightweight approximation using a 20-day EWMA as a first pass for comparison.
    # The main goal is side-by-side comparison, not a full GARCH production model.
    return ewma_volatility(returns, span=20)


def vol_method_feature_enriched(returns: pd.Series, horizon_days: int) -> float:
    recent = returns.tail(60)
    rolling_20 = annualized_volatility_from_returns(returns, window=20)
    rolling_40 = annualized_volatility_from_returns(returns, window=40)
    rolling_60 = annualized_volatility_from_returns(returns, window=60)
    ewma = ewma_volatility(returns, span=20)
    recency = recent.abs().mean() * np.sqrt(252)

    blended = 0.35 * rolling_20 + 0.25 * rolling_40 + 0.20 * rolling_60 + 0.20 * ewma
    return float(max(blended * 0.9 + recency * 0.1, 0.02))


# ---------------------------------------------------------------------------
# MC comparison runner
# ---------------------------------------------------------------------------

def compare_mc_floor_methods(
    ticker: str,
    start_date: str,
    horizons: Iterable[int] = (7, 14, 30, 40, 45),
    iterations: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the same Monte Carlo floor logic using different volatility assumptions and return a comparison table."""
    prices = fetch_close_prices(ticker, start_date=start_date)
    returns = log_returns(prices)
    current_price = float(prices.iloc[-1])

    methods = {
        "historical_vol": vol_method_historical,
        "rolling_baseline": vol_method_rolling,
        "ewma": vol_method_ewma,
        "garch11": vol_method_garch11,
        "feature_enriched": vol_method_feature_enriched,
    }

    rows: List[Dict[str, float]] = []

    for horizon in list(horizons):
        for method_name, method_fn in methods.items():
            vol = method_fn(returns, horizon)
            terminal_prices = build_mc_paths(
                current_price=current_price,
                annualized_vol=vol,
                horizon_days=horizon,
                iterations=iterations,
                rng=np.random.default_rng(seed + horizon + hash(method_name) % 1000),
            )

            lower_5 = float(np.percentile(terminal_prices, 5))
            lower_10 = float(np.percentile(terminal_prices, 10))
            lower_25 = float(np.percentile(terminal_prices, 25))
            mean_terminal = float(np.mean(terminal_prices))

            rows.append(
                {
                    "ticker": ticker,
                    "horizon_days": horizon,
                    "method": method_name,
                    "annualized_vol": vol,
                    "current_price": current_price,
                    "lower_5": lower_5,
                    "lower_10": lower_10,
                    "lower_25": lower_25,
                    "mean_terminal_price": mean_terminal,
                    "lower_10_return_pct": (lower_10 / current_price - 1.0) * 100.0,
                }
            )

    comparison = pd.DataFrame(rows)
    return comparison.sort_values(["horizon_days", "method"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Combined forecast-selection + MC runner
# ---------------------------------------------------------------------------

MODEL_METHOD_MAP: Dict[str, Callable[[pd.Series, int, pd.Series | None], float]] = {
    "rolling_baseline": lambda returns, horizon, price_series=None: rolling_vol_predictor(returns, horizon, window=252),
    "ewma": lambda returns, horizon, price_series=None: ewma_vol_predictor(returns, horizon, span=20),
    "garch11": lambda returns, horizon, price_series=None: garch11_predictor(returns, horizon),
    "feature_enriched": lambda returns, horizon, price_series=None: feature_enriched_vol_predictor(returns, horizon, price_series=price_series),
}


def select_best_forecast_model_by_horizon(
    ticker: str,
    start_date: str,
    horizons: Iterable[int] = (7, 14, 30, 40, 45),
    train_min_size: int = 252,
    step_size: int = 30,
    show_model_metrics: bool = False,
) -> Tuple[Dict[int, str], pd.DataFrame]:
    """Rank volatility forecast models by walk-forward validation and pick the lowest-MAE winner for each horizon."""
    _, summary_table, _ = compare_volatility_models(
        ticker=ticker,
        start_date=start_date,
        horizons=horizons,
        train_min_size=train_min_size,
        step_size=step_size,
    )

    if show_model_metrics:
        print(f"\nForecast model ranking for {ticker} by horizon:")
        print(
            summary_table[["horizon_days", "model", "mae", "rmse", "mape", "mean_pct_error"]]
            .sort_values(["horizon_days", "mae"])
            .to_string(index=False)
        )

    winners: Dict[int, str] = {}
    for horizon in list(horizons):
        candidates = summary_table[summary_table["horizon_days"] == horizon].copy()
        if candidates.empty:
            raise ValueError(f"No model metrics available for horizon={horizon} for {ticker}")
        winner = candidates.sort_values(["mae", "mape", "rmse"], ascending=[True, True, True]).iloc[0]
        winners[int(horizon)] = str(winner["model"])

    return winners, summary_table


def forecast_volatility_for_model(
    prices: pd.Series,
    returns: pd.Series,
    horizon_days: int,
    model_name: str,
) -> float:
    """Estimate the forecasted annualized volatility implied by the selected model."""
    if model_name == "historical_vol":
        return historical_volatility(returns, window=252)
    if model_name not in MODEL_METHOD_MAP:
        raise ValueError(f"Unsupported model_name={model_name}")
    return float(MODEL_METHOD_MAP[model_name](returns, horizon_days, price_series=prices))


def run_selected_forecast_vs_historical(
    tickers: Iterable[str],
    start_date: str,
    horizons: Iterable[int] = (7, 14, 30, 40, 45),
    iterations: int = 5000,
    seed: int = 42,
    show_model_metrics: bool = False,
) -> pd.DataFrame:
    """Select the best future-looking volatility model by horizon for each ticker, compare it to historical vol in the MC floor, and return the final table."""
    rows: List[Dict[str, float | str]] = []
    horizons = list(horizons)

    for ticker in list(tickers):
        prices = compare_fetch_close_prices(ticker, start_date=start_date)
        returns = compare_log_returns(prices)
        current_price = float(prices.iloc[-1])

        winners, summary_table = select_best_forecast_model_by_horizon(
            ticker=ticker,
            start_date=start_date,
            horizons=horizons,
            show_model_metrics=show_model_metrics,
        )

        for horizon in horizons:
            selected_model = winners[int(horizon)]
            selected_vol = forecast_volatility_for_model(prices, returns, horizon, selected_model)
            historical_vol = historical_volatility(returns, window=252)

            selected_terminal_prices = build_mc_paths(
                current_price=current_price,
                annualized_vol=selected_vol,
                horizon_days=horizon,
                iterations=iterations,
                rng=np.random.default_rng(seed + horizon + 1),
            )
            historical_terminal_prices = build_mc_paths(
                current_price=current_price,
                annualized_vol=historical_vol,
                horizon_days=horizon,
                iterations=iterations,
                rng=np.random.default_rng(seed + horizon + 2),
            )

            selected_floor_10 = float(np.percentile(selected_terminal_prices, 10))
            historical_floor_10 = float(np.percentile(historical_terminal_prices, 10))
            selected_return_pct = (selected_floor_10 / current_price - 1.0) * 100.0
            historical_return_pct = (historical_floor_10 / current_price - 1.0) * 100.0

            best_model_row = summary_table[
                (summary_table["horizon_days"] == horizon)
                & (summary_table["model"] == selected_model)
            ].sort_values(["mae", "mape", "rmse"]).iloc[0]

            rows.append(
                {
                    "ticker": ticker,
                    "horizon_days": horizon,
                    "selected_model": selected_model,
                    "selected_model_mae": float(best_model_row["mae"]),
                    "selected_model_mape": float(best_model_row["mape"]),
                    "selected_model_rmse": float(best_model_row["rmse"]),
                    "selected_annualized_vol": selected_vol,
                    "historical_annualized_vol": historical_vol,
                    "selected_lower_10": selected_floor_10,
                    "historical_lower_10": historical_floor_10,
                    "selected_lower_10_return_pct": selected_return_pct,
                    "historical_lower_10_return_pct": historical_return_pct,
                    "current_price": current_price,
                }
            )

    comparison = pd.DataFrame(rows)
    return comparison.sort_values(["ticker", "horizon_days"]).reset_index(drop=True)


def print_selected_forecast_report(df: pd.DataFrame) -> None:
    """Print the final comparison between selected future-looking volatility forecasts and the historical-vol baseline."""
    print("\nSelected forecast-vs-historical MC comparison:")
    print(
        df[[
            "ticker",
            "horizon_days",
            "selected_model",
            "selected_model_mae",
            "selected_model_mape",
            "selected_annualized_vol",
            "historical_annualized_vol",
            "selected_lower_10",
            "historical_lower_10",
            "selected_lower_10_return_pct",
            "historical_lower_10_return_pct",
        ]].to_string(index=False)
    )

#%%
# ---------------------------------------------------------------------------
# Example runner
# ---------------------------------------------------------------------------

def main() -> None:
    tickers = ["AAPL", "AMD", "AMZN", "GOOG","LLY", "NVDA", "SPY", "UPRO"]
    result = run_selected_forecast_vs_historical(
        tickers=tickers,
        start_date="2016-01-01",
        horizons=(7, 14, 30, 40, 45),
        iterations=5000,
        seed=42,
        show_model_metrics=True,
    )
    print_selected_forecast_report(result)


if __name__ == "__main__":
    main()


#%%