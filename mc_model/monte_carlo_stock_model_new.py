# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import yfinance as yf
from datetime import date


# %%
# Import data
def inputs(ticket, start_date):
    today = date.today()
    data_df = yf.download(ticket, start=start_date, end=today)
    return data_df


# %%
# Data processing
def preprocessing_data(data_df):
    close_prices = data_df["Close"]

    if isinstance(close_prices, pd.DataFrame):
        if close_prices.shape[1] != 1:
            raise ValueError("Expected exactly one close-price column")
        close_prices = close_prices.iloc[:, 0]

    close_prices = close_prices.dropna().astype(float)
    close_prices.index = pd.to_datetime(close_prices.index)
    return close_prices


# %%
# Calculate log returns
def log_returns(df):
    return np.log(1 + df.pct_change())


# %%
# Calculate drift
def drift_calc(df):
    lr = log_returns(df)
    u = lr.mean()
    var = lr.var()
    drift = u - (0.5 * var)

    try:
        return drift.values
    except Exception:
        return drift


# %%
def daily_returns(prices, days, iterations, vol_window=None, rng=None):
    """Generate daily gross-return multipliers for Monte Carlo paths."""
    if days < 1 or iterations < 1:
        raise ValueError("days and iterations must be positive integers")

    if isinstance(prices, pd.DataFrame):
        if prices.shape[1] != 1:
            raise ValueError("Expected a Series or a one-column DataFrame")
        prices = prices.iloc[:, 0]
    else:
        prices = pd.Series(prices)

    prices = pd.to_numeric(prices, errors="coerce").dropna().astype(float)

    if (prices <= 0).any():
        raise ValueError("Prices must be positive")

    if vol_window is not None:
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        if len(prices) < vol_window + 1:
            raise ValueError(
                f"vol_window={vol_window} requires at least {vol_window + 1} prices; found {len(prices)}"
            )
        prices = prices.tail(vol_window + 1)

    historical_log_returns = np.log(prices).diff().dropna()

    if len(historical_log_returns) < 2:
        raise ValueError("At least two historical daily returns are required")

    mean_log_return = historical_log_returns.mean()
    volatility = historical_log_returns.std(ddof=1)

    if rng is None:
        rng = np.random.default_rng()

    shocks = rng.standard_normal(size=(days, iterations))
    return np.exp(mean_log_return + volatility * shocks)


# %%
def probs_find(predicted, higherthan, on="value"):
    if on == "return":
        predicted0 = predicted.iloc[0, 0]
        predicted = predicted.iloc[-1]
        pred_list = list(predicted)
        over = [
            (i * 100) / predicted0 for i in pred_list if ((i - predicted0) * 100) / predicted0 >= higherthan
        ]
        less = [
            (i * 100) / predicted0 for i in pred_list if ((i - predicted0) * 100) / predicted0 < higherthan
        ]
    elif on == "value":
        predicted = predicted.iloc[-1]
        pred_list = list(predicted)
        over = [i for i in pred_list if i >= higherthan]
        less = [i for i in pred_list if i < higherthan]
    else:
        raise ValueError("'on' must be either value or return")

    total = len(over) + len(less)
    if total == 0:
        return 0.0
    return len(over) / total


# %%
def simulate_mc(ticker, start_date, days, iterations, vol_window, plot=True):
    """Execute a Monte Carlo price simulation and display distribution plots."""
    data_df = inputs(ticker, start_date)
    final_data_df = preprocessing_data(data_df)
    returns = daily_returns(final_data_df, days, iterations, vol_window)

    price_list = np.zeros_like(returns)
    price_list[0] = final_data_df.iloc[-1]

    for t in range(1, days):
        price_list[t] = price_list[t - 1] * returns[t]

    if plot:
        x = pd.DataFrame(price_list).iloc[-1]
        fig, ax = plt.subplots(1, 2, figsize=(14, 4))
        sns.distplot(x, ax=ax[0])
        sns.distplot(x, hist_kws={"cumulative": True}, kde_kws={"cumulative": True}, ax=ax[1])
        ax[0].set_xlabel("Stock Price")
        ax[1].set_xlabel("Cumulative Distribution")
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(final_data_df.index[-20:], final_data_df.values[-20:], label="Historical price")
    price_list_index = range(len(final_data_df), len(final_data_df) + len(price_list))
    plt.plot(price_list_index, price_list, label="Monte Carlo path")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    print(f"Days: {days}")
    print(f"Expected Value: ${round(pd.DataFrame(price_list).iloc[-1].mean(), 2)}")
    print(f"Probability of Breakeven: {probs_find(pd.DataFrame(price_list), 0, on='return')}")
    print(f"Latest Price: {final_data_df.iloc[-1]}")
    print(f"Quantile (5%): {np.percentile(price_list, 5)}")
    print(f"Quantile (10%): {np.percentile(price_list, 10)}")
    print(f"Quantile (50%): {np.percentile(price_list, 50)}")
    print(f"Quantile (95%): {np.percentile(price_list, 95)}")

    return pd.DataFrame(price_list)


# %%
def backtest_once(price_series, horizon, iterations, vol_window=None):
    """Compare a single forecast origin against the realized outcome."""
    split = -horizon
    train = price_series.iloc[:split]
    actual = float(price_series.iloc[split + horizon - 1])
    sim = daily_returns(train, horizon, iterations, vol_window=vol_window)
    last_price = float(train.iloc[-1])
    end_prices = last_price * np.prod(sim, axis=0)

    mean_pred = end_prices.mean()
    rmse = np.sqrt(np.mean((end_prices - actual) ** 2))
    mape = np.mean(np.abs((end_prices - actual) / actual)) * 100
    lower, upper = np.percentile(end_prices, 5), np.percentile(end_prices, 95)
    coverage = 1.0 if (actual >= lower and actual <= upper) else 0.0

    return {
        "mean_pred": mean_pred,
        "rmse": rmse,
        "mape": mape,
        "coverage_5_95": coverage,
    }


# %%
def backtest_rolling(
    price_series,
    horizon,
    iterations=2000,
    n_origins=1000,
    min_train=253,
    vol_window=None,
    origin_step=None,
    target_start=None,
    target_end=None,
    seed=42,
):
    """Backtest terminal-price percentile floors across historical origins."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if iterations < 1 or n_origins < 1 or min_train < 2:
        raise ValueError("iterations, n_origins, and min_train must be positive")

    if origin_step is None:
        origin_step = horizon
    if origin_step < 1:
        raise ValueError("origin_step must be at least 1")

    if isinstance(price_series, pd.DataFrame):
        if price_series.shape[1] != 1:
            raise ValueError("Expected a Series or a one-column DataFrame")
        prices = price_series.iloc[:, 0]
    else:
        prices = pd.Series(price_series)

    prices = pd.to_numeric(prices, errors="coerce").dropna().astype(float)
    dates = pd.DatetimeIndex(pd.to_datetime(prices.index))

    if (prices <= 0).any():
        raise ValueError("Prices must be positive")

    if vol_window is not None:
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2")
        min_train = max(min_train, vol_window + 1)

    first_origin = min_train - 1
    last_origin = len(prices) - horizon - 1

    if last_origin < first_origin:
        raise ValueError(f"Not enough prices for horizon={horizon} and min_train={min_train}")

    eligible_origins = list(range(first_origin, last_origin + 1, origin_step))

    if target_start is not None:
        target_start = pd.Timestamp(target_start)
        eligible_origins = [
            origin for origin in eligible_origins if dates[origin + horizon] >= target_start
        ]

    if target_end is not None:
        target_end = pd.Timestamp(target_end)
        eligible_origins = [
            origin for origin in eligible_origins if dates[origin + horizon] <= target_end
        ]

    origins = eligible_origins[-n_origins:]
    if not origins:
        raise ValueError("No forecast origins in the requested target-date range")

    rng = np.random.default_rng(seed)
    rows = []

    for origin in origins:
        train = prices.iloc[:origin + 1]
        current_price = float(train.iloc[-1])
        actual = float(prices.iloc[origin + horizon])
        simulated_returns = daily_returns(
            train,
            days=horizon,
            iterations=iterations,
            vol_window=vol_window,
            rng=rng,
        )
        terminal_prices = current_price * np.prod(simulated_returns, axis=0)

        lower_5, lower_10, upper_95 = np.percentile(terminal_prices, [5, 10, 95])
        above_lower_5 = actual >= lower_5
        above_lower_10 = actual >= lower_10
        shortfall = max(lower_10 - actual, 0.0)
        predicted_mean = float(terminal_prices.mean())

        rows.append({
            "origin": dates[origin],
            "target": dates[origin + horizon],
            "current_price": current_price,
            "actual": actual,
            "predicted_mean": predicted_mean,
            "lower_5": lower_5,
            "lower_10": lower_10,
            "upper_95": upper_95,
            "actual_above_lower_5": above_lower_5,
            "actual_above_lower_10": above_lower_10,
            "covered_5_95": lower_5 <= actual <= upper_95,
            "lower_10_return_pct": (lower_10 / current_price - 1) * 100,
            "shortfall_below_lower_10": shortfall,
            "absolute_error": abs(predicted_mean - actual),
        })

    results = pd.DataFrame(rows)
    misses = results.loc[~results["actual_above_lower_10"], "shortfall_below_lower_10"]

    summary = {
        "horizon": horizon,
        "vol_window": vol_window,
        "origin_step": origin_step,
        "forecasts": len(results),
        "actual_above_lower_5_rate": results["actual_above_lower_5"].mean(),
        "actual_above_lower_10_rate": results["actual_above_lower_10"].mean(),
        "mean_lower_10_price": results["lower_10"].mean(),
        "mean_lower_10_return_pct": results["lower_10_return_pct"].mean(),
        "mean_shortfall_below_lower_10": results["shortfall_below_lower_10"].mean(),
        "mean_shortfall_when_breached": misses.mean() if not misses.empty else np.nan,
        "coverage_5_95": results["covered_5_95"].mean(),
        "mean_absolute_error": results["absolute_error"].mean(),
    }

    return results, summary


# %%
def build_decision_reports(price_series, selected_windows, iterations=2000, seed=42):
    """Create a single current forecast report for the chosen volatility settings."""
    current_date = pd.Timestamp(price_series.index.max())
    rows = []

    for horizon, vol_window in selected_windows.items():
        history = price_series if vol_window is None else price_series.tail(vol_window + 1)
        current_price = float(history.iloc[-1])
        rng = np.random.default_rng(seed + horizon)
        simulated_returns = daily_returns(history, days=horizon, iterations=iterations, rng=rng)
        terminal_prices = current_price * np.prod(simulated_returns, axis=0)

        lower_5, lower_10, upper_95 = np.percentile(terminal_prices, [5, 10, 95])
        rows.append({
            "date": current_date,
            "horizon": horizon,
            "vol_window": vol_window,
            "current_price": current_price,
            "mean_terminal_price": float(terminal_prices.mean()),
            "lower_5": lower_5,
            "lower_10": lower_10,
            "upper_95": upper_95,
            "lower_10_return_pct": (lower_10 / current_price - 1) * 100,
        })

    return pd.DataFrame(rows)


# %%
def make_live_forecasts(price_series, selected_windows, forecast_date=None, iterations=2000, seed=42):
    """Record a single forecast ledger for the next horizon windows."""
    if forecast_date is None:
        forecast_date = price_series.index.max()

    forecast_date = pd.Timestamp(forecast_date)
    rows = []

    for horizon, vol_window in selected_windows.items():
        origin_index = price_series.index.get_loc(forecast_date)
        train = price_series.iloc[:origin_index + 1]
        current_price = float(train.iloc[-1])
        history = train if vol_window is None else train.tail(vol_window + 1)
        rng = np.random.default_rng(seed + horizon)
        simulated_returns = daily_returns(history, days=horizon, iterations=iterations, rng=rng)
        terminal_prices = current_price * np.prod(simulated_returns, axis=0)
        target_date = train.index[-1] + pd.Timedelta(days=horizon)

        rows.append({
            "origin": forecast_date,
            "target": target_date,
            "horizon": horizon,
            "vol_window": vol_window,
            "current_price": current_price,
            "lower_10": np.percentile(terminal_prices, 10),
            "lower_5": np.percentile(terminal_prices, 5),
            "mean_terminal_price": float(terminal_prices.mean()),
        })

    return pd.DataFrame(rows)


# %%
def resolve_matured_forecasts(forecast_ledger, price_series):
    """Resolve each forecast once its target date has arrived."""
    if forecast_ledger.empty:
        return forecast_ledger.copy()

    resolved = forecast_ledger.copy()
    resolved["actual_price"] = resolved["target"].map(
        lambda target: float(price_series.loc[target]) if target in price_series.index else np.nan
    )
    resolved["actual_above_lower_10"] = resolved["actual_price"].notna() & (
        resolved["actual_price"] >= resolved["lower_10"]
    )
    return resolved


# %%
def main():
    ticker = "AAPL"
    start_date = "2016-10-31"

    print("Loading data for", ticker)
    data = inputs(ticker, start_date)
    prices = preprocessing_data(data)

    horizons = [7, 14, 30, 40, 45]
    vol_windows = [20, 60, 252, None]

    selection_rows = []
    for horizon in horizons:
        for window in vol_windows:
            _, summary = backtest_rolling(
                prices,
                horizon=horizon,
                iterations=2000,
                n_origins=1000,
                min_train=253,
                vol_window=window,
                origin_step=horizon,
                target_start="2018-01-01",
                target_end="2019-12-31",
                seed=42,
            )
            selection_rows.append(summary)

    selection_table = pd.DataFrame(selection_rows)
    print("\nSelection comparison (2018-2019):")
    print(
        selection_table.sort_values(
            ["horizon", "actual_above_lower_10_rate"],
            ascending=[True, False],
        ).to_string(index=False)
    )

    selected_windows = {
        7: 20,
        14: 252,
        30: 252,
        40: 252,
        45: None,
    }

    test_periods = {
        "2020-2021": ("2020-01-01", "2021-12-31"),
        "2022-2026": ("2022-01-01", "2026-05-30"),
    }

    test_rows = []
    for period_name, (start_date, end_date) in test_periods.items():
        for horizon, window in selected_windows.items():
            _, summary = backtest_rolling(
                prices,
                horizon=horizon,
                iterations=2000,
                n_origins=1000,
                min_train=253,
                vol_window=window,
                origin_step=horizon,
                target_start=start_date,
                target_end=end_date,
                seed=42,
            )
            summary["test_period"] = period_name
            test_rows.append(summary)

    test_table = pd.DataFrame(test_rows)
    columns_to_show = [
        "test_period",
        "horizon",
        "vol_window",
        "forecasts",
        "actual_above_lower_10_rate",
        "mean_lower_10_return_pct",
        "mean_shortfall_when_breached",
        "actual_above_lower_5_rate",
    ]

    print("\nValidation results (frozen windows):")
    print(
        test_table[columns_to_show]
        .sort_values(["test_period", "horizon"])
        .to_string(index=False)
    )

    decision_report = build_decision_reports(
        prices,
        selected_windows=selected_windows,
        iterations=2000,
        seed=42,
    )
    print("\nCurrent decision report:")
    print(decision_report.to_string(index=False))

    forecast_ledger = make_live_forecasts(
        prices,
        selected_windows=selected_windows,
        forecast_date=prices.index.max(),
        iterations=2000,
        seed=42,
    )
    print("\nForecast ledger:")
    print(forecast_ledger.to_string(index=False))

    print("\nLatest available price date:", prices.index.max())

    current_detail, current_summary = backtest_rolling(
        prices,
        horizon=30,
        iterations=2000,
        n_origins=1000,
        min_train=253,
        vol_window=selected_windows[30],
        origin_step=30,
        target_start="2026-06-01",
        target_end=prices.index.max(),
        seed=42,
    )
    print("\nLatest in-sample backtest (30-day horizon):")
    print(pd.Series(current_summary).to_string())
    print(current_detail.tail(10).to_string(index=False))

    print("\nMain execution complete.")


if __name__ == "__main__":
    main()
