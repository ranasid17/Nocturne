"""Download monthly market and economic data and fit a SPY regression model."""

import argparse
import sys
from io import StringIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from joblib import load
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA


FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"
FEATURE_LAGS = {
    "POILWTIUSDM": 1,
    "CPIAUCSL": 2,
    "PPIACO": 2,
    "FEDFUNDS": 1,
    "UNRATE": 2,
}
LAGGED_FEATURE_COLUMNS = {
    column: f"{column}_LAG{lag}" for column, lag in FEATURE_LAGS.items()
}
FEATURE_COLUMNS = [
    "POILWTIUSDM_LAG1",
    "CPIAUCSL_LAG2",
    "PPIACO_LAG2",
    "FEDFUNDS_LAG1",
    "UNRATE_LAG2",
]
SPY_TECHNICAL_FEATURE_COLUMNS = [
    "SPY_RETURN_1M_LAG1",
    "SPY_RETURN_3M_LAG1",
    "SPY_RETURN_6M_LAG1",
    "SPY_RETURN_12M_LAG1",
    "SPY_VOLATILITY_3M_LAG1",
    "SPY_VOLATILITY_6M_LAG1",
    "SPY_SMA200_GAP_PCT_LAG1",
]
RETURN_FEATURE_COLUMNS = SPY_TECHNICAL_FEATURE_COLUMNS
MACRO_RETURN_FEATURE_COLUMNS = [
    "OIL_MONTHLY_CHANGE_PCT_LAG1",
    "CPI_YOY_PCT_LAG2",
    "PPI_YOY_PCT_LAG2",
    "FEDFUNDS_RATE_LAG1",
    "UNEMPLOYMENT_RATE_LAG2",
]
COMBINED_RETURN_FEATURE_COLUMNS = (
    RETURN_FEATURE_COLUMNS + MACRO_RETURN_FEATURE_COLUMNS
)
CURRENT_SPY_FEATURE_COLUMNS = {
    feature: feature.removesuffix("_LAG1")
    for feature in SPY_TECHNICAL_FEATURE_COLUMNS
}
MONTHLY_FORECAST_LEDGER = (
    Path(__file__).resolve().parents[2] / "spy_monthly_forecast_ledger.csv"
)
MONTHLY_MODEL_VERSION = "technical_ridge_common_v2"
FRED_SERIES = {
    "CPIAUCSL": "CPIAUCSL",
    "FEDFUNDS": "FEDFUNDS",
    "DCOILWTICO": "POILWTIUSDM",
    "PPIACO": "PPIACO",
    "UNRATE": "UNRATE",
}


def _fred_series(series_id, start_date, end_date=None):
    """Fetch one FRED series as a numeric DATE/value DataFrame."""
    params = {"id": series_id, "cosd": start_date}
    if end_date is not None:
        params["coed"] = end_date

    response = requests.get(FRED_CSV_URL, params=params, timeout=30)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text), na_values=["."])

    date_column = "observation_date"
    if date_column not in data.columns or series_id not in data.columns:
        raise ValueError(f"Unexpected FRED response for series {series_id!r}")

    data = data.rename(columns={date_column: "DATE", series_id: "value"})
    data["DATE"] = pd.to_datetime(data["DATE"], errors="coerce")
    data["value"] = pd.to_numeric(data["value"], errors="coerce")
    data = data.dropna(subset=["DATE", "value"])
    data["DATE"] = data["DATE"].dt.to_period("M").dt.to_timestamp()
    return data[["DATE", "value"]]


def _download_spy_adjusted_close(start_date, end_date=None, ticker="SPY"):
    """Download daily adjusted closes, including history for lookback features."""
    download_start = (
        pd.Timestamp(start_date) - pd.Timedelta(days=400)
    ).strftime("%Y-%m-%d")
    data = yf.download(
        ticker,
        start=download_start,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        actions=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No Yahoo Finance data returned for {ticker!r}")

    if isinstance(data.columns, pd.MultiIndex):
        adjusted_close = next(
            (
                column
                for column in data.columns
                if "Adj Close" in column
            ),
            None,
        )
        if adjusted_close is None:
            raise ValueError(f"Adjusted close is missing for {ticker!r}")
        prices = data.loc[:, adjusted_close]
    else:
        if "Adj Close" not in data.columns:
            raise ValueError(f"Adjusted close is missing for {ticker!r}")
        prices = data["Adj Close"]

    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.dropna().sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    return prices.rename("Adj Close")


def _spy_monthly_data(start_date, end_date=None, ticker="SPY"):
    """Build monthly SPY prices and lagged market features from daily data."""
    prices = _download_spy_adjusted_close(
        start_date,
        end_date=end_date,
        ticker=ticker,
    )
    monthly_prices = prices.resample("MS").mean().dropna()
    monthly_returns = monthly_prices.pct_change(fill_method=None)
    monthly = pd.DataFrame(index=monthly_prices.index)
    monthly["Adj Close"] = monthly_prices
    monthly["Previous Avg SPY"] = monthly_prices.shift(1)
    monthly["Monthly Return (%)"] = monthly_returns * 100

    for window in (1, 3, 6, 12):
        monthly[f"SPY_RETURN_{window}M"] = (
            monthly_prices.pct_change(window, fill_method=None) * 100
        )
        monthly[f"SPY_RETURN_{window}M_LAG1"] = (
            monthly_prices.pct_change(window, fill_method=None).shift(1) * 100
        )

    for window in (3, 6):
        monthly[f"SPY_VOLATILITY_{window}M"] = (
            monthly_returns.rolling(window).std() * np.sqrt(12) * 100
        )
        monthly[f"SPY_VOLATILITY_{window}M_LAG1"] = (
            monthly_returns.rolling(window).std().shift(1) * np.sqrt(12) * 100
        )

    daily_sma = prices.rolling(200, min_periods=200).mean()
    daily_sma_gap = (prices / daily_sma - 1) * 100
    monthly["SPY_SMA200_GAP_PCT"] = daily_sma_gap.resample("MS").last()
    monthly["SPY_SMA200_GAP_PCT_LAG1"] = (
        daily_sma_gap.resample("MS").last().shift(1)
    )

    start_month = pd.Timestamp(start_date).to_period("M").to_timestamp()
    monthly = monthly.loc[monthly.index >= start_month]
    return monthly.rename_axis("DATE").reset_index()


def inputs(start_date="1993-02-01", end_date=None, ticker="SPY"):
    """Return monthly SPY, CPI, Fed funds, oil, PPI, and unemployment DataFrames.

    SPY uses the average adjusted close of each month and includes market
    features available before that month. WTI oil is averaged from daily FRED
    observations; the remaining FRED series are already monthly.
    """
    spy_df = end_date_for_spy_df(
        _spy_monthly_data(start_date, end_date=end_date, ticker=ticker)
    )

    fred_data = {}
    for series_id, column_name in FRED_SERIES.items():
        series_df = _fred_series(series_id, start_date, end_date=end_date)
        if series_id == "DCOILWTICO":
            series_df = (
                series_df.set_index("DATE")["value"]
                .resample("MS")
                .mean()
                .rename("value")
                .reset_index()
            )
        fred_data[column_name] = series_df.rename(
            columns={"value": column_name}
        )

    return (
        spy_df,
        fred_data["CPIAUCSL"],
        fred_data["FEDFUNDS"],
        fred_data["POILWTIUSDM"],
        fred_data["PPIACO"],
        fred_data["UNRATE"],
    )


def model_load(filename):
    """Load a previously saved scikit-learn model from a trusted file."""
    return load(Path(filename))


def start_date_for_each_df_besides_spy(
    oil_df, cpi_df, ppi_df, unr_df, fed_df, start_date="1993-02-01"
):
    """Trim the macroeconomic DataFrames to a common starting date."""
    start = pd.Timestamp(start_date)
    frames = []
    for frame in (oil_df, cpi_df, ppi_df, unr_df, fed_df):
        normalized = frame.copy()
        normalized["DATE"] = pd.to_datetime(normalized["DATE"])
        frames.append(normalized.loc[normalized["DATE"] >= start].copy())
    return tuple(frames)


def end_date_for_spy_df(input_df, as_of=None):
    """Exclude the current, potentially incomplete, month from SPY data."""
    date_column = "DATE" if "DATE" in input_df.columns else "Date"
    if date_column not in input_df.columns:
        raise ValueError("SPY data must contain a DATE or Date column")

    spy_df = input_df.copy()
    spy_df["DATE"] = pd.to_datetime(spy_df[date_column])
    spy_df["DATE"] = spy_df["DATE"].dt.to_period("M").dt.to_timestamp()
    current_month = pd.Timestamp(as_of or pd.Timestamp.today())
    current_month = current_month.to_period("M").to_timestamp()
    return spy_df.loc[spy_df["DATE"] < current_month].drop(
        columns=[date_column] if date_column != "DATE" else []
    ).reset_index(drop=True)


def merge_all_dfs_for_predictions(oil_df, cpi_df, ppi_df, unr_df, fed_df, spy_df):
    """Join monthly indicators onto the SPY calendar without dropping gaps."""
    frames = [oil_df, cpi_df, ppi_df, fed_df, unr_df, spy_df]
    normalized_frames = []
    for frame in frames:
        normalized = frame.copy()
        date_column = "DATE" if "DATE" in normalized.columns else "Date"
        if date_column not in normalized.columns:
            raise ValueError("Each input DataFrame must contain DATE or Date")
        normalized["DATE"] = (
            pd.to_datetime(normalized[date_column])
            .dt.to_period("M")
            .dt.to_timestamp()
        )
        if date_column != "DATE":
            normalized = normalized.drop(columns=[date_column])
        normalized_frames.append(normalized)

    merged_df = normalized_frames[-1]
    for frame in normalized_frames[:-1]:
        merged_df = merged_df.merge(frame, on="DATE", how="left")

    return merged_df.sort_values("DATE").reset_index(drop=True)


def preprocessing_final(df):
    """Create release-aware macro features and the monthly price change."""
    required_columns = ["DATE", "Adj Close", *LAGGED_FEATURE_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Missing required preprocessing columns: {missing_columns}")

    processed_df = df.copy()
    processed_df["DATE"] = (
        pd.to_datetime(processed_df["DATE"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    processed_df = processed_df.sort_values("DATE").reset_index(drop=True)

    lagged_frames = []
    for source_column, lag_months in FEATURE_LAGS.items():
        lagged_frame = processed_df[["DATE", source_column]].copy()
        lagged_frame["DATE"] += pd.DateOffset(months=lag_months)
        lagged_frames.append(
            lagged_frame.rename(
                columns={source_column: LAGGED_FEATURE_COLUMNS[source_column]}
            )
        )

    lagged_df = lagged_frames[0]
    for frame in lagged_frames[1:]:
        lagged_df = lagged_df.merge(frame, on="DATE", how="outer")

    processed_df = processed_df.merge(
        lagged_df,
        on="DATE",
        how="left",
        validate="one_to_one",
    )
    processed_df["stock_price_diff"] = processed_df["Adj Close"].diff()
    return processed_df


def _add_macro_return_features(df):
    """Add transformed, release-lagged macro features for return models."""
    required_columns = [
        "DATE",
        "POILWTIUSDM",
        "CPIAUCSL",
        "PPIACO",
        "FEDFUNDS",
        "UNRATE",
    ]
    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Missing macro feature columns: {missing_columns}")

    processed = df.copy()
    processed["DATE"] = (
        pd.to_datetime(processed["DATE"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    processed = processed.sort_values("DATE").reset_index(drop=True)
    oil_levels = pd.to_numeric(
        processed["POILWTIUSDM"],
        errors="coerce",
    )
    cpi_levels = pd.to_numeric(
        processed["CPIAUCSL"],
        errors="coerce",
    )
    ppi_levels = pd.to_numeric(
        processed["PPIACO"],
        errors="coerce",
    )
    transformed = {
        "OIL_MONTHLY_CHANGE_PCT": (
            oil_levels.pct_change(fill_method=None) * 100
        ).ffill(),
        "CPI_YOY_PCT": (
            cpi_levels.pct_change(12, fill_method=None) * 100
        ).ffill(),
        "PPI_YOY_PCT": (
            ppi_levels.pct_change(12, fill_method=None) * 100
        ).ffill(),
        "FEDFUNDS_RATE": pd.to_numeric(
            processed["FEDFUNDS"],
            errors="coerce",
        ).ffill(),
        "UNEMPLOYMENT_RATE": pd.to_numeric(
            processed["UNRATE"],
            errors="coerce",
        ).ffill(),
    }
    feature_specs = [
        ("OIL_MONTHLY_CHANGE_PCT_LAG1", "OIL_MONTHLY_CHANGE_PCT", 1),
        ("CPI_YOY_PCT_LAG2", "CPI_YOY_PCT", 2),
        ("PPI_YOY_PCT_LAG2", "PPI_YOY_PCT", 2),
        ("FEDFUNDS_RATE_LAG1", "FEDFUNDS_RATE", 1),
        ("UNEMPLOYMENT_RATE_LAG2", "UNEMPLOYMENT_RATE", 2),
    ]

    for feature_column, source_name, lag_months in feature_specs:
        lagged = pd.DataFrame(
            {
                "DATE": processed["DATE"] + pd.DateOffset(months=lag_months),
                feature_column: transformed[source_name],
            }
        )
        processed = processed.merge(
            lagged,
            on="DATE",
            how="left",
            validate="one_to_one",
        )
    return processed


def set_up_train_test_split(df, test_size=0.2):
    """Split observations chronologically so test data follows training data."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    missing_columns = [
        column for column in FEATURE_COLUMNS + ["Adj Close"] if column not in df
    ]
    if missing_columns:
        raise ValueError(f"Missing required model columns: {missing_columns}")

    model_df = df.dropna(subset=FEATURE_COLUMNS + ["Adj Close"])
    x = model_df[FEATURE_COLUMNS]
    y = model_df["Adj Close"]
    if "DATE" in model_df.columns:
        dates = pd.to_datetime(model_df["DATE"])
        x.index = dates
        y.index = dates
        x.index.name = "DATE"
        y.index.name = "DATE"

    split_index = int(len(model_df) * (1 - test_size))
    if split_index == 0 or split_index == len(model_df):
        raise ValueError("Not enough rows for the requested train/test split")

    return (
        x.iloc[:split_index],
        x.iloc[split_index:],
        y.iloc[:split_index],
        y.iloc[split_index:],
    )


def fit_and_evaluate_model_performance(x_train, y_train, x_test, y_test):
    """Fit linear regression and report holdout metrics and comparisons."""
    model = LinearRegression().fit(x_train, y_train)
    predictions = model.predict(x_test)

    result_df = pd.DataFrame(
        {"Actual": y_test, "Predicted": predictions},
        index=y_test.index,
    )
    result_df["Absolute Error"] = (result_df["Actual"] - result_df["Predicted"]).abs()
    result_df["Absolute Error (%)"] = (
        result_df["Absolute Error"] / result_df["Actual"].abs() * 100
    )
    model_df = pd.DataFrame(
        {"Variable": x_train.columns, "Coefficient": model.coef_}
    )
    model_df.loc[len(model_df)] = ["Intercept", model.intercept_]

    print("Intercept:", model.intercept_)
    print("Coefficients:", list(zip(x_train.columns, model.coef_)))
    print(f"Test R squared: {r2_score(y_test, predictions):.2f}")
    print(f"Test MAE: ${mean_absolute_error(y_test, predictions):,.2f}")
    print(f"Test RMSE: ${np.sqrt(mean_squared_error(y_test, predictions)):,.2f}")
    print(
        "Test MAPE: "
        f"{result_df['Absolute Error (%)'].mean():.2f}%"
    )
    print("\nMost recent test-set actual vs. predicted values:")
    print(
        result_df.tail(12).to_string(
            formatters={
                "Actual": "${:,.2f}".format,
                "Predicted": "${:,.2f}".format,
                "Absolute Error": "${:,.2f}".format,
                "Absolute Error (%)": "{:,.2f}%".format,
            }
        )
    )

    plt.scatter(result_df.index, result_df["Actual"], label="Actual")
    plt.scatter(result_df.index, result_df["Predicted"], label="Predicted")
    plt.legend()

    return result_df, model_df, model


def _select_ridge_alpha(training_data, feature_columns, alpha_candidates):
    splitter = TimeSeriesSplit(n_splits=5)
    validation_results = {}
    for alpha in alpha_candidates:
        fold_errors = []
        fold_maes = []
        for train_indices, validation_indices in splitter.split(training_data):
            fold_train = training_data.iloc[train_indices]
            fold_validation = training_data.iloc[validation_indices]
            fold_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
            fold_model.fit(
                fold_train[feature_columns],
                fold_train["Monthly Return (%)"],
            )
            validation_predictions = fold_model.predict(
                fold_validation[feature_columns]
            )
            fold_maes.append(
                mean_absolute_error(
                    fold_validation["Monthly Return (%)"],
                    validation_predictions,
                )
            )
            fold_errors.extend(
                (
                    fold_validation["Monthly Return (%)"].to_numpy()
                    - validation_predictions
                ).tolist()
            )
        validation_results[alpha] = (np.mean(fold_maes), fold_errors)

    best_alpha = min(
        validation_results,
        key=lambda alpha: validation_results[alpha][0],
    )
    validation_mae, calibration_errors = validation_results[best_alpha]
    return best_alpha, validation_mae, calibration_errors


def _print_ridge_feature_contributions(
    model,
    feature_values,
    forecast_label,
    feature_columns=RETURN_FEATURE_COLUMNS,
):
    features = pd.DataFrame(
        [feature_values],
        columns=feature_columns,
    )
    scaler = model.named_steps["standardscaler"]
    ridge = model.named_steps["ridge"]
    z_scores = scaler.transform(features)[0]
    contributions = ridge.coef_ * z_scores
    result = pd.DataFrame(
        {
            "Feature": feature_columns,
            "Input value": features.iloc[0].to_numpy(),
            "Standard deviations from training mean": z_scores,
            "Coefficient (return pp per SD)": ridge.coef_,
            "Contribution (return pp)": contributions,
        }
    ).sort_values(
        "Contribution (return pp)",
        key=lambda values: values.abs(),
        ascending=False,
    )
    predicted_return = float(ridge.intercept_ + contributions.sum())
    print(f"\nRidge feature contributions for {forecast_label}:")
    print(
        "Coefficients are signed return percentage points per one training "
        "standard deviation; contributions show this forecast's feature effects."
    )
    print(
        f"Intercept: {float(ridge.intercept_):+.3f} return percentage points; "
        f"predicted return from equation: {predicted_return:+.3f}%"
    )
    print(
        result.to_string(
            index=False,
            formatters={
                "Input value": "{:,.3f}".format,
                "Standard deviations from training mean": "{:+.3f}".format,
                "Coefficient (return pp per SD)": "{:+.3f}".format,
                "Contribution (return pp)": "{:+.3f}".format,
            },
        )
    )


def walk_forward_return_evaluation(
    df,
    test_start="2024-01-01",
    alpha_candidates=(0.1, 1.0, 10.0, 100.0),
    interval_window=60,
):
    """Evaluate Ridge return forecasts and price baselines in expanding windows."""
    required_columns = [
        "DATE",
        "Adj Close",
        "Previous Avg SPY",
        "Monthly Return (%)",
        *SPY_TECHNICAL_FEATURE_COLUMNS,
    ]
    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Missing required evaluation columns: {missing_columns}")
    if interval_window < 1:
        raise ValueError("interval_window must be at least 1")

    model_df = df.copy()
    model_df["DATE"] = pd.to_datetime(model_df["DATE"])
    model_df = model_df.dropna(
        subset=RETURN_FEATURE_COLUMNS
        + ["Adj Close", "Previous Avg SPY", "Monthly Return (%)"]
    ).sort_values("DATE").reset_index(drop=True)

    cutoff = pd.Timestamp(test_start)
    training_data = model_df.loc[model_df["DATE"] < cutoff].reset_index(drop=True)
    test_data = model_df.loc[model_df["DATE"] >= cutoff].reset_index(drop=True)
    if len(training_data) < 120 or test_data.empty:
        raise ValueError("Need at least 120 training rows and one test row")

    best_alpha, validation_mae, calibration_errors = _select_ridge_alpha(
        training_data,
        RETURN_FEATURE_COLUMNS,
        alpha_candidates,
    )
    calibration_errors = list(calibration_errors)

    forecasts = []
    latest_model = None
    latest_row = None
    for _, row in test_data.iterrows():
        historical_data = model_df.loc[model_df["DATE"] < row["DATE"]]
        model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=best_alpha),
        )
        model.fit(
            historical_data[RETURN_FEATURE_COLUMNS],
            historical_data["Monthly Return (%)"],
        )
        latest_model = model
        latest_row = row
        predicted_return = model.predict(
            row[RETURN_FEATURE_COLUMNS].to_frame().T
        )[0]

        recent_errors = calibration_errors[-interval_window:]
        lower_error, upper_error = np.quantile(recent_errors, [0.1, 0.9])
        previous_price = row["Previous Avg SPY"]
        predicted_price = previous_price * (1 + predicted_return / 100)
        lower_price = previous_price * (
            1 + (predicted_return + lower_error) / 100
        )
        upper_price = previous_price * (
            1 + (predicted_return + upper_error) / 100
        )
        momentum_return = row["SPY_RETURN_1M_LAG1"]

        forecasts.append(
            {
                "DATE": row["DATE"],
                "Actual": row["Adj Close"],
                "Ridge Forecast": predicted_price,
                "Persistence": previous_price,
                "Momentum": previous_price * (1 + momentum_return / 100),
                "Interval Low": lower_price,
                "Interval High": upper_price,
            }
        )
        calibration_errors.append(row["Monthly Return (%)"] - predicted_return)

    results = pd.DataFrame(forecasts).set_index("DATE")
    results["Absolute Error"] = (
        results["Actual"] - results["Ridge Forecast"]
    ).abs()
    results["Absolute Error (%)"] = (
        results["Absolute Error"] / results["Actual"].abs() * 100
    )
    metric_rows = []
    for forecast_column in ("Ridge Forecast", "Persistence", "Momentum"):
        errors = results["Actual"] - results[forecast_column]
        metric_rows.append(
            {
                "Forecast": forecast_column,
                "MAE": mean_absolute_error(
                    results["Actual"], results[forecast_column]
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(results["Actual"], results[forecast_column])
                ),
                "MAPE": np.mean(
                    (errors.abs() / results["Actual"].abs()) * 100
                ),
                "R squared": r2_score(
                    results["Actual"], results[forecast_column]
                ),
            }
        )
    metrics_df = pd.DataFrame(metric_rows).set_index("Forecast")
    interval_coverage = (
        (results["Actual"] >= results["Interval Low"])
        & (results["Actual"] <= results["Interval High"])
    ).mean()
    interval_width = (results["Interval High"] - results["Interval Low"]).mean()

    print(f"Selected Ridge alpha: {best_alpha:g}")
    print(f"Mean validation return MAE: {validation_mae:.3f} percentage points")
    print(
        f"Expanding-window test: {results.index.min():%Y-%m} to "
        f"{results.index.max():%Y-%m} ({len(results)} months)"
    )
    print(metrics_df.to_string(float_format=lambda value: f"{value:.3f}"))
    print(
        f"Empirical 80% interval coverage: {interval_coverage:.1%}; "
        f"mean width: ${interval_width:,.2f}"
    )
    print("\nMost recent test forecasts:")
    formatters = {
        column: "${:,.2f}".format
        for column in results.columns
        if column != "Absolute Error (%)"
    }
    formatters["Absolute Error (%)"] = "{:,.2f}%".format
    print(
        results.tail(12).to_string(
            formatters=formatters
        )
    )

    _print_ridge_feature_contributions(
        latest_model,
        latest_row[RETURN_FEATURE_COLUMNS],
        f"backtest month {results.index[-1]:%Y-%m}",
    )

    _plot_forecasts_vs_actuals(results)
    return results, metrics_df, best_alpha


def _plot_forecasts_vs_actuals(
    results,
    forecast_columns=None,
    title="SPY Monthly Average: Walk-Forward Forecast vs. Actual",
):
    """Plot walk-forward predictions, actual monthly averages, and intervals."""
    if forecast_columns is None:
        forecast_columns = {
            "Ridge Forecast": "Ridge forecast",
            "Persistence": "Persistence baseline",
            "Momentum": "Momentum baseline",
        }

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.plot(
        results.index,
        results["Actual"],
        marker="o",
        label="Actual monthly average",
    )
    for forecast_column, label in forecast_columns.items():
        axis.plot(
            results.index,
            results[forecast_column],
            marker="o" if forecast_column.startswith("Ridge") else None,
            label=label,
        )
    if {"Interval Low", "Interval High"}.issubset(results.columns):
        axis.fill_between(
            results.index,
            results["Interval Low"].to_numpy(dtype=float),
            results["Interval High"].to_numpy(dtype=float),
            alpha=0.15,
            label="Technical Ridge empirical 80% interval",
        )
    axis.set_title(title)
    axis.set_xlabel("Target month")
    axis.set_ylabel("Adjusted close monthly average ($)")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def run_recent_month_backtest(
    months=12,
    as_of=None,
    start_date="1993-02-01",
    ticker="SPY",
    alpha_candidates=(0.1, 1.0, 10.0, 100.0),
):
    """Print expanding-window forecasts for the most recent complete months."""
    if months < 1:
        raise ValueError("months must be at least 1")

    as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    current_month = as_of.to_period("M").to_timestamp()
    test_start = current_month - pd.DateOffset(months=months)
    spy_df = _spy_monthly_data(
        start_date,
        end_date=as_of.strftime("%Y-%m-%d"),
        ticker=ticker,
    )
    spy_df = end_date_for_spy_df(spy_df, as_of=as_of)
    results, metrics, alpha = walk_forward_return_evaluation(
        spy_df,
        test_start=test_start,
        alpha_candidates=alpha_candidates,
    )
    if len(results) != months:
        raise ValueError(
            f"Requested {months} complete months, but only {len(results)} "
            "out-of-sample months were available"
        )
    return results, metrics, alpha


def run_model_comparison_backtest(
    months=12,
    as_of=None,
    start_date="1993-02-01",
    ticker="SPY",
    alpha_candidates=(0.1, 1.0, 10.0, 100.0),
):
    """Compare Ridge, a shallow tree, ARIMA, and simple price baselines."""
    if months < 1:
        raise ValueError("months must be at least 1")

    as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    spy_df, cpi_df, fed_df, oil_df, ppi_df, unr_df = inputs(
        start_date=start_date,
        end_date=as_of.strftime("%Y-%m-%d"),
        ticker=ticker,
    )
    spy_df = end_date_for_spy_df(spy_df, as_of=as_of)
    merged_df = merge_all_dfs_for_predictions(
        oil_df,
        cpi_df,
        ppi_df,
        unr_df,
        fed_df,
        spy_df,
    )
    model_df = _add_macro_return_features(merged_df)
    required_features = (
        COMBINED_RETURN_FEATURE_COLUMNS
        + ["Adj Close", "Previous Avg SPY", "Monthly Return (%)"]
    )
    model_df = model_df.dropna(subset=required_features)
    model_df = model_df.sort_values("DATE").reset_index(drop=True)

    test_data = model_df.tail(months).reset_index(drop=True)
    if len(test_data) != months:
        raise ValueError(
            f"Requested {months} complete months, but only {len(test_data)} "
            "common out-of-sample months were available"
        )
    test_start = pd.Timestamp(test_data["DATE"].iloc[0])
    training_data = model_df.loc[model_df["DATE"] < test_start].reset_index(
        drop=True
    )
    if len(training_data) < 120:
        raise ValueError("Need at least 120 common training rows")

    technical_alpha, _, technical_errors = _select_ridge_alpha(
        training_data,
        RETURN_FEATURE_COLUMNS,
        alpha_candidates,
    )
    combined_alpha, _, _ = _select_ridge_alpha(
        training_data,
        COMBINED_RETURN_FEATURE_COLUMNS,
        alpha_candidates,
    )
    technical_errors = list(technical_errors)

    forecasts = []
    latest_models = {}
    latest_row = None
    for _, row in test_data.iterrows():
        historical_data = model_df.loc[model_df["DATE"] < row["DATE"]]
        technical_model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=technical_alpha),
        )
        technical_model.fit(
            historical_data[RETURN_FEATURE_COLUMNS],
            historical_data["Monthly Return (%)"],
        )
        combined_model = make_pipeline(
            StandardScaler(),
            Ridge(alpha=combined_alpha),
        )
        combined_model.fit(
            historical_data[COMBINED_RETURN_FEATURE_COLUMNS],
            historical_data["Monthly Return (%)"],
        )
        tree_model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=75,
            max_leaf_nodes=4,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )
        tree_model.fit(
            historical_data[COMBINED_RETURN_FEATURE_COLUMNS],
            historical_data["Monthly Return (%)"],
        )
        arima_history = spy_df.loc[spy_df["DATE"] < row["DATE"]]
        arima_history = arima_history.dropna(
            subset=["Monthly Return (%)"]
        ).sort_values("DATE")
        arima_model = ARIMA(
            arima_history["Monthly Return (%)"].to_numpy(dtype=float),
            order=(1, 0, 1),
            trend="c",
        ).fit()

        latest_models = {
            "Ridge technical": technical_model,
            "Ridge technical + macro": combined_model,
        }
        latest_row = row
        predicted_returns = {
            "Ridge technical": float(
                technical_model.predict(
                    row[RETURN_FEATURE_COLUMNS].to_frame().T
                )[0]
            ),
            "Ridge technical + macro": float(
                combined_model.predict(
                    row[COMBINED_RETURN_FEATURE_COLUMNS].to_frame().T
                )[0]
            ),
            "Shallow tree (technical + macro)": float(
                tree_model.predict(
                    row[COMBINED_RETURN_FEATURE_COLUMNS].to_frame().T
                )[0]
            ),
            "ARIMA(1,0,1)": float(
                np.asarray(arima_model.forecast(steps=1))[0]
            ),
        }

        anchor_price = float(row["Previous Avg SPY"])
        forecast_row = {
            "DATE": row["DATE"],
            "Actual": row["Adj Close"],
        }
        for model_name, predicted_return in predicted_returns.items():
            forecast_row[model_name] = anchor_price * (
                1 + predicted_return / 100
            )

        lower_error, upper_error = np.quantile(
            technical_errors[-60:],
            [0.1, 0.9],
        )
        technical_return = predicted_returns["Ridge technical"]
        forecast_row["Interval Low"] = anchor_price * (
            1 + (technical_return + lower_error) / 100
        )
        forecast_row["Interval High"] = anchor_price * (
            1 + (technical_return + upper_error) / 100
        )
        forecast_row["Persistence"] = anchor_price
        forecast_row["Momentum"] = anchor_price * (
            1 + float(row["SPY_RETURN_1M_LAG1"]) / 100
        )
        forecasts.append(forecast_row)
        technical_errors.append(
            float(row["Monthly Return (%)"]) - technical_return
        )

    results = pd.DataFrame(forecasts).set_index("DATE")
    forecast_columns = [
        "Ridge technical",
        "Ridge technical + macro",
        "Shallow tree (technical + macro)",
        "ARIMA(1,0,1)",
        "Persistence",
        "Momentum",
    ]
    metric_rows = []
    for model_name in forecast_columns:
        errors = results["Actual"] - results[model_name]
        metric_rows.append(
            {
                "Forecast": model_name,
                "MAE": mean_absolute_error(
                    results["Actual"],
                    results[model_name],
                ),
                "RMSE": np.sqrt(
                    mean_squared_error(
                        results["Actual"],
                        results[model_name],
                    )
                ),
                "MAPE": np.mean(
                    errors.abs() / results["Actual"].abs() * 100
                ),
                "R squared": r2_score(
                    results["Actual"],
                    results[model_name],
                ),
            }
        )
    metrics = pd.DataFrame(metric_rows).set_index("Forecast")
    annual_mae_rows = []
    for year, year_data in results.groupby(results.index.year):
        for model_name in forecast_columns:
            annual_mae_rows.append(
                {
                    "Year": year,
                    "Forecast": model_name,
                    "MAE": mean_absolute_error(
                        year_data["Actual"],
                        year_data[model_name],
                    ),
                }
            )
    annual_mae = pd.DataFrame(annual_mae_rows).pivot(
        index="Year",
        columns="Forecast",
        values="MAE",
    )
    annual_mae.insert(
        0,
        "Months",
        results.groupby(results.index.year).size(),
    )
    coverage = (
        (results["Actual"] >= results["Interval Low"])
        & (results["Actual"] <= results["Interval High"])
    ).mean()

    print(f"Technical Ridge alpha: {technical_alpha:g}")
    print(f"Combined Ridge alpha: {combined_alpha:g}")
    print(
        f"Common expanding-window test: {results.index.min():%Y-%m} to "
        f"{results.index.max():%Y-%m} ({len(results)} months)"
    )
    calendar_months = pd.period_range(
        results.index.min(),
        results.index.max(),
        freq="M",
    )
    tested_months = results.index.to_period("M")
    omitted_months = [
        month.strftime("%Y-%m")
        for month in calendar_months
        if month not in tested_months
    ]
    if omitted_months:
        print(
            "Omitted months with unavailable common features: "
            + ", ".join(omitted_months)
        )
    print("Macro features: oil monthly % change, CPI/PPI year-over-year %,")
    print("               Fed funds rate, unemployment rate (release-lagged)")
    print(metrics.to_string(float_format=lambda value: f"{value:.3f}"))
    print("\nAnnual MAE ($):")
    print(annual_mae.to_string(float_format=lambda value: f"{value:.2f}"))
    print(
        f"Technical Ridge 80% interval coverage: {coverage:.1%} "
        "(common test window)"
    )
    print("\nActual monthly averages and model forecasts:")
    print(
        results.tail(12).to_string(
            formatters={
                column: "${:,.2f}".format
                for column in results.columns
            }
        )
    )

    _print_ridge_feature_contributions(
        latest_models["Ridge technical"],
        latest_row[RETURN_FEATURE_COLUMNS],
        f"latest test month {results.index[-1]:%Y-%m} (technical only)",
    )
    _print_ridge_feature_contributions(
        latest_models["Ridge technical + macro"],
        latest_row[COMBINED_RETURN_FEATURE_COLUMNS],
        f"latest test month {results.index[-1]:%Y-%m} (technical + macro)",
        feature_columns=COMBINED_RETURN_FEATURE_COLUMNS,
    )
    _plot_forecasts_vs_actuals(
        results,
        forecast_columns={
            "Ridge technical": "Ridge - technical",
            "Ridge technical + macro": "Ridge - technical + macro",
            "Shallow tree (technical + macro)": "Shallow tree - technical + macro",
            "ARIMA(1,0,1)": "ARIMA(1,0,1)",
            "Persistence": "Persistence baseline",
            "Momentum": "Momentum baseline",
        },
        title="SPY Monthly Average: Model Comparison vs. Actual",
    )
    return results, metrics


def _write_monthly_forecast_ledger(forecast, ledger_path):
    ledger_path = Path(ledger_path)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    key_columns = ["forecast_origin", "target_month", "model_version"]
    actual_columns = [
        "actual_monthly_average",
        "absolute_error",
        "absolute_percentage_error",
        "interval_contains_actual",
    ]

    if ledger_path.exists():
        ledger = pd.read_csv(ledger_path)
        missing_columns = [
            column for column in forecast if column not in ledger.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Forecast ledger is missing columns: {missing_columns}"
            )
        matching = pd.Series(True, index=ledger.index)
        for column in key_columns:
            matching &= ledger[column].astype(str) == str(forecast[column])
        if matching.any():
            previous = ledger.loc[matching].iloc[-1]
            for column in actual_columns:
                if pd.isna(forecast[column]) and pd.notna(previous[column]):
                    forecast[column] = previous[column]
            if pd.notna(forecast["actual_monthly_average"]):
                actual = float(forecast["actual_monthly_average"])
                predicted = float(forecast["predicted_monthly_average"])
                forecast["absolute_error"] = abs(actual - predicted)
                forecast["absolute_percentage_error"] = (
                    abs(actual - predicted) / abs(actual) * 100
                )
                forecast["interval_contains_actual"] = (
                    float(forecast["interval_lower"])
                    <= actual
                    <= float(forecast["interval_upper"])
                    if pd.notna(forecast["interval_lower"])
                    and pd.notna(forecast["interval_upper"])
                    else np.nan
                )
            ledger = ledger.loc[~matching]
    else:
        ledger = pd.DataFrame(columns=forecast.keys())

    if ledger.empty:
        ledger = pd.DataFrame([forecast], columns=ledger.columns)
    else:
        ledger = pd.concat(
            [ledger, pd.DataFrame([forecast])],
            ignore_index=True,
        )
    ledger = ledger.drop_duplicates(subset=key_columns, keep="last")
    ledger = ledger.sort_values(key_columns).reset_index(drop=True)
    ledger.to_csv(ledger_path, index=False)
    return ledger


def update_spy_monthly_forecast_ledger(
    ledger_path=MONTHLY_FORECAST_LEDGER,
    as_of=None,
    ticker="SPY",
):
    """Fill realized monthly prices and errors for completed forecast months."""
    ledger_path = Path(ledger_path)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Forecast ledger not found: {ledger_path}")

    ledger = pd.read_csv(ledger_path)
    required_columns = [
        "target_month",
        "actual_monthly_average",
        "predicted_monthly_average",
        "interval_lower",
        "interval_upper",
        "absolute_error",
        "absolute_percentage_error",
        "interval_contains_actual",
    ]
    missing_columns = [column for column in required_columns if column not in ledger]
    if missing_columns:
        raise ValueError(f"Forecast ledger is missing columns: {missing_columns}")
    if ledger.empty:
        return ledger

    as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    current_month = as_of.to_period("M").to_timestamp()
    first_target = pd.to_datetime(ledger["target_month"]).min()
    actuals = _spy_monthly_data(
        first_target.strftime("%Y-%m-%d"),
        end_date=as_of.strftime("%Y-%m-%d"),
        ticker=ticker,
    )
    actuals = end_date_for_spy_df(actuals, as_of=as_of)
    actuals_by_month = actuals.set_index("DATE")["Adj Close"]

    for index, row in ledger.iterrows():
        target_month = pd.Timestamp(row["target_month"]).to_period("M").to_timestamp()
        if target_month >= current_month or target_month not in actuals_by_month.index:
            continue

        actual = float(actuals_by_month.loc[target_month])
        predicted = float(row["predicted_monthly_average"])
        ledger.loc[index, "actual_monthly_average"] = actual
        ledger.loc[index, "absolute_error"] = abs(actual - predicted)
        ledger.loc[index, "absolute_percentage_error"] = (
            abs(actual - predicted) / abs(actual) * 100
        )
        if pd.notna(row["interval_lower"]) and pd.notna(row["interval_upper"]):
            ledger.loc[index, "interval_contains_actual"] = (
                float(row["interval_lower"])
                <= actual
                <= float(row["interval_upper"])
            )
        else:
            ledger.loc[index, "interval_contains_actual"] = np.nan

    ledger.to_csv(ledger_path, index=False)
    return ledger


def _latest_month_value(frame, column, month, series_name):
    month = pd.Timestamp(month).to_period("M").to_timestamp()
    dates = pd.to_datetime(frame["DATE"]).dt.to_period("M").dt.to_timestamp()
    values = pd.to_numeric(
        frame.loc[dates <= month, column],
        errors="coerce",
    ).dropna()
    if values.empty:
        raise ValueError(
            f"No {series_name} observation is available on or before "
            f"{month:%Y-%m}"
        )
    observation_index = values.index[-1]
    return dates.loc[observation_index], float(values.iloc[-1])


def _next_month_macro_features(
    target_month,
    as_of,
    cpi_df,
    oil_df,
    ppi_df,
    unr_df,
    fed_df,
):
    target_month = pd.Timestamp(target_month).to_period("M").to_timestamp()
    current_month = pd.Timestamp(as_of).to_period("M").to_timestamp()
    oil_source_month = target_month - pd.DateOffset(
        months=FEATURE_LAGS["POILWTIUSDM"]
    )
    oil_observation_month, oil_current = _latest_month_value(
        oil_df,
        "POILWTIUSDM",
        oil_source_month,
        "WTI oil",
    )
    _, oil_previous = _latest_month_value(
        oil_df,
        "POILWTIUSDM",
        oil_observation_month - pd.DateOffset(months=1),
        "prior-month WTI oil",
    )
    if oil_previous == 0:
        raise ValueError("Prior-month WTI oil price cannot be zero")

    cpi_source_month = target_month - pd.DateOffset(
        months=FEATURE_LAGS["CPIAUCSL"]
    )
    ppi_source_month = target_month - pd.DateOffset(
        months=FEATURE_LAGS["PPIACO"]
    )
    cpi_observation_month, cpi_current = _latest_month_value(
        cpi_df,
        "CPIAUCSL",
        cpi_source_month,
        "CPI",
    )
    _, cpi_prior_year = _latest_month_value(
        cpi_df,
        "CPIAUCSL",
        cpi_observation_month - pd.DateOffset(months=12),
        "prior-year CPI",
    )
    ppi_observation_month, ppi_current = _latest_month_value(
        ppi_df,
        "PPIACO",
        ppi_source_month,
        "PPI",
    )
    _, ppi_prior_year = _latest_month_value(
        ppi_df,
        "PPIACO",
        ppi_observation_month - pd.DateOffset(months=12),
        "prior-year PPI",
    )
    if cpi_prior_year == 0 or ppi_prior_year == 0:
        raise ValueError("Prior-year CPI and PPI values cannot be zero")
    unemployment_source_month = target_month - pd.DateOffset(
        months=FEATURE_LAGS["UNRATE"]
    )
    unemployment_observation_month, unemployment = _latest_month_value(
        unr_df,
        "UNRATE",
        unemployment_source_month,
        "unemployment rate",
    )

    daily_fed = _fred_series(
        "DFF",
        current_month.strftime("%Y-%m-%d"),
        end_date=pd.Timestamp(as_of).strftime("%Y-%m-%d"),
    )
    fed_values = daily_fed.loc[
        daily_fed["DATE"] == current_month,
        "value",
    ].dropna()
    if fed_values.empty:
        fed_observation_month, fed_rate = _latest_month_value(
            fed_df,
            "FEDFUNDS",
            current_month,
            "Fed funds rate",
        )
    else:
        fed_observation_month = current_month
        fed_rate = float(fed_values.mean())

    features = {
        "OIL_MONTHLY_CHANGE_PCT_LAG1": (oil_current / oil_previous - 1) * 100,
        "CPI_YOY_PCT_LAG2": (cpi_current / cpi_prior_year - 1) * 100,
        "PPI_YOY_PCT_LAG2": (ppi_current / ppi_prior_year - 1) * 100,
        "FEDFUNDS_RATE_LAG1": fed_rate,
        "UNEMPLOYMENT_RATE_LAG2": unemployment,
    }
    sources = {
        "oil": oil_observation_month.strftime("%Y-%m"),
        "CPI": cpi_observation_month.strftime("%Y-%m"),
        "PPI": ppi_observation_month.strftime("%Y-%m"),
        "Fed funds": fed_observation_month.strftime("%Y-%m"),
        "unemployment": unemployment_observation_month.strftime("%Y-%m"),
    }
    return features, sources


def run_and_log_next_month_forecast(
    as_of=None,
    start_date="1993-02-01",
    ledger_path=MONTHLY_FORECAST_LEDGER,
    ticker="SPY",
    alpha_candidates=(0.1, 1.0, 10.0, 100.0),
):
    """Forecast next month's average with every tested model and log the results."""
    as_of = pd.Timestamp(as_of or pd.Timestamp.today()).normalize()
    current_month = as_of.to_period("M").to_timestamp()
    target_month = current_month + pd.DateOffset(months=1)
    end_date = as_of.strftime("%Y-%m-%d")
    ledger_path = Path(ledger_path)

    if ledger_path.exists():
        update_spy_monthly_forecast_ledger(
            ledger_path=ledger_path,
            as_of=as_of,
            ticker=ticker,
        )

    spy_df, cpi_df, fed_df, oil_df, ppi_df, unr_df = inputs(
        start_date=start_date,
        end_date=end_date,
        ticker=ticker,
    )
    spy_df = end_date_for_spy_df(spy_df, as_of=as_of)
    merged_df = merge_all_dfs_for_predictions(
        oil_df,
        cpi_df,
        ppi_df,
        unr_df,
        fed_df,
        spy_df,
    )
    training_data = _add_macro_return_features(merged_df)
    training_data = training_data.dropna(
        subset=COMBINED_RETURN_FEATURE_COLUMNS
        + ["Adj Close", "Previous Avg SPY", "Monthly Return (%)"]
    ).sort_values("DATE").reset_index(drop=True)
    if len(training_data) < 120:
        raise ValueError("Need at least 120 complete common training months")

    partial_spy_df = _spy_monthly_data(
        start_date,
        end_date=end_date,
        ticker=ticker,
    )
    partial_month = partial_spy_df.loc[partial_spy_df["DATE"] == current_month]
    if partial_month.empty:
        raise ValueError(
            f"No SPY month-to-date data is available for {current_month:%Y-%m}"
        )
    partial_spy = partial_month.iloc[-1]

    forecast_features = {}
    for feature_column, current_column in CURRENT_SPY_FEATURE_COLUMNS.items():
        value = partial_spy[current_column]
        if pd.isna(value):
            raise ValueError(
                f"SPY feature {current_column!r} is unavailable for "
                f"{current_month:%Y-%m}"
            )
        forecast_features[feature_column] = float(value)
    macro_features, macro_sources = _next_month_macro_features(
        target_month,
        as_of,
        cpi_df,
        oil_df,
        ppi_df,
        unr_df,
        fed_df,
    )
    forecast_features.update(macro_features)

    technical_alpha, technical_validation_mae, technical_errors = (
        _select_ridge_alpha(
            training_data,
            RETURN_FEATURE_COLUMNS,
            alpha_candidates,
        )
    )
    combined_alpha, combined_validation_mae, combined_errors = (
        _select_ridge_alpha(
            training_data,
            COMBINED_RETURN_FEATURE_COLUMNS,
            alpha_candidates,
        )
    )

    technical_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=technical_alpha),
    )
    technical_model.fit(
        training_data[RETURN_FEATURE_COLUMNS],
        training_data["Monthly Return (%)"],
    )
    combined_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=combined_alpha),
    )
    combined_model.fit(
        training_data[COMBINED_RETURN_FEATURE_COLUMNS],
        training_data["Monthly Return (%)"],
    )
    tree_model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=75,
        max_leaf_nodes=4,
        min_samples_leaf=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )
    tree_model.fit(
        training_data[COMBINED_RETURN_FEATURE_COLUMNS],
        training_data["Monthly Return (%)"],
    )
    arima_training_data = spy_df.dropna(
        subset=["Monthly Return (%)"]
    ).sort_values("DATE")
    arima_model = ARIMA(
        arima_training_data["Monthly Return (%)"].to_numpy(dtype=float),
        order=(1, 0, 1),
        trend="c",
    ).fit()

    technical_vector = pd.DataFrame(
        [{column: forecast_features[column] for column in RETURN_FEATURE_COLUMNS}],
        columns=RETURN_FEATURE_COLUMNS,
    )
    combined_vector = pd.DataFrame(
        [forecast_features],
        columns=COMBINED_RETURN_FEATURE_COLUMNS,
    )
    predicted_returns = {
        "Ridge technical": float(technical_model.predict(technical_vector)[0]),
        "Ridge technical + macro": float(combined_model.predict(combined_vector)[0]),
        "Shallow tree (technical + macro)": float(
            tree_model.predict(combined_vector)[0]
        ),
        "ARIMA(1,0,1)": float(
            np.asarray(arima_model.forecast(steps=1))[0]
        ),
        "Persistence": 0.0,
        "Momentum": float(partial_spy["SPY_RETURN_1M"]),
    }
    _print_ridge_feature_contributions(
        technical_model,
        technical_vector.iloc[0],
        f"next-month forecast for {target_month:%Y-%m} (technical only)",
    )
    _print_ridge_feature_contributions(
        combined_model,
        combined_vector.iloc[0],
        f"next-month forecast for {target_month:%Y-%m} (technical + macro)",
        feature_columns=COMBINED_RETURN_FEATURE_COLUMNS,
    )

    anchor_price = float(partial_spy["Adj Close"])
    technical_error_bounds = np.quantile(
        list(technical_errors)[-60:],
        [0.1, 0.9],
    )
    combined_error_bounds = np.quantile(
        list(combined_errors)[-60:],
        [0.1, 0.9],
    )
    bounds = {
        "Ridge technical": technical_error_bounds,
        "Ridge technical + macro": combined_error_bounds,
    }
    model_versions = {
        "Ridge technical": MONTHLY_MODEL_VERSION,
        "Ridge technical + macro": "technical_macro_ridge_v1",
        "Shallow tree (technical + macro)": "technical_macro_hist_gradient_v1",
        "ARIMA(1,0,1)": "arima_1_0_1_v1",
        "Persistence": "persistence_v1",
        "Momentum": "momentum_v1",
    }
    feature_names = {
        "Ridge technical": RETURN_FEATURE_COLUMNS,
        "Ridge technical + macro": COMBINED_RETURN_FEATURE_COLUMNS,
        "Shallow tree (technical + macro)": COMBINED_RETURN_FEATURE_COLUMNS,
        "ARIMA(1,0,1)": ["Monthly Return (%) history", "ARIMA(1,0,1)"],
        "Persistence": ["Previous monthly average"],
        "Momentum": ["Current month-to-date return"],
    }
    ridge_alphas = {
        "Ridge technical": technical_alpha,
        "Ridge technical + macro": combined_alpha,
    }
    validation_maes = {
        "Ridge technical": technical_validation_mae,
        "Ridge technical + macro": combined_validation_mae,
    }
    training_frames = {
        "Ridge technical": training_data,
        "Ridge technical + macro": training_data,
        "Shallow tree (technical + macro)": training_data,
        "ARIMA(1,0,1)": arima_training_data,
    }

    forecast_records = []
    summary_rows = []
    for model_name, predicted_return in predicted_returns.items():
        interval_lower = np.nan
        interval_upper = np.nan
        interval_level = np.nan
        if model_name in bounds:
            lower_error, upper_error = bounds[model_name]
            interval_lower = anchor_price * (
                1 + (predicted_return + lower_error) / 100
            )
            interval_upper = anchor_price * (
                1 + (predicted_return + upper_error) / 100
            )
            interval_level = 0.8

        model_training_data = training_frames.get(model_name)
        forecast = {
            "forecast_origin": as_of.strftime("%Y-%m-%d"),
            "target_month": target_month.strftime("%Y-%m-%d"),
            "model_version": model_versions[model_name],
            "data_as_of": as_of.strftime("%Y-%m-%d"),
            "training_start": (
                model_training_data["DATE"].min().strftime("%Y-%m-%d")
                if model_training_data is not None
                else ""
            ),
            "training_end": (
                model_training_data["DATE"].max().strftime("%Y-%m-%d")
                if model_training_data is not None
                else ""
            ),
            "training_rows": (
                len(model_training_data)
                if model_training_data is not None
                else 0
            ),
            "ridge_alpha": ridge_alphas.get(model_name, np.nan),
            "validation_return_mae_pp": validation_maes.get(model_name, np.nan),
            "features": "|".join(feature_names[model_name]),
            "anchor_month": current_month.strftime("%Y-%m-%d"),
            "anchor_avg_spy_mtd": anchor_price,
            "predicted_return_pct": predicted_return,
            "predicted_monthly_average": anchor_price
            * (1 + predicted_return / 100),
            "interval_level": interval_level,
            "interval_lower": interval_lower,
            "interval_upper": interval_upper,
            "actual_monthly_average": np.nan,
            "absolute_error": np.nan,
            "absolute_percentage_error": np.nan,
            "interval_contains_actual": np.nan,
        }
        ledger = _write_monthly_forecast_ledger(forecast, ledger_path)
        forecast_records.append(forecast)
        summary_rows.append(
            {
                "Model": model_name,
                "Predicted return (%)": predicted_return,
                "Predicted monthly average": forecast[
                    "predicted_monthly_average"
                ],
                "80% interval": (
                    f"${interval_lower:,.2f}-${interval_upper:,.2f}"
                    if pd.notna(interval_lower)
                    else "Not estimated"
                ),
            }
        )

    print(
        f"\nNext-month forecasts for {target_month:%Y-%m}; "
        f"anchor ${anchor_price:,.2f} from {current_month:%Y-%m} MTD."
    )
    print(
        "Oil and Fed funds use current month-to-date values; CPI, PPI, and "
        "unemployment use release-lagged published values."
    )
    print(
        "Latest macro observations used: "
        + ", ".join(
            f"{series} {observation_month}"
            for series, observation_month in macro_sources.items()
        )
    )
    print(
        pd.DataFrame(summary_rows).to_string(
            index=False,
            formatters={
                "Predicted return (%)": "{:+.3f}%".format,
                "Predicted monthly average": "${:,.2f}".format,
            },
        )
    )
    print(f"All model forecasts saved to {ledger_path}")
    technical_forecast = next(
        record
        for record in forecast_records
        if record["model_version"] == MONTHLY_MODEL_VERSION
    )
    return technical_forecast, ledger


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Forecast monthly SPY average prices."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "forecast",
        help="forecast and log next month's prices from every tested model",
    )
    backtest_parser = commands.add_parser(
        "backtest",
        help="compare rolling forecasts from every tested model with actuals",
    )
    backtest_parser.add_argument(
        "--months",
        type=int,
        default=60,
        help="number of most recent common months to compare (default: 60)",
    )
    compare_parser = commands.add_parser(
        "compare",
        help="compare technical, macro, tree, and ARIMA forecasts",
    )
    compare_parser.add_argument(
        "--months",
        type=int,
        default=60,
        help="number of most recent common months to compare (default: 60)",
    )
    args = parser.parse_args(argv)

    if args.command == "forecast":
        run_and_log_next_month_forecast()
    else:
        run_model_comparison_backtest(months=args.months)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        main(["compare", "--months", "60"])
# %%
# With no command-line arguments, the Run button compares the last 60 common
# months. Pass "forecast", "backtest", or "compare" in the terminal to choose.
# %%
 