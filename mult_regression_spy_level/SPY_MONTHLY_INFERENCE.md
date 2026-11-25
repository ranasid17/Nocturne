# Monthly SPY Forecast Integration

## Purpose and source of truth

Integrate the monthly SPY average-price forecasts into the app using the
existing model. Do not copy model calculations into the frontend.

- Model, data retrieval, forecast, and backtest entry points:
  `qusa/mult_regression_spy_level/mult_regression_stock_model.py`
- Persistent forecast ledger:
  `spy_monthly_forecast_ledger.csv` at the repository root
- Python dependencies:
  `qusa/requirements.txt`

The script downloads SPY prices from Yahoo Finance and macroeconomic series
from FRED. It does not require user-supplied monthly inputs or a FRED API key.
Use the repository's configured Python environment and installed requirements.

## Python entry points

Run from the repository root:

```powershell
python .\qusa\mult_regression_spy_level\mult_regression_stock_model.py forecast
python .\qusa\mult_regression_spy_level\mult_regression_stock_model.py backtest --months 60
```

`forecast` generates next month's predictions from every supported model,
updates actuals for completed ledger targets, and upserts forecast rows.
`backtest` and `compare` run the same expanding-window model comparison for
the requested number of recent common months. With no command-line arguments,
the file runs a 60-month comparison.

The app should call the Python functions from a backend worker/service rather
than run a long model fit in a browser request:

```python
from qusa.mult_regression_spy_level.mult_regression_stock_model import (
    MONTHLY_FORECAST_LEDGER,
    run_and_log_next_month_forecast,
    run_model_comparison_backtest,
    update_spy_monthly_forecast_ledger,
)

# Explicit user-triggered refresh; returns the technical forecast and full ledger.
technical_forecast, ledger = run_and_log_next_month_forecast()

# Backtest returns (results_dataframe, metrics_dataframe).
results, metrics = run_model_comparison_backtest(months=60)

# Update realized values after target months have completed.
ledger = update_spy_monthly_forecast_ledger()
```

Resolve the ledger path from `MONTHLY_FORECAST_LEDGER`; do not hard-code a
second output location. Show data-fetch/model errors to the user or log them
as failures; do not present stale data as a successful refresh. Avoid
automatically triggering downloads and model fitting on every page load.

## Forecast models

The next-month command currently records these model versions:

| Model | `model_version` | Inputs / notes |
|---|---|---|
| Ridge, technical | `technical_ridge_common_v2` | Lagged SPY returns, volatility, and 200-day SMA distance |
| Ridge, technical + macro | `technical_macro_ridge_v1` | Technical features plus transformed, lagged FRED features |
| Shallow tree, technical + macro | `technical_macro_hist_gradient_v1` | Constrained histogram gradient boosting regression |
| ARIMA(1,0,1) | `arima_1_0_1_v1` | Historical monthly SPY returns |
| Persistence | `persistence_v1` | Predicts no change from the anchor average |
| Momentum | `momentum_v1` | Applies the current month-to-date return to the anchor |

The regression target is monthly SPY average-price return. Forecast returns
are converted back into estimated monthly average prices using the current
month-to-date SPY average as the anchor. Macro inputs include oil monthly
percentage change, CPI/PPI year-over-year percentage changes, Fed funds, and
unemployment, with the lag policy implemented in the model. The printed output
reports the observation month used for each macro input.

The two Ridge models currently have empirical 80% intervals. Other models do
not yet have estimated intervals; display “Not estimated” rather than
fabricating a range. Ridge coefficients/contributions are available in the
forecast run output. They describe model associations, not causal effects.

## Ledger contract

The CSV is append/upsert history, not a single “current forecast” record.
Identify a forecast by `(forecast_origin, target_month, model_version)`.
Repeated runs for the same key replace that forecast row rather than adding a
duplicate. Preserve older versions; when displaying current model comparisons,
filter to the supported model versions above and group by origin and target
month.

Important fields:

- `forecast_origin`, `data_as_of`, `target_month`, `model_version`
- `training_start`, `training_end`, `training_rows`
- `ridge_alpha`, `validation_return_mae_pp`, `features`
- `anchor_month`, `anchor_avg_spy_mtd`
- `predicted_return_pct`, `predicted_monthly_average`
- `interval_level`, `interval_lower`, `interval_upper`
- `actual_monthly_average`, `absolute_error`,
  `absolute_percentage_error`, `interval_contains_actual`

Actual and error fields are blank until a target month is complete and the
ledger updater has run. The same target month can have multiple forecast
origins and model versions; retain those records so users can see how each
forecast performed. Do not mistake the legacy `technical_ridge_v1` row for
the current `technical_ridge_common_v2` model.

## App presentation requirements

1. **Next-month forecast:** Show target month, as-of date, anchor price,
   predicted return, and predicted average for each supported model. Show the
   Ridge intervals where available and clearly indicate missing actuals for
   future targets.
2. **Model comparison:** Show a sortable table and a chart comparing actual
   monthly averages with all six model forecasts. Include persistence and
   momentum baselines. Do not plot an incomplete current month as an actual.
3. **Backtest controls:** Allow the user to select a test-window length
   (default 60 common months). Show MAE, RMSE, MAPE, and R-squared, plus the
   forecast-versus-actual chart. Label the evaluation as a historical
   expanding-window backtest, not as forecasts saved prospectively at the
   time.
4. **Ridge interpretation:** For both Ridge models, display the latest
   forecast's signed feature contributions and the input values. Clarify that
   the coefficient is return percentage points per training standard
   deviation and contribution is the feature's effect on that prediction.
5. **Data and status:** Display the data-as-of date, target month, anchor
   month, and macro observation months. Report the latest successful update
   and surface data/provider failures.

`run_model_comparison_backtest` currently prints tables and calls
`matplotlib.pyplot.show()`, which is suitable for a local interactive run but
not a web request. Before using it in a web backend, adapt the plotting helper
to return a Matplotlib `Figure` (or provide an option to suppress `show`) and
render that figure through the app's existing chart library. Keep model
evaluation and prediction calculations in the Python backend.

The comparison requires common rows with all technical and macro features;
months with unavailable macro inputs can therefore be omitted. Display the
actual tested date range and sample count from the returned results rather
than assuming every calendar month is present. Backtest outcomes are
exploratory and do not guarantee future performance.
