# Daily Monte Carlo Inference

## Source of truth

- Model and entry point: `monte_carlo_stock_model.py`
- Run configuration: `monte_carlo_stock_config.json`
- Persistent forecast output: `forecast_ledger.csv` (the path is controlled by `forecast_output_path` in the config)

The config contains the tickers, price-history start date, horizons, simulation settings, and the frozen `selected_windows_by_ticker` mapping. A `null` window means the model uses all available history. Daily inference must use these saved window selections; it must not select or optimize windows again.

## Daily run

Use a Python 3.10+ environment with the packages in `qusa/requirements.txt` installed. Run from the repository root, after the market data provider has published the latest close:

```powershell
python .\monte_carlo_stock_model.py --mode daily --config .\monte_carlo_stock_config.json
```

The callable entry point for an agent that invokes Python directly is:

```python
from pathlib import Path
from monte_carlo_stock_model import run_daily_from_config

ledger = run_daily_from_config(Path("monte_carlo_stock_config.json"))
```

`run_daily_from_config` validates that every configured ticker has a selection for every configured horizon, calls `run_daily_forecast_ledger`, and upserts the result into the configured CSV. Upsert identity is `(ticker, origin, horizon)`, so a rerun for the same origin replaces those rows instead of duplicating them. The function returns the full persisted ledger as a DataFrame.

The origin is the latest available downloaded price date. A normal run produces one new row per ticker and active horizon. Check the process exit code and confirm the ledger's origin date is current before treating a run as successful.

Always pass `--mode daily`. The script's default mode is `calibrate`, not daily.

## Recalibration

Run calibration only when intentionally refreshing the frozen windows, not as part of the daily job:

```powershell
python .\monte_carlo_stock_model.py --mode calibrate --config .\monte_carlo_stock_config.json
```

This runs the historical selection and validation workflow, then writes `selected_windows_by_ticker` and `calibrated_at_utc` back into the same config. It can take substantially longer than daily inference. Review the resulting config before resuming the daily job.

## Current limitation

Forecast simulation horizons count trading observations, but the ledger's `target` currently adds calendar days to the origin. Treat `target` as an approximate date, not the exact session when a horizon matures, until trading-calendar alignment is implemented.
