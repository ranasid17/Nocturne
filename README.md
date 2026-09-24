# Nocturne

Nocturne is a local research and prediction app for overnight US equity moves. Flask serves feature generation, predictions, and saved history; explicit CLI commands handle data fetching, training, evaluation, backtesting, and clustering. It is for research, not financial advice.

## Install

Python 3.9 or newer is required. From a clone:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
flask --app app run
```

Open <http://127.0.0.1:5000>. The server binds to loopback by default and has no authentication. On Windows, create the environment with `py -3 -m venv .venv` and activate `.venv\Scripts\Activate.ps1`. Install `python -m pip install '.[research,test]'` for Matplotlib plots and tests, or `python -m pip install -r requirements.txt` for an editable development install. Runtime dependencies are declared in [pyproject.toml](pyproject.toml).

The app starts without market data, a provider key, or a model. Local history is enough to generate features. Fetching new data requires `POLYGON_API_KEY`; put it in your environment or a local `.env` made from [the example](.env.example). Generated data, models, logs, and credentials stay outside Git.

## Configuration

Config precedence is an explicit `--config` path for the installed CLI, then `QUSA_CONFIG_PATH`, then the packaged [defaults](qusa/utils/defaults.yaml). The packaged paths are relative to the working directory. Paths in a custom YAML file are relative to that file. Existing [local settings](qusa/utils/config.yaml) are used only when selected explicitly; for example:

```sh
export QUSA_CONFIG_PATH="$PWD/qusa/utils/config.yaml"
export QUSA_DATA_ROOT="$PWD/data"
```

`QUSA_DATA_ROOT` overrides raw, processed, figure, prediction, and report directories. `QUSA_DATABASE_PATH` overrides the SQLite ledger path; by default it is `data/predictions/qusa.sqlite3`. The packaged defaults disable plotting, email, and AI reporting. To enable plots, install the research extra and set `backtest.save_plots` or `clustering.save_plots`. To enable reports, set `reporting.enabled` and run a local Ollama server with the configured model; the reporter uses its HTTP API through `requests`.

## Use The App

The Flask page lists tickers from local `data/raw/{TICKER}_history.csv` files. Generate features from that history, or select **Fetch latest data** to request a new provider bar. A prediction also needs a trained bundle at `saved_models/{ticker_lowercase}_model.pkl` and `data/processed/{TICKER}_processed.csv`. The page shows generation-time readiness and current data freshness separately. Latest and history show only completed predictions.

For a local research workflow:

```sh
python scripts/fetch_data.py -ticker UPRO --days 504
nocturne features UPRO
nocturne research UPRO
nocturne predict UPRO
```

The fetch script also accepts `--start YYYY-MM-DD --end YYYY-MM-DD`. The installed CLI accepts `features`, `research`, `predict`, and `cluster`, with one ticker and an optional `--config`; `features` and `predict` also accept `--fetch`. The legacy script flags remain available:

```sh
python scripts/run_FE_pipeline.py -ticker UPRO
python scripts/run_model_pipeline.py -ticker UPRO AAPL --volatility 3
python scripts/run_clustering.py -ticker UPRO
python scripts/model_prediction.py -ticker UPRO --fetch
```

Training writes the current model bundle and a run-owned snapshot. Enabled numerical outputs, reports, cluster statistics, and figures have run-owned paths recorded in SQLite. Plots are noninteractive. Optional report failures appear separately from completed numerical phases. Flask does not expose training or email controls; notification delivery is an explicit worker operation.

## Verify

```sh
python -m pip install '.[research,test]'
python -m pip check
python -m pytest -q --cov=qusa --cov-fail-under=40
python -m compileall -q app.py web_app qusa scripts
```

`GET /health` and `GET /api/health` return `{"status":"healthy"}`. `/api/tickers` and `/api/predictions/history` may be empty on a fresh install. [Sprint 7 verification](docs/sprint7-verification.md) describes the clean-wheel, synthetic Flask/SQLite, and browser checks. No live provider, SMTP, or Ollama service is needed for that suite.

## History And Backup

SQLite is the source of truth for runs and predictions. The CSV tool imports legacy history idempotently by source fingerprint and can export successful predictions:

```sh
python scripts/migrate_prediction_history.py --database data/predictions/qusa.sqlite3 --csv old-history.csv --dry-run
python scripts/migrate_prediction_history.py --database data/predictions/qusa.sqlite3 --csv old-history.csv
python scripts/migrate_prediction_history.py --database data/predictions/qusa.sqlite3 --csv history-export.csv --export
```

Keep the original CSV. See [operations](docs/operations.md) for backup, restore, and rollback steps. The importer preserves unknown legacy risk fields rather than inferring values.

## Documentation

- [Operations](docs/operations.md)
- [Sprint 7 verification](docs/sprint7-verification.md)
- [Historical Flask conversion plan](flask_conversion_execution_plan.md) and [archived roadmap](docs/archive/roadmap.md)

GitHub Pages serves static documentation from `docs/`; it does not run the Flask backend. For installation problems, check the active virtual environment, `QUSA_CONFIG_PATH`, and the configured data/model paths before running `flask --app app run`.
