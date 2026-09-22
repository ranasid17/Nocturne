# Nocturne: Quantitative Overnight Regime Analysis & Alpha Discovery

A Python-based quantitative analysis framework for feature engineering, signal identification, and pattern discovery 
in US equity markets. **Nocturne** focuses on overnight price movements (close-to-open gaps), technical indicator analysis, 
and unsupervised clustering to identify trading regimes.

## Overview

Nocturne provides a comprehensive toolkit for analyzing stock market data through:

- **Feature Engineering**: Calculate technical indicators (RSI, ATR, volume metrics) and calendar-based features
- **Overnight Analysis**: Identify and analyze overnight price gaps and abnormal movements
- **Clustering Analysis**: Discover market regimes and trading patterns using K-Means and DBSCAN
- **Predictive Modeling**: Train decision tree models to predict overnight price direction
- **Backtesting**: Evaluate trading strategies with realistic transaction costs

The framework is designed for researchers and quantitative analysts who want to explore pattern-based trading 
signals beyond traditional technical analysis.

## Roadmap

The product roadmap for the small-team signal app is available in [docs/index.md](docs/index.md). The static review site should be published with GitHub Pages from the `main` branch `/docs` folder.

## Key Features

### Feature Engineering
- **Technical Indicators**: RSI, ATR, Volume ratios, 52-week high/low proximity, and momentum metrics.
- **Overnight Calculations**: Close-to-open gaps, abnormal movement z-scores, and gap pattern statistics.
- **Calendar Features**: Day of week, month of year, and month start/end effects.

### Unified History & Deconfliction
- **Consolidated Storage**: Maintains a single `{TICKER}_history.csv` source of truth for each ticker.
- **Automated Deconfliction**: Automatically merges new fetches with existing data, removes duplicates, and archives fragmented files.
- **Standardized CLI**: Unified `-ticker` flag across all scripts for a consistent user experience.

### Clustering Analysis
- Unsupervised learning (K-Means/DBSCAN) to group trading days into interpretable regimes.
- PCA-based visualization and feature importance ranking by cluster separation.

### Machine Learning & Backtesting
- Decision tree classifiers for overnight direction prediction with high-confidence filtering.
- Comprehensive backtesting engine with realistic costs and Sharpe/Alpha/Drawdown metrics.
- **AI-Powered Reporting**: Automated report generation using local LLMs (via Ollama).

## Getting Started

### Prerequisites

- Python 3.9+ (CI uses Python 3.11)
- Polygon.io API key for fetching and feature generation; the dashboard and health checks can start without it
- Ollama (optional, for AI-powered reports)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ranasid17/Nocturne.git
cd Nocturne
```

2. **Create an isolated environment and install dependencies** (macOS/Linux):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell, create it with `py -3 -m venv .venv` and activate with `.venv\Scripts\Activate.ps1`, then run the same pip commands. Run all commands below from the repository root with this environment active.

3. **Set up environment variables**:
For a new checkout, create the local file from the placeholders:
```bash
cp .env.example .env
```

On PowerShell use `Copy-Item .env.example .env`. Do not overwrite an existing `.env`. Edit the new file to set `POLYGON_API_KEY`. Flask loads it at startup; CLI workflows load it when reading configuration. Existing exported environment variables take precedence. `.env` is ignored by Git; `.env.example` contains no credentials.

`QUSA_SMTP_USER` and `QUSA_SMTP_PASSWORD` are optional defaults for dashboard email notifications. You can also enter the SMTP username and password directly in the dashboard for the current Streamlit session. For Gmail, use an app password rather than your normal account password.

4. **Configure the project**:
Keep `qusa/utils/config.yaml` unchanged unless you intentionally maintain a local override. Set `QUSA_DATA_ROOT` to put raw, processed, figure, prediction, and report files beneath one portable directory:

```bash
export QUSA_DATA_ROOT="$PWD/data"
```

Set `QUSA_CONFIG_PATH` to use a separate YAML file. Set `QUSA_DATABASE_PATH` to place the local SQLite run history elsewhere; by default it is `data/predictions/qusa.sqlite3` (or the equivalent beneath `QUSA_DATA_ROOT`). Exported values take precedence; neither setting rewrites an existing configuration file.

Set `reporting.enabled: false` unless you have a local Ollama server and the model specified in `reporting.llm.model`. AI reporting is optional; it is not required to serve the Flask dashboard.

### Start the Flask Dashboard

```bash
flask --app app run
```

Open <http://127.0.0.1:5000>. If port 5000 is occupied, use `flask --app app run --port 5050` and open <http://127.0.0.1:5050>. Stop the server with Ctrl+C. The equivalent interpreter-specific command is `python -m flask --app app run`.

The page supports ticker entry, feature generation, predictions with an optional maximum ATR percentage, and the latest 50 logged predictions. Requests run synchronously, so feature generation may take time. This is a local development app without authentication; keep the default loopback binding. GitHub hosts the source and PRs, not the Flask process. The separate GitHub Pages roadmap site cannot run this Python backend.

A fresh clone has no market data or trained models because generated artifacts are ignored. The page still opens with an empty state. To run predictions, follow the CLI workflow below or supply your own compatible data and trained model:

- Ticker discovery reads `data/raw/{TICKER}_history.csv`.
- Feature generation writes `data/processed/{TICKER}_processed.csv` and currently requires a Polygon API key even when using local history, because the loader initializes the API client.
- Prediction requires `saved_models/{ticker_lowercase}_model.pkl` and processed data. Training is performed through the CLI, not the Flask page.
- Prediction runs and history are stored in the local SQLite database. The legacy CSV setting is no longer written during normal operation.

Paths above assume the relative configuration shown earlier. The **Fetch latest data** option needs a valid Polygon API key and network access. Missing models/data produce an error in the page rather than creating a model automatically.

### Verify the Setup

```bash
python -m pip check
python -m pytest -q
python -m compileall -q app.py web_app qusa scripts
```

With the server running, `curl http://127.0.0.1:5000/health` and `curl http://127.0.0.1:5000/api/health` should return `{"status":"healthy"}`. `/api/tickers` returns a `tickers` list; `/api/predictions/history` returns a `history` list. Both lists may be empty on a new checkout. Tests use temporary data and mocked service calls and do not require live Polygon or Ollama access.

Optional browser checks use Node.js and Playwright, not application runtime dependencies:

```bash
npm install --prefix /tmp/nocturne-browser-tools playwright
/tmp/nocturne-browser-tools/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/nocturne-browser-tools/node_modules node tests/web_dashboard.cjs http://127.0.0.1:5000
```

These macOS/Linux browser commands expect at least one configured raw-history ticker. Run actions are mocked to avoid data fetching and database writes. Screenshots are saved under `/tmp/qusa-sprint3-*.png`. `CHROME_PATH` may point to an installed Chrome executable instead of installing Chromium.

### Prediction History Migration And Recovery

SQLite is the authoritative local ledger for prediction runs, artifacts, and history. A CSV is supported only as a one-time import source or explicit export. Stop the Flask process before changing or restoring local state.

Preview a legacy CSV import without writing anything:

```bash
python scripts/migrate_prediction_history.py --database data/predictions/qusa.sqlite3 --csv data/predictions/prediction_log.csv --dry-run
```

Run the import once the row count looks right. Re-running the same source is idempotent because QUSA records the source fingerprint and row ordinal.

```bash
python scripts/migrate_prediction_history.py --database data/predictions/qusa.sqlite3 --csv data/predictions/prediction_log.csv
python scripts/migrate_prediction_history.py --database data/predictions/qusa.sqlite3 --csv data/predictions/history-export.csv --export
```

For a consistent backup while the app is stopped, use SQLite's backup command rather than copying only the main file during WAL activity:

```bash
sqlite3 data/predictions/qusa.sqlite3 ".backup data/predictions/qusa-backup.sqlite3"
```

To roll back this migration, stop Flask, retain the original CSV and the SQLite backup, point `QUSA_DATABASE_PATH` at the backup or remove the new database, then restart. The importer never modifies its source CSV, so a fresh database can be rebuilt from that file. Check the active schema version with `sqlite3 data/predictions/qusa.sqlite3 'SELECT * FROM schema_migrations;'`.

### Troubleshooting

- An import error mentioning Flask, Jinja2, or `escape` usually indicates an old global Flask installation. Activate `.venv`, reinstall `requirements.txt`, and launch with `python -m flask --app app run`.
- Empty tickers or missing artifacts: verify every configured path and complete fetch, feature generation, and training below. Model filenames use lowercase tickers.
- `POLYGON_API_KEY ... is required`: fill in `.env` and restart Flask, or export the key before launching the CLI.
- Ollama connection/model errors during training reports: configure the local model or set `reporting.enabled: false`.

### Review Milestones

The conversion is delivered in four PR-sized commits: service extraction and app skeleton (Sprint 1), API routes (Sprint 2), templates and browser interactions (Sprint 3), and deployment/docs verification (Sprint 4). See [the execution plan](flask_conversion_execution_plan.md). Generated data, models, logs, environment files, and caches remain local.

## Recommended Workflow

QUSA follows a standardized CLI pattern. You can use `-ticker` or `--ticker` interchangeably.

### 1. Research & Model Development

Use this workflow to build and evaluate a trading strategy for a ticker.

**Step A: Fetch Historical Data**
Fetch exactly the amount of history you need. Repeated fetches will be automatically deconflicted.
```bash
python scripts/fetch_data.py -ticker UPRO --days 504
```

**Step B: Generate Features**
Processes the consolidated history into engineered indicators.
```bash
python scripts/run_FE_pipeline.py -ticker UPRO
```

**Step C: Train & Backtest**
Trains the model and evaluates performance.
```bash
python scripts/run_model_pipeline.py -ticker UPRO
```

### 2. Live Prediction (One-Step)

Once a model is trained, use this command for live "overnight" prediction tests. The `--fetch` flag automates data retrieval and feature engineering in a single step.

```bash
python scripts/model_prediction.py -ticker UPRO --fetch
```

**Output**:
- Prediction direction (UP/DOWN) and confidence level.
- Durable local run and prediction record in `data/predictions/qusa.sqlite3` by default.

### Dashboard Email Notifications

The Streamlit dashboard can notify recipients after a successful "Generate New Inference" run. Configure SMTP host defaults in `qusa/utils/config.yaml`, then enter the SMTP username, SMTP password, and one or more comma- or semicolon-separated recipients in the dashboard before running inference. If `QUSA_SMTP_USER` and `QUSA_SMTP_PASSWORD` are set in your environment, the dashboard can use them as defaults instead of requiring credentials in the UI.

---

## Detailed Usage Guide

### Fetching Data (`scripts/fetch_data.py`)
- Fetch last $N$ trading days: `python scripts/fetch_data.py -ticker AMZN --days 252`
- Fetch specific range: `python scripts/fetch_data.py -ticker AMZN --start 2024-01-01 --end 2024-05-01`
*Fragmented source files are moved to `data/raw/archive/` after consolidation.*

### Clustering Analysis (`scripts/run_clustering.py`)
Discover market regimes:
```bash
python scripts/run_clustering.py -ticker AMZN
```
**Output**: Elbow curves, PCA cluster plots, and feature heatmaps in `data/figures/`.

### Full Model Pipeline (`scripts/run_model_pipeline.py`)
Supports multiple tickers:
```bash
python scripts/run_model_pipeline.py -ticker AMZN AAPL MSFT
```
**Output**: Trained `.pkl` bundles in `saved_models/` and performance metrics in `data/figures/`.

---

## Data Pipeline Architecture

```
[Polygon.io API]
    ↓
(fetch_data.py) → [data/raw/{ticker}_history.csv] ← (Archive fragmented files)
    ↓
[Feature Engineering Pipeline]
    ↓
[data/processed/{ticker}_processed.csv]
    ↓
[Model Training & Backtesting]
    ↓
[saved_models/{ticker}_model.pkl] → [AI Reports & Figures]
```

## Configuration

Key settings in `qusa/utils/config.yaml`:

```yaml
data:
  start_date: '2023-12-01'  # Legacy default
  end_date: '2025-12-01'    # Legacy default

features:
  rsi_window: 14
  atr_window: 14

model:
  parameters:
    probability_threshold: 0.7  # Cutoff for "High Confidence" predictions

backtest:
  initial_capital: 10000
  transaction_cost: 0.05       # % cost per trade (slippage + commission)
```

## Dependencies

- `pandas`, `numpy` - Data manipulation
- `scikit-learn` - Machine learning and clustering
- `matplotlib` - Visualization
- `requests` - API communication
- `ollama` - Local LLM integration

## Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. Trading stocks involves substantial risk of loss. Past performance does not guarantee future results.
