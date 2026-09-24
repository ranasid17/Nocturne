# Nocturne Contributor Guide

Nocturne is a Python 3.9+ local Flask application with explicit CLI research workflows. [README.md](README.md) is the user setup guide; [docs/operations.md](docs/operations.md) covers backups and restore.

## Install And Check

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q --cov=qusa --cov-fail-under=40
flask --app app run
```

`requirements.txt` installs the package with research and test extras. Runtime dependencies are declared in `pyproject.toml`. The base package needs no Matplotlib or Ollama SDK. It starts with no provider credentials; fetching data requires `POLYGON_API_KEY`.

## Execution Paths

- `web_app/api.py` validates Flask requests and calls importable services in `qusa/services/`.
- `qusa/cli.py` provides the installed `nocturne` command. `scripts/run_model_pipeline.py` and `scripts/run_clustering.py` are thin legacy wrappers preserving their flags and exit codes.
- `qusa/model/` owns numerical training, evaluation, backtesting, and prediction. Keep probability-class interpretation consistent through `qusa/model/probability.py`.
- `qusa/storage/runs.py` owns run transactions and successful-only prediction reads. `qusa/storage/history_csv.py` owns CSV parsing and writing for legacy migration.
- `qusa/services/research_outputs.py` and `clustering_outputs.py` save run-owned artifacts; plotting is optional and noninteractive.
- `qusa/utils/settings.py` selects explicit YAML, `QUSA_CONFIG_PATH`, or packaged defaults, in that order. `QUSA_DATA_ROOT` and `QUSA_DATABASE_PATH` override local storage paths.

Run `nocturne --help` for supported commands. For a local sequence, fetch data with `python scripts/fetch_data.py -ticker UPRO --days 504`, then run `nocturne features UPRO`, `nocturne research UPRO`, and `nocturne predict UPRO`. No Flask page load performs research or sends notifications. Reports use the local Ollama HTTP endpoint only when enabled in configuration.

## Data And Tests

Raw history is `data/raw/{TICKER}_history.csv`, processed features are `data/processed/{TICKER}_processed.csv`, and the current model is `saved_models/{ticker_lowercase}_model.pkl` under default paths. The SQLite ledger is normally `data/predictions/qusa.sqlite3`. Do not commit local data, model bundles, logs, `.env`, or database files.

`tests/test_mvp_reliability.py` and `tests/test_sprint8_parity.py` cover service behavior, run state, artifacts, and CLI parity with synthetic data. The installed-wheel and browser checks are in `tests/installed_workflow.py` and `tests/web_installed.cjs`. Keep external Polygon, SMTP, and LLM boundaries mocked in tests; preserve scientific calculations and result schemas when refactoring.
