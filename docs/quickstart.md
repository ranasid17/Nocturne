---
title: Quickstart
nav_order: 2
---

# Quickstart

## Install and launch

Python 3.9 or newer is required:

```sh
git clone https://github.com/ranasid17/Nocturne.git
cd Nocturne
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
flask --app app run
```

Open `http://127.0.0.1:5000/`. On Windows use `py -3 -m venv .venv` and activate `.venv\Scripts\Activate.ps1`. The local server has **no authentication**; leave it on the default loopback interface. `GET /api/health` should return `{"status":"healthy"}` even before data is loaded.

## Add local data

The dashboard lists tickers with files named `data/raw/{TICKER}_history.csv`. With existing local history, select a ticker and generate features. The `Fetch latest data` option contacts Polygon and needs `POLYGON_API_KEY` in the environment or local `.env`; it is not required for offline use. Training is an explicit CLI action:

```sh
nocturne features UPRO
nocturne research UPRO
nocturne predict UPRO
```

The research command writes a current bundle at `saved_models/upro_model.pkl` and a run-owned snapshot. A prediction requires that bundle and `data/processed/UPRO_processed.csv`. Without either artifact, the request cannot produce a new prediction; previously saved history remains readable.

## Configuration

The CLI's explicit `--config` wins, followed by `QUSA_CONFIG_PATH`, then packaged defaults. A custom YAML's relative paths resolve from that YAML's directory; packaged-default paths resolve from the working directory. `QUSA_DATA_ROOT` overrides the data directories and `QUSA_DATABASE_PATH` overrides the SQLite ledger path. The repo's `qusa/utils/config.yaml` is **not** loaded unless selected explicitly. Local artifacts and secrets should stay outside Git.

For a reproducible synthetic install and browser check, see [Sprint 7 verification]({{ '/sprint7-verification/' | relative_url }}). The [prediction guide]({{ '/prediction-semantics/' | relative_url }}) explains the result fields before you interpret a signal.
