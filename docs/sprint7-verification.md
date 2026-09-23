# Sprint 7 verification

Issue #93 covers portable installation, prediction integrity, research outputs, and current result freshness. The app uses local SQLite and synchronous Flask requests.

## Reproduce the installed check

From the repository root, build a wheel and install it in a new virtual environment:

```sh
python -m pip install build
python -m build --wheel --outdir dist
python -m venv /tmp/nocturne-base
/tmp/nocturne-base/bin/python -m pip install dist/nocturne_qusa-*.whl
/tmp/nocturne-base/bin/python -m pip check
mkdir -p /tmp/nocturne-test
cd /tmp/nocturne-test
/tmp/nocturne-base/bin/python /path/to/Nocturne/tests/installed_workflow.py "$PWD" --base-only
```

This generates synthetic OHLCV and processed features in the temporary directory, trains a model, runs real Flask prediction and feature endpoints, checks SQLite history after an app restart, tests a missing-model failure, exports dates, and checks the installed CLI. The `--base-only` check confirms optional Matplotlib and Ollama packages are absent. The fixture leaves no generated data in the repository.

For browser verification, start the installed app with `QUSA_CONFIG_PATH=/tmp/nocturne-test/config.yaml /tmp/nocturne-base/bin/python -m flask --app app run --port 5053` from the temporary directory. Install Playwright 1.63 and Chromium, then run `node /path/to/Nocturne/tests/web_installed.cjs http://127.0.0.1:5053` with `NODE_PATH` pointing at Playwright's `node_modules`. The browser test exercises saved and newly generated predictions, history, reload, failed inference, race handling, and desktop/mobile layouts. CI stores screenshots as an artifact.

The source-tree suite runs with `python -m pytest -q --cov=qusa --cov-fail-under=40`. Its synthetic research tests check run-owned backtest plots, metrics, clustered statistics and figures, disabled outputs, and optional report failures.

## Limits of this check

No live Polygon request, SMTP delivery, Ollama response, or market forecasting performance is asserted. The browser and installed checks validate application plumbing and persistence using deterministic synthetic data. The local Flask server is for loopback use; GitHub Pages hosts documentation only.
