---
title: Research Workflows
nav_order: 4
---

# Research workflows

Training, evaluation, backtesting, clustering, and live data fetching are **explicit CLI workflows**, not dashboard buttons or background schedules.

```sh
python scripts/fetch_data.py -ticker UPRO --days 504
nocturne features UPRO
nocturne research UPRO
nocturne predict UPRO
nocturne cluster UPRO
```

`fetch_data.py` also accepts `--start YYYY-MM-DD --end YYYY-MM-DD`; fetching needs `POLYGON_API_KEY`. `features` and `predict` accept `--fetch`. All installed commands accept `--config /path/to/config.yaml`. Run `python -m pip install '.[research]'` for optional Matplotlib figures. The base installation can run numeric workflows without that plotting dependency.

The research result contains phase statuses (`succeeded`, `skipped`, or `failed`) and metrics. The SQLite run ledger records run-owned model snapshots and enabled numerical outputs. With matching configuration, these include metrics JSON, backtest CSV and plot, and cluster statistics/figures under local configured directories. The current model bundle is replaced only after a new snapshot has been written. Consult run detail to locate recorded artifacts rather than assuming a single output path.

AI reports are **disabled by default**. To enable them, set `reporting.enabled` and select report types in configuration; a local Ollama server and configured model are then required. Report generation is optional and its failure is reported separately from numerical phase success. No result promises that a strategy is profitable or a probability is calibrated.

The current Flask API exposes feature generation and prediction, but **does not** expose training, backtesting, clustering, watchlists, scheduled jobs, email controls, or a report browser. The app is not prepared for unauthenticated network deployment. See [Operations]({{ '/operations/' | relative_url }}) for artifact backup.
