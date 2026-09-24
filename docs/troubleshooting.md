---
title: Troubleshooting
nav_order: 7
---

# Troubleshooting

| Symptom | Check |
|:--|:--|
| Dashboard has no tickers | Confirm `data/raw/{TICKER}_history.csv` exists under the configured raw directory. A fresh install has no data. |
| Feature generation fails | Check local history format and configured paths. `fetch_latest` requires `POLYGON_API_KEY` and network access. |
| Prediction returns 404 | Train a bundle with `nocturne research TICKER`; confirm the model and processed CSV exist at the configured paths. Existing saved history remains available. |
| Result says stale | Compare `readiness.feature_as_of`, `readiness.expected_session`, and current `freshness`. Regenerate features from a completed market session when appropriate. |
| Volatility says unavailable | The latest processed row is missing a valid non-negative `atr_pct`. It is not equivalent to passing the filter. |
| Ticker busy (409) | Another operation holds that ticker's local lock. Wait for it to finish and retry. |
| Database or artifacts missing after restore | Restore the same point-in-time SQLite backup and all referenced artifact directories; check recorded absolute paths if the workspace moved. |
| `flask` command not found | Activate the intended virtual environment and reinstall with `python -m pip install .`. |
| Site opens but search or images fail | The docs project uses `/Nocturne/`; build with the configured `baseurl` and verify the Pages Actions deployment. |

See [Quickstart]({{ '/quickstart/' | relative_url }}) for setup, [Prediction semantics]({{ '/prediction-semantics/' | relative_url }}) for result interpretation, and [Operations]({{ '/operations/' | relative_url }}) for recovery.
