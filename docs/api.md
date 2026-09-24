---
title: API
nav_order: 5
---

# Local API

These endpoints are served by the local Flask process at `http://127.0.0.1:5000`. They are synchronous and unauthenticated. Do not expose this server on a public interface.

| Method | Path | Purpose |
|:--|:--|:--|
| GET | `/api/health` | Returns `{"status":"healthy"}`. |
| GET | `/api/tickers` | Returns `{"success":true,"tickers":[...]}` from local raw history filenames. |
| POST | `/api/pipeline/run` | Generate features for a ticker; `fetch_latest` defaults to false. |
| POST | `/api/predictions/run` | Run prediction; optional `fetch_latest` and `volatility`. |
| GET | `/api/predictions/latest?ticker=UPRO` | Latest completed prediction plus current freshness, or `null`. |
| GET | `/api/predictions/history?ticker=UPRO&limit=20&offset=0` | Saved predictions, newest first, plus `next_offset`. Limit defaults to 50 and cannot exceed 100. |
| GET | `/api/runs/{run_id}` | Run metadata and recorded artifacts; 404 for an unknown ID. |

For a ticker with prepared data and a trained bundle:

```sh
curl -sS http://127.0.0.1:5000/api/health
curl -sS -X POST http://127.0.0.1:5000/api/pipeline/run \
  -H 'Content-Type: application/json' -d '{"ticker":"UPRO","fetch_latest":false}'
curl -sS -X POST http://127.0.0.1:5000/api/predictions/run \
  -H 'Content-Type: application/json' -d '{"ticker":"UPRO","fetch_latest":false}'
curl -sS 'http://127.0.0.1:5000/api/predictions/latest?ticker=UPRO'
```

Successful prediction responses include `success`, `ticker`, `run_id`, `prediction`, `readiness`, `freshness`, and local artifact paths. This is a **subset of the saved-latest response** from the synthetic browser fixture; IDs and timestamps are omitted:

```json
{
  "success": true,
  "prediction": {
    "ticker": "UPRO",
    "date": "2025-08-13 00:00:00",
    "probability_up": 0.0,
    "confidence": "HIGH",
    "volatility_state": "pass",
    "readiness_status": "stale",
    "freshness": {"status": "stale", "feature_as_of": "2025-08-13"}
  }
}
```

Actual values depend on the trained model and run date. Fields may include additional metadata; clients should read the fields they need. Invalid bodies and pagination return 400 with `success:false` and `code:"invalid_request"`; simultaneous work on the same ticker may return 409 `ticker_busy`; a missing model or processed artifact returns 404 `artifact_not_found`; service failures return 500 with a stable operation code. No research or scheduling HTTP endpoint is supported.
